import dataclasses
import logging
from collections.abc import Iterable
from functools import partial
from typing import NamedTuple

import grain.python as grain
import jax
import numpy as np
import pandas as pd
import xarray as xr
from grain.experimental import pick_performance_config
from grain.python import DatasetIterator, IterDataset
from jax.sharding import Mesh
from jax.sharding import PartitionSpec as P

# Importing grain_utils also patches Grain's debug-mode specs for xarray (in workers too, which import this module).
from graphcast import grain_utils, xarray_jax
from graphcast.data_utils import _get_steps_per_window, extract_inputs_targets_forcings
from graphcast.dataloader import (
  InputsTargetsForcings,
  LevelSelection,
  StandardizedAnomalySource,
  open_standardized_anomalies,
  select_levels,
)
from graphcast.model import TaskConfig
from graphcast.training_config import Levels, LevelSliceConfig, MaskConfig, TrainingConfig

logger = logging.getLogger(__name__)

InputsTargetsForcingsIterator = DatasetIterator[InputsTargetsForcings]
InputsTargetsForcingsIterDataset = IterDataset[InputsTargetsForcings]


def _get_level_selection(levels: Levels) -> LevelSelection:
  if isinstance(levels, LevelSliceConfig):
    return slice(levels.start, levels.stop, levels.step)
  if levels == "all":
    return slice(None)
  return list(levels)


def get_dataset(configs: TrainingConfig) -> xr.Dataset:
  """Open the standardized anomaly store lazily, with the task levels and all timesteps."""
  logger.info(f"Opening standardized anomalies from {configs.dataset.path}")
  return open_standardized_anomalies(
    configs.dataset.path, _get_level_selection(configs.task.levels), io_threads=configs.dataset.io_threads
  )


class Grid(NamedTuple):
  """The model grid: latitudes, longitudes and the boolean (lat, lon) mask of the grid points that are nodes."""

  lat: np.ndarray
  lon: np.ndarray
  mask: np.ndarray


def get_tendency_std(configs: TrainingConfig) -> xr.Dataset:
  """Load the tendency_std written by graphcast compute-tendency-std, with the task levels.

  It is 0 where a series is constant (including outside the masks of the variables): those points are excluded from
  the loss (`get_loss_masks`) and predicted as their input. Missing values are an error.
  """
  logger.info(f"Loading tendency_std from {configs.tendency_std.path}")
  read_configs = configs.tendency_std.read.model_dump(exclude_unset=True)
  tendency_std = xr.open_dataset(configs.tendency_std.path, **read_configs)
  tendency_std = select_levels(tendency_std, _get_level_selection(configs.task.levels)).load()
  for name, values in tendency_std.data_vars.items():
    if not np.isfinite(values.values).all() or (values.values < 0).any():
      raise ValueError(
        f"tendency_std of {name} has missing or negative values: write it with graphcast compute-tendency-std"
      )
  return tendency_std


def get_loss_masks(tendency_std: xr.Dataset, grid: Grid, target_variables: Iterable[str]) -> xr.Dataset:
  """Points of each target variable (and level) that count in the loss: grid nodes whose series is not constant.

  Constant points (tendency_std 0) include those outside the masks of the variables (0 in the standardized store),
  e.g. land below the surface of ocean columns.
  """
  missing = [name for name in target_variables if name not in tendency_std]
  if missing:
    raise ValueError(f"No tendency_std for the target variables {missing}")
  for dim, values in (("lat", grid.lat), ("lon", grid.lon)):
    if not np.array_equal(tendency_std[dim].values, values):
      raise ValueError(f"tendency_std and the grid mask have different {dim} coordinates")
  grid_mask = xr.DataArray(grid.mask, dims=("lat", "lon"), coords={"lat": grid.lat, "lon": grid.lon})
  # For volume variables (thetao, so, ...) `tendency_std[name] > 0` is a (level, lat, lon) array, so each level gets
  # its own mask; `&` broadcasts the 2D grid mask over the levels. Surface variables get (lat, lon) masks.
  return xr.Dataset({name: (tendency_std[name] > 0) & grid_mask for name in target_variables})


def get_grid(configs: MaskConfig) -> Grid:
  """Load the grid from the boolean (lat, lon) mask written by graphcast download-mask."""
  logger.info(f"Loading the grid mask from {configs.path}")
  mask = xr.open_dataarray(configs.path, engine="zarr", chunks=None).load()
  if mask.dtype != bool or set(mask.dims) != {"lat", "lon"}:
    raise ValueError(
      f"Expected a boolean (lat, lon) mask in {configs.path}, found {mask.dtype} {mask.dims}: "
      "write it with graphcast download-mask"
    )
  return Grid(lat=mask["lat"].to_numpy(), lon=mask["lon"].to_numpy(), mask=mask.transpose("lat", "lon").to_numpy())


def get_task_config(
  configs: TrainingConfig,
  dataset: xr.Dataset | None = None,
) -> TaskConfig:
  if dataset is not None:
    levels = tuple(int(level) for level in dataset.level.values)
  elif isinstance(configs.task.levels, tuple):
    levels = configs.task.levels
  else:
    raise ValueError("A selected dataset is required to resolve task.levels")
  return TaskConfig(**{**configs.task.model_dump(), "levels": levels})


@dataclasses.dataclass(frozen=True)
class DatasetStream:
  """An IterDataset with the Grain options it was built with (e.g. num_workers, which checkpoints depend on)."""

  dataset: InputsTargetsForcingsIterDataset
  options: grain_utils.GrainOptions


def get_dataset_stream(
  configs: TrainingConfig,
  dataset: xr.Dataset,
  seed: int,
  grain_runtime: grain_utils.GrainRuntimeConfig | None = None,
) -> DatasetStream:
  """Return the training stream of the standardized anomaly store, opened with the task levels (get_dataset).

  Grain options come from grain_runtime (GRAPHCAST_GRAIN_* environment variables by default), unless
  grain_dataset.pick_performance_config is set.
  """
  grain_runtime = grain_utils.GrainRuntimeConfig() if grain_runtime is None else grain_runtime
  map_dataset = get_map_dataset(configs=configs, dataset=dataset, seed=seed)
  options = get_grain_options(configs, map_dataset, grain_runtime)
  return DatasetStream(to_dataset_iterator(map_dataset, options), options)


# TODO: current implementation cannot handle target_lead_times which are slices, as needed for autoregressive rollouts.
def get_map_dataset(
  configs: TrainingConfig,
  dataset: xr.Dataset,
  seed: int,
) -> grain.MapDataset:
  """Build the shuffled, batched and extracted MapDataset of the store, opened with the task levels (get_dataset)."""
  task_config = get_task_config(configs, dataset)

  # Split the dataset "causally": train on dates up to split_date, later ones are held out for evaluation. The source
  # reopens the store in each Grain worker, with the same levels and split.
  datasource = _get_source(configs, dataset, task_config, end_date=configs.grain_dataset.split_date)

  map_dataset = (
    grain.MapDataset.source(_FlatSource(datasource))
    .map(_unflatten_sample)
    .repeat(num_epochs=None)
    .shuffle(seed=seed)
    .slice(slice(jax.process_index(), None, jax.process_count()))
  )
  return _batch_and_extract(map_dataset, configs, task_config)


def get_init_sample(configs: TrainingConfig, dataset: xr.Dataset) -> InputsTargetsForcings:
  """The first training window, with a batch of one: model parameters depend only on the shapes of the variables, not
  on their values or the batch size."""
  task_config = get_task_config(configs, dataset)
  window = _get_source(configs, dataset, task_config, end_date=configs.grain_dataset.split_date)[0]
  return _extract(window, configs, task_config)


def _get_source(
  configs: TrainingConfig,
  dataset: xr.Dataset,
  task_config: TaskConfig,
  start_date: str | None = None,
  end_date: str | None = None,
) -> StandardizedAnomalySource:
  """Windows of the store between start_date and end_date (included), as long as one input and target period."""
  return StandardizedAnomalySource(
    configs.dataset.path,
    levels=_get_level_selection(configs.task.levels),
    start_date=start_date,
    end_date=end_date,
    io_threads=configs.dataset.io_threads,
    window_size=_get_steps_per_window(
      dataset=dataset,
      input_duration=task_config.input_duration,
      target_lead_times=configs.grain_dataset.target_lead_times,
    ),
  )


def get_validation_dataset(configs: TrainingConfig, dataset: xr.Dataset) -> grain.MapDataset:
  """Held-out batches after grain_dataset.split_date: validation.num_samples windows, evenly spaced, in order.

  The windows depend only on the store and the config (not on the seed or mesh), so validation losses of different
  runs are comparable. Each process gets every process_count-th window; the number of windows is rounded up to a
  multiple of the global batch size.
  """
  if configs.validation is None:
    raise ValueError("No [validation] section in the config")
  task_config = get_task_config(configs, dataset)
  times = dataset["time"].values
  held_out = times[times > np.datetime64(pd.Timestamp(configs.grain_dataset.split_date))]
  if held_out.size == 0:
    raise ValueError(f"No timesteps after split_date {configs.grain_dataset.split_date}")
  datasource = _get_source(configs, dataset, task_config, start_date=str(held_out[0]))
  global_batch_size = jax.process_count() * configs.grain_dataset.local_batch_size
  num_samples = -(-configs.validation.num_samples // global_batch_size) * global_batch_size
  if num_samples > len(datasource):
    raise ValueError(f"{num_samples} validation windows requested, the held-out period has {len(datasource)}")
  indices = np.linspace(0, len(datasource) - 1, num_samples).round().astype(int)
  map_dataset = (
    grain.MapDataset.source(_FlatSource(_Subset(datasource, indices.tolist())))
    .map(_unflatten_sample)
    .slice(slice(jax.process_index(), None, jax.process_count()))
  )
  return _batch_and_extract(map_dataset, configs, task_config)


class _FlatSource(grain.RandomAccessDataSource):
  """Expose xarray sample arrays to Grain's multiprocessing transport.

  Grain's shared-memory handling does not use the custom JAX pytree
  registrations for xarray objects. Passing datasets directly would therefore
  hide their arrays from shared-memory transport and restoration. Flattening
  with JAX exposes those arrays as ordinary list elements, while the tree
  definition retains the structure and xarray metadata needed to rebuild them.

  Pair this adapter with `_unflatten_sample` in the pipeline before applying
  transformations that expect datasets. Random access and length are unchanged.
  """

  def __init__(self, source: grain.RandomAccessDataSource):
    """Wrap a source whose samples are registered JAX pytrees."""
    self._source = source

  def __len__(self):
    return len(self._source)

  def __getitem__(self, record_key):
    """Return array leaves and reconstruction metadata for one source sample."""
    # Use JAX's registry to traverse xarray containers, not Grain's traversal.
    return jax.tree_util.tree_flatten(self._source[record_key])

  def __repr__(self) -> str:
    return f"{self.__class__.__name__}(source={self._source!r})"


def _unflatten_sample(flat_sample):
  """Rebuild a sample flattened by `_FlatSource` using its JAX tree definition."""
  leaves, treedef = flat_sample
  # The tree definition restores dataset structure and coordinates around leaves.
  return jax.tree_util.tree_unflatten(treedef, leaves)


class _Subset(grain.RandomAccessDataSource):
  """The records of a source at the given indices, in their order."""

  def __init__(self, source: grain.RandomAccessDataSource, indices: list[int]):
    self._source = source
    self._indices = indices

  def __len__(self) -> int:
    return len(self._indices)

  def __getitem__(self, record_key):
    return self._source[self._indices[record_key.__index__()]]


def _batch_and_extract(map_dataset: grain.MapDataset, configs: TrainingConfig, task_config: TaskConfig):
  """Batch local_batch_size samples per device, then split them into inputs, targets and forcings."""
  local_batch_size = configs.grain_dataset.local_batch_size
  if local_batch_size > 1:
    logger.info(f"Batching {local_batch_size} samples per device.")
    map_dataset = map_dataset.batch(
      batch_size=local_batch_size,
      drop_remainder=True,
      batch_fn=partial(xr.concat, dim="batch"),
    )

  # Extract once per batch, before prefetch/device transfer.
  return map_dataset.map(partial(_extract, configs=configs, task_config=task_config))


def _extract(window: xr.Dataset, configs: TrainingConfig, task_config: TaskConfig) -> InputsTargetsForcings:
  """Split a window (or batch of windows) into inputs, targets and forcings of the task."""
  return extract_inputs_targets_forcings(
    window,
    input_variables=task_config.input_variables,
    target_variables=task_config.target_variables,
    forcing_variables=task_config.forcing_variables,
    levels=None,
    input_duration=task_config.input_duration,
    target_lead_times=configs.grain_dataset.target_lead_times,
    to_jax=False,
  )


def get_grain_options(
  configs: TrainingConfig,
  grain_dataset: grain.MapDataset,
  grain_runtime: grain_utils.GrainRuntimeConfig,
) -> grain_utils.GrainOptions:
  """Return Grain options from grain_runtime, or picked at runtime if grain_dataset.pick_performance_config is set."""
  if (pick_config := configs.grain_dataset.pick_performance_config) is None:
    return grain_runtime.options()
  else:
    # pick_performance_config needs to measure the size of elements drawn from a DatasetIterator. However, the current
    # implementation of _get_element_size_bytes in grain/_src/python/dataset/transformations/prefetch_autotune.py
    # does not work with xarray datasets. We get around the issue by flattening the dataset beforehand.
    flatten_dataset_iterator = grain_dataset.map(jax.tree_util.tree_flatten).to_iter_dataset()
    logger.info(
      f"Picking dataset iterator performance configurations ({pick_config=}); "
      f"{grain_utils.GRAIN_ENV_PREFIX}* reader and multiprocessing settings are ignored."
    )
    performance_config = pick_performance_config(
      ds=flatten_dataset_iterator,
      **pick_config.model_dump(),
    )
    options = grain_utils.GrainOptions(performance_config.read_options, performance_config.multiprocessing_options)
    logger.warning(
      f"pick_performance_config chose num_workers={options.num_workers} at runtime. Grain's iterator state depends "
      "on num_workers: restoring a checkpoint saved with a different value fails. The choice may differ between "
      f"runs or machines; for resumable training, set {grain_utils.GRAIN_ENV_PREFIX}NUM_WORKERS and disable "
      "pick_performance_config."
    )
    return options


def to_dataset_iterator(
  grain_dataset: grain.MapDataset,
  options: grain_utils.GrainOptions,
) -> InputsTargetsForcingsIterDataset:
  """Convert a MapDataset to an IterDataset with the given reader and (optional) multiprocessing options."""
  read_options, mp_options = options.read_options, options.multiprocessing_options
  logger.info(f"Using read options: {read_options}")
  dataset_iterator = grain_dataset.to_iter_dataset(read_options=read_options)
  if mp_options is not None:
    logger.info(f"Using multiprocessing prefetch with options: {mp_options}")
    # Make arrays contiguous for shared-memory transport and flatten xarray containers so Grain can
    # replace arrays with shared-memory metadata when checkpointing the prefetch buffer.
    dataset_iterator = (
      grain_dataset.map(lambda value: xarray_jax.wrap_data(value, to_jax=False, np_contiguous=True))
      .map(jax.tree_util.tree_flatten)
      .to_iter_dataset(read_options=read_options)
      .mp_prefetch(options=mp_options)
      .map(_unflatten_sample)
    )

  return dataset_iterator


def next_batch_on_device(
  iterator: InputsTargetsForcingsIterator,
  device_mesh: Mesh,
  mp_prefetch: bool = False,
) -> InputsTargetsForcings:
  batch_on_host = next(iterator)
  batch = xarray_jax.make_array_from_process_local_data(batch_on_host, mesh=device_mesh, spec=P("batch"))
  if mp_prefetch:
    # Release this process's references to Grain's shared-memory arrays now, rather than at the next batch. No wait for
    # the transfer is needed: the device arrays keep the host arrays alive until it completes.
    del batch_on_host
  return batch
