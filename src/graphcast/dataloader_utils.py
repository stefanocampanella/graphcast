import dataclasses
import logging
from collections.abc import Iterable
from functools import partial
from typing import NamedTuple

import grain.python as grain
import jax
import numpy as np
import xarray as xr
from grain.experimental import pick_performance_config
from grain.python import DatasetIterator, IterDataset
from jax.experimental import multihost_utils
from jax.sharding import Mesh
from jax.sharding import PartitionSpec as P

# Importing grain_utils also patches Grain's debug-mode specs for xarray (in workers too, which import this module).
from graphcast import grain_utils, xarray_jax
from graphcast.data_utils import _get_steps_per_window, extract_inputs_targets_forcings
from graphcast.dataloader import (
  FlatSource,
  InputsTargetsForcings,
  LevelSelection,
  StandardizedAnomalySource,
  _unflatten_sample,
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
  datasource = StandardizedAnomalySource(
    configs.dataset.path,
    levels=_get_level_selection(configs.task.levels),
    end_date=configs.grain_dataset.split_date,
    io_threads=configs.dataset.io_threads,
    window_size=_get_steps_per_window(
      dataset=dataset,
      input_duration=task_config.input_duration,
      target_lead_times=configs.grain_dataset.target_lead_times,
    ),
  )

  map_dataset = (
    grain.MapDataset.source(FlatSource(datasource))
    .map(_unflatten_sample)
    .repeat(num_epochs=None)
    .shuffle(seed=seed)
    .slice(slice(jax.process_index(), None, jax.process_count()))
  )

  local_batch_size = configs.grain_dataset.local_batch_size
  if local_batch_size > 1:
    logger.info(f"Batching {local_batch_size} samples per device.")
    map_dataset = map_dataset.batch(
      batch_size=local_batch_size,
      drop_remainder=True,
      batch_fn=partial(xr.concat, dim="batch"),
    )

  # Extract once per batch, before prefetch/device transfer.
  map_dataset = map_dataset.map(
    partial(
      extract_inputs_targets_forcings,
      input_variables=task_config.input_variables,
      target_variables=task_config.target_variables,
      forcing_variables=task_config.forcing_variables,
      levels=None,
      input_duration=task_config.input_duration,
      target_lead_times=configs.grain_dataset.target_lead_times,
      to_jax=False,
    )
  )
  return map_dataset


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


def get_first_sample_and_reset(
  dataset_iterator: DatasetIterator[InputsTargetsForcings],
) -> InputsTargetsForcings:
  state = dataset_iterator.get_state()
  sample = next(dataset_iterator)
  dataset_iterator.set_state(state)
  return sample


def next_batch_on_device(
  iterator: InputsTargetsForcingsIterator,
  device_mesh: Mesh,
  mp_prefetch: bool = False,
) -> InputsTargetsForcings:
  if mp_prefetch:
    try:
      batch_on_host = next(iterator)
      batch = xarray_jax.make_array_from_process_local_data(batch_on_host, mesh=device_mesh, spec=P("batch"))
      batch = jax.block_until_ready(batch)
    finally:
      # Explicitly delete batch_on_host to trigger shared memory release in Grain.
      # It must be done after checkpointing, otherwise current SharedMemoryArrays could be released.
      del batch_on_host
  else:
    batch_on_host = next(iterator)
    batch = xarray_jax.make_array_from_process_local_data(batch_on_host, mesh=device_mesh, spec=P("batch"))

  return batch


# As the Grain pipeline calls extract_inputs_targets_forcings, which removes the datetime coordinate, the following
# isn't needed anymore.
def reshard_data(dataset, sharding, datetime_coord_name="time"):
  """Utility to reshard datasets with datetime coordinates."""
  datetime_coord = dataset[datetime_coord_name]
  datetime_coord.data = datetime_coord.data.astype("datetime64[s]").astype(np.int64)
  dataset = dataset.drop(datetime_coord_name)
  dataset = dataset.map(
    lambda da: jax.tree_util.tree_map(
      lambda xs: jax.make_array_from_process_local_data(sharding=sharding, local_data=xs),
      da,
    )
  )
  dataset = xarray_jax.Dataset(
    data_vars={var: (data.dims, data.data) for var, data in dataset.data_vars.items()},
    coords=dataset.coords,
    jax_coords={datetime_coord_name: datetime_coord},
    attrs=dataset.attrs,
  )
  datetime_coordinate = multihost_utils.process_allgather(datetime_coord)
  # In the case of single-host setups it might happen that datetime is still a numpy array
  datetime_coordinate = xarray_jax.unwrap_data(datetime_coordinate, require_jax=False)
  datetime_coordinate = np.asarray(datetime_coordinate).astype("datetime64[s]")
  dataset = dataset.drop(datetime_coord_name)
  dataset = dataset.assign_coords({datetime_coord_name: (datetime_coord_name, datetime_coordinate)})
  return dataset
