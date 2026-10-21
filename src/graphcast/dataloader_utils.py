import dataclasses
import logging
from functools import partial

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
  ARCOTimeseriesSource,
  FlatSource,
  InputsTargetsForcings,
  _unflatten_sample,
)
from graphcast.model import TaskConfig
from graphcast.training_config import Levels, LevelSliceConfig, TrainingConfig

logger = logging.getLogger(__name__)

InputsTargetsForcingsIterator = DatasetIterator[InputsTargetsForcings]
InputsTargetsForcingsIterDataset = IterDataset[InputsTargetsForcings]


def _get_level_selection(levels: Levels) -> list[int] | slice:
  if isinstance(levels, LevelSliceConfig):
    return slice(levels.start, levels.stop, levels.step)
  if levels == "all":
    return slice(None)
  return list(levels)


def _select_task_levels(dataset: xr.Dataset, levels: Levels) -> xr.Dataset:
  """Label original depth positions before selecting the ordered task levels."""
  dataset = dataset.assign_coords(level=("depth", range(dataset.depth.size)))
  dataset = dataset.swap_dims({"depth": "level"})
  dataset = dataset.sel(level=_get_level_selection(levels))
  if dataset.sizes["level"] == 0:
    raise ValueError(f"Level selection {levels!r} resolved to zero levels")
  return dataset


def get_dataset(configs: TrainingConfig) -> xr.Dataset:
  logger.info(f"Loading dataset from {configs.dataset.path}")
  read_configs = configs.dataset.read.model_dump(exclude_unset=True)
  dataset = xr.open_dataset(configs.dataset.path, **read_configs)
  dataset = _select_task_levels(dataset, configs.task.levels)
  return dataset


def get_climatology(configs: TrainingConfig) -> xr.Dataset:
  logger.info(f"Loading climatology from {configs.climatology.path}")
  read_configs = configs.climatology.read.model_dump(exclude_unset=True)
  climatology = xr.open_dataset(configs.climatology.path, **read_configs)
  climatology = _select_task_levels(climatology, configs.task.levels)
  return climatology


def get_anomaly_std(configs: TrainingConfig) -> xr.Dataset:
  logger.info(f"Loading anomaly_std from {configs.anomaly_std.path}")
  read_configs = configs.anomaly_std.read.model_dump(exclude_unset=True)
  anomaly_std = xr.open_dataset(configs.anomaly_std.path, **read_configs)
  anomaly_std = _select_task_levels(anomaly_std, configs.task.levels)
  return anomaly_std


def get_tendency_std(configs: TrainingConfig) -> xr.Dataset:
  logger.info(f"Loading tendency_std from {configs.tendency_std.path}")
  read_configs = configs.tendency_std.read.model_dump(exclude_unset=True)
  tendency_std = xr.open_dataset(configs.tendency_std.path, **read_configs)
  tendency_std = _select_task_levels(tendency_std, configs.task.levels)
  tendency_std = tendency_std.fillna(value=1.0)
  tendency_std = tendency_std.clip(min=1e-16)
  return tendency_std


def get_surface_mask(configs: TrainingConfig) -> xr.DataArray:
  logger.info(f"Loading mask from {configs.mask.path}")
  mask = xr.open_dataset(
    configs.mask.path, **configs.mask.read.model_dump(exclude_unset=True)
  )
  mask_name = configs.mask.name
  mask = mask[mask_name]
  mask = mask.assign_coords(level=("depth", range(mask.depth.size)))
  mask = mask.sel(level=0, drop=True)
  return mask


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


def get_dataset_iterators(
    configs: TrainingConfig,
    dataset: xr.Dataset,
    climatology: xr.Dataset,
    anomaly_std: xr.Dataset,
    seed: int,
    grain_runtime: grain_utils.GrainRuntimeConfig | None = None,
) -> tuple[DatasetStream, DatasetStream]:
  """Return independent training and test streams, in that order.

  Each stream retains its own prefetch/performance configuration and checkpoint
  state. The split date remains included in both temporal partitions.
  """
  grain_runtime = (
    grain_utils.GrainRuntimeConfig() if grain_runtime is None else grain_runtime
  )
  streams = []
  for train in (True, False):
    map_dataset = get_map_dataset(
      configs=configs,
      dataset=dataset,
      climatology=climatology,
      anomaly_std=anomaly_std,
      seed=seed,
      train=train,
    )
    options = get_grain_options(configs, map_dataset, grain_runtime)
    streams.append(DatasetStream(to_dataset_iterator(map_dataset, options), options))
  return tuple(streams)


def get_dataset_iterator(
    configs: TrainingConfig,
    dataset: xr.Dataset,
    climatology: xr.Dataset,
    anomaly_std: xr.Dataset,
    seed: int,
    train: bool = True,
    grain_runtime: grain_utils.GrainRuntimeConfig | None = None,
) -> InputsTargetsForcingsIterDataset:
  """Build a stream from artifacts already selected to ordered task levels.

  Grain options come from grain_runtime (GRAPHCAST_GRAIN_* environment variables by default), unless
  grain_dataset.pick_performance_config is set.
  """
  map_dataset = get_map_dataset(
    configs=configs,
    dataset=dataset,
    climatology=climatology,
    anomaly_std=anomaly_std,
    seed=seed,
    train=train,
  )
  grain_runtime = (
    grain_utils.GrainRuntimeConfig() if grain_runtime is None else grain_runtime
  )
  return to_dataset_iterator(
    map_dataset, get_grain_options(configs, map_dataset, grain_runtime)
  )


# TODO: current implementation cannot handle target_lead_times which are slices, as needed for autoregressive rollouts.
def get_map_dataset(
    configs: TrainingConfig,
    dataset: xr.Dataset,
    climatology: xr.Dataset,
    anomaly_std: xr.Dataset,
    seed: int,
    train: bool = True,
) -> grain.MapDataset:
  """Build the shuffled, batched and extracted MapDataset from artifacts selected to ordered task levels."""
  task_config = get_task_config(configs, dataset)

  # Split the dataset "causally"
  split_date = configs.grain_dataset.split_date
  if train:
    dataset = dataset.sel(time=slice(None, split_date))
  else:
    dataset = dataset.sel(time=slice(split_date, None))

  datasource = ARCOTimeseriesSource(
    dataset=dataset,
    climatology=climatology,
    anomaly_std=anomaly_std,
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
    flatten_dataset_iterator = grain_dataset.map(
      jax.tree_util.tree_flatten
    ).to_iter_dataset()
    logger.info(
      f"Picking dataset iterator performance configurations ({pick_config=}); "
      f"{grain_utils.GRAIN_ENV_PREFIX}* reader and multiprocessing settings are ignored."
    )
    performance_config = pick_performance_config(
      ds=flatten_dataset_iterator,
      **pick_config.model_dump(),
    )
    options = grain_utils.GrainOptions(
      performance_config.read_options, performance_config.multiprocessing_options
    )
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
      grain_dataset.map(
        lambda value: xarray_jax.wrap_data(value, to_jax=False, np_contiguous=True)
      )
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


def next_batches_on_device(
    train_iterator: InputsTargetsForcingsIterator,
    test_iterator: InputsTargetsForcingsIterator,
    device_mesh: Mesh,
    mp_prefetch: bool = False,
) -> tuple[InputsTargetsForcings, InputsTargetsForcings]:
  if mp_prefetch:
    try:
      batches_on_host = next(train_iterator), next(test_iterator)
      batches = xarray_jax.make_array_from_process_local_data(
        batches_on_host, mesh=device_mesh, spec=P("batch")
      )
      batch, batch_test = jax.block_until_ready(batches)
    finally:
      # Explicitly delete batch_on_host to trigger shared memory release in Grain.
      # It must be done after checkpointing, otherwise current SharedMemoryArrays could be released.
      del batches_on_host
  else:
    batches_on_host = next(train_iterator), next(test_iterator)
    batch, batch_test = xarray_jax.make_array_from_process_local_data(
      batches_on_host, mesh=device_mesh, spec=P("batch")
    )

  return batch, batch_test


# As the Grain pipeline calls extract_inputs_targets_forcings, which removes the datetime coordinate, the following
# isn't needed anymore.
def reshard_data(dataset, sharding, datetime_coord_name="time"):
  """Utility to reshard datasets with datetime coordinates."""
  datetime_coord = dataset[datetime_coord_name]
  datetime_coord.data = datetime_coord.data.astype("datetime64[s]").astype(np.int64)
  dataset = dataset.drop(datetime_coord_name)
  dataset = dataset.map(
    lambda da: jax.tree_util.tree_map(
      lambda xs: jax.make_array_from_process_local_data(
        sharding=sharding, local_data=xs
      ),
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
  dataset = dataset.assign_coords(
    {datetime_coord_name: (datetime_coord_name, datetime_coordinate)}
  )
  return dataset
