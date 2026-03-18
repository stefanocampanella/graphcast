import logging
import os
import pathlib
from typing import Tuple, Union, Callable

import grain.python as grain
import haiku as hk
import jax
import numpy as np
import optax
import orbax.checkpoint as ocp
import xarray as xr
from etils import epath
from grain.checkpoint import CheckpointRestore as GrainCkptRestore, CheckpointSave as GrainCkptSave
from grain.experimental import pick_performance_config
from grain.python import IterDataset, DatasetIterator
from jax import checkpoint_policies as cp
from jax.experimental import multihost_utils
from jax.experimental.multihost_utils import sync_global_devices
from jax.experimental.shard_map import shard_map
from jax.sharding import PartitionSpec as P

from graphcast import xarray_jax, checkpoint
from graphcast.casting import Bfloat16Cast
from graphcast.cli_utils import Configs, OrbaxLogger
from graphcast.dataloader import ARCODataSource, InputsTargetsForcings
from graphcast.dataset_utils import Process
from graphcast.geospatial_mesh_utils import read_mesh_data
from graphcast.mask import MaskedPredictor
from graphcast.mesh_graph import MeshData
from graphcast.model import ModelConfig, TaskConfig, GraphCast
from graphcast.normalization import InputsAndResiduals
from graphcast.predictor_base import Predictor

logger = logging.getLogger(__name__)

Datasets = Tuple[xr.Dataset, ...]
DatasetsOrDataArrays = Tuple[Union[xr.Dataset, xr.DataArray], ...]
Paths = Tuple[epath.Path, ...]


def check_paths(output: epath.Path,
                train: epath.Path,
                data: epath.Path,
                tb: epath.Path,
                start_fresh: bool = False,
                overwrite: bool = False) -> Paths:
  """Check preconditions for path-arguments, safely creating and deleting directories if necessary."""
  def ensure_directory_exists(path: epath.Path):
    if jax.process_index() == 0 and not path.exists():
      path.mkdir(parents=True, exist_ok=True)

  def empty_directory(path: epath.Path):
    if jax.process_index() == 0:
      if path.exists():
        path.rmtree()
      path.mkdir(parents=True, exist_ok=True)

  mesh = jax.make_mesh((jax.device_count(), jax.local_device_count()), ('process', 'local_device'))
  with jax.sharding.use_mesh(mesh):

    # output is where the final checkpoint will be saved:
    #   1. It should not point to an existing file if overwrite=False.
    #   2. It should be writable
    # We ensure that the parent directory exists with write permission.
    ensure_directory_exists(output.parent)
    if output.exists() and not overwrite:
      raise FileExistsError(f"{output} already exists.")
    if not os.access(output, os.W_OK):
      raise PermissionError(f"{output} is not writable.")
    sync_global_devices("check_output_path")

    # train is the directory where checkpoints will be saved during training:
    #   1. If it exists and start_fresh=True, it should be empty.
    #   2. It should be readable and writable.
    # We ensure that the directory exists and is empty if start_fresh=True.
    ensure_directory_exists(train)
    if any(train.iterdir()) and start_fresh:
      empty_directory(train)
    if not os.access(train, os.R_OK & os.W_OK):
      raise PermissionError(f"{train} is not readable and/or writable.")
    sync_global_devices("check_train_path")

    # data_path is the directory containing the mesh, the datasets, and the normalization artifacts:
    #   1. It should be readable.
    # We ensure that the directory exists.
    ensure_directory_exists(data)
    if not os.access(data, os.R_OK):
      raise PermissionError(f"{data} is not readable.")
    sync_global_devices("check_data_path")

    # tb is where the summarywriter will write the logs during training:
    #   1. It should be writable.
    # We ensure that the directory exists.
    ensure_directory_exists(tb)
    if not os.access(tb, os.W_OK):
      raise PermissionError(f"{tb} is not writable.")
    sync_global_devices("check_tb_path")

  return output, train, data, tb


def get_mesh(data_path: epath.Path, configs: Configs) -> MeshData:

  mesh_data_path = data_path / configs.get('mesh.filepath', required=True)
  logger.info(f"Loading mesh from {mesh_data_path}")
  mesh_data = read_mesh_data(mesh_data_path,
                             mesh_size_tag_name=configs.get('mesh.mesh_size_tag_name', 'MeshSize'),
                             mesh_size_tag_step=configs.get('mesh.mesh_size_tag_step', 0))
  return mesh_data


def get_mask(data_path: epath.Path, configs: Configs) -> xr.DataArray:
  path = data_path / configs.get('mask.filepath', required=True)
  logger.info(f"Loading mask from {path}")
  mask = xr.open_dataset(path, engine='zarr')
  mask_name = configs.get('mask.name', required=True)
  mask = mask[mask_name]
  mask_level = configs.get('mask.level', required=True)
  mask = mask.isel(level=mask_level, drop=True)
  return mask


def get_artifacts(data_path: epath.Path, configs: Configs) -> Datasets:
  path = data_path / configs.get('artifacts.filepath', required=True)
  logger.info(f"Loading normalization artifacts from {path}")
  artifacts = xr.open_datatree(path, engine='zarr')

  def _get_ds(name):
    logger.info(f"Getting {name} dataset from normalization artifacts.")
    ds_path = configs.get(f"artifacts.{name}.path", required=True)
    ds = artifacts[ds_path].dataset
    postprocess = Process(steps=configs.get(f"artifacts.{name}.postprocess", None))
    ds = postprocess(ds)
    return ds

  return tuple(_get_ds(name) for name in ['mean_by_level', 'stddev_by_level', 'diffs_stddev_by_level'])


def get_predictor(configs: Configs,
                  mesh_data: MeshData,
                  grid_lat: np.ndarray,
                  grid_lon: np.ndarray,
                  grid_mask: np.ndarray,
                  mean_by_level: xr.Dataset,
                  stddev_by_level: xr.Dataset,
                  diffs_stddev_by_level: xr.Dataset,
                  mask_da: xr.DataArray,
                  ) -> Predictor:
  model_config = ModelConfig(
    latent_size=configs.get('model.latent_size', required=True),
    gnn_msg_steps=configs.get('model.gnn_msg_steps', required=True),
    hidden_layers=configs.get('model.hidden_layers', required=True),
    radius_query_fraction_edge_length=configs.get('model.radius_query_fraction_edge_length', required=True),
    per_variable_weights=configs.get('model.per_variable_weights', {}),
    learnable_fourier_features=configs.get('model.learnable_fourier_features', required=True))

  task_config = TaskConfig(
    input_variables=configs.get('task.input_variables', required=True),
    target_variables=configs.get('task.target_variables', required=True),
    forcing_variables=configs.get('task.forcing_variables', required=True),
    levels=configs.get('task.levels', required=True),
    input_duration=configs.get('task.input_duration', required=True))

  policy = cp.save_and_offload_only_these_names(
    names_which_can_be_saved=configs.get("policy.save", []),
    names_which_can_be_offloaded=configs.get("policy.offload", []),
    offload_src=configs.get("policy.offload_src", "device"),
    offload_dst=configs.get("policy.offload_dst", "pinned_host")
  )

  # Deeper one-step predictor.
  predictor = GraphCast(
    _model_config=model_config,
    _task_config=task_config,
    _grid_lat=grid_lat,
    _grid_lon=grid_lon,
    _grid_mask=grid_mask,
    _mesh_data=mesh_data,
    _scan=False,
    _remat=True,
    _policy=policy,
    _prevent_cse=False,
  )

  # Modify inputs/outputs to `graphcast.GraphCast` to handle conversion to from/to float32 to/from BFloat16.
  predictor = Bfloat16Cast(predictor)

  # Modify inputs/outputs to `casting.Bfloat16Cast` so the casting to/from BFloat16 happens after applying
  # normalization to the inputs/targets.
  predictor = InputsAndResiduals(
    predictor,
    diffs_stddev_by_level=diffs_stddev_by_level,
    mean_by_level=mean_by_level,
    stddev_by_level=stddev_by_level,
    skip_names=configs.get('artifacts.skip_names', []))

  # Mask inputs/outputs. Notice, other not finite values (i.e., inf) are not filled with mask.fill_value.
  fill_value = configs.get('mask.fill_value', required=True)
  predictor = MaskedPredictor(predictor, mask=mask_da, value=fill_value)

  return predictor


# FIXME: this should use cosine_decay_schedule_with_warmup instead of chaining schedules as it currently does.
def get_optimizer(configs: Configs) -> optax.GradientTransformationExtraArgs:
  logger.info(f"Optimizing with {configs.get('optimizer')}")
  schedule_configs = configs.get('optimizer.schedule', [])
  if not schedule_configs:
    raise ValueError("No learning rate schedule specified.")
  schedules = []
  for schedule_config in schedule_configs:
    schedule_name = schedule_config.pop('name')
    schedule = getattr(optax, schedule_name)
    schedules.append(schedule(**schedule_config))
  boundaries = configs.get('optimizer.schedule_boundaries', [])
  if not boundaries:
    raise ValueError("No boundaries specified for learning rate schedule.")
  scheduler = optax.join_schedules(schedules, boundaries=boundaries)

  gradient_transformation_configs = configs.get('optimizer.gradient_transformation', [])
  if not gradient_transformation_configs:
    raise ValueError("No gradient transformation specified.")
  gradient_transformations = []
  for gradient_transformation_config in gradient_transformation_configs:
    gradient_transformation_name = gradient_transformation_config.pop('name')
    gradient_transformation = getattr(optax, gradient_transformation_name)
    if gradient_transformation_name == 'adamw':
      gradient_transformation_config['learning_rate'] = scheduler
    gradient_transformations.append(gradient_transformation(**gradient_transformation_config))
  return optax.chain(*gradient_transformations)


def get_dataset_iterator(data_path: epath.Path, configs: Configs, train: bool = True) -> IterDataset[InputsTargetsForcings]:
  dataset_path = data_path / configs.get('dataset.filepath', required=True)
  logger.info(f"Loading training and test datasource from {dataset_path}")

  task_config = TaskConfig(
    input_variables=configs.get('task.input_variables', required=True),
    target_variables=configs.get('task.target_variables', required=True),
    forcing_variables=configs.get('task.forcing_variables', required=True),
    levels=configs.get('task.levels', required=True),
    input_duration=configs.get('task.input_duration', required=True))

  split_date = configs.get('dataset.split_date', required=True)
  if train:
    from_date, to_date = None, split_date
  else:
    from_date, to_date = split_date, None

  datasource = ARCODataSource(dataset_path,
                              task=task_config,
                              target_lead_times=configs.get('dataset.target_lead_times', required=True),
                              from_date=from_date,
                              to_date=to_date)

  dataset = (grain.MapDataset.source(datasource)
             .repeat(num_epochs=None)
             .shuffle(seed=configs.get('seed', required=True))
             .slice(slice(jax.process_index(), None, jax.process_count())))

  local_batch_size = configs.get('local_batch_size', 1)
  if local_batch_size > 1:
    logger.info(f"Batching {local_batch_size} samples per device.")
    def batch_fn(samples):
      return tuple(map(lambda datasets: xr.concat(datasets, dim='batch'), zip(*samples)))
    dataset = dataset.batch(batch_size=local_batch_size,
                            drop_remainder=True,
                            batch_fn=batch_fn)

  if pick_config := configs.get('dataset.pick_performance_config', None):
    # pick_performance_config needs to measure the size of elements drawn from a DatasetIterator. However, the current
    # implementation of _get_element_size_bytes in grain/_src/python/dataset/transformations/prefetch_autotune.py
    # does not work with xarray datasets. We get around the issue by flattening the dataset beforehand.
    flatten_dataset_iterator = dataset.map(jax.tree_util.tree_flatten).to_iter_dataset()
    logger.info(f"Picking dataset iterator performance configurations ({pick_config=}).")
    performance_config = pick_performance_config(ds=flatten_dataset_iterator,
                                                 ram_budget_mb = pick_config.get('ram_budget_mb', None),
                                                 max_workers = pick_config.get('max_workers', None),
                                                 max_buffer_size = pick_config.get('max_buffer_size', None),
                                                 samples_to_check = pick_config.get('samples_to_check', None))
    read_options = performance_config.read_options
    mp_options = performance_config.multiprocessing_options
  else:
    read_config = configs.get('dataset.read_options', required=True)
    read_options = grain.ReadOptions(num_threads=read_config.get('num_threads', required=True),
                                     prefetch_buffer_size=read_config.get('prefetch_buffer_size', required=True))
    if mp_config := configs.get('dataset.multiprocessing_options', None):
      mp_options = grain.MultiprocessingOptions(
        num_workers=mp_config.get('num_workers', required=True),
        per_worker_buffer_size=mp_config.get('per_worker_buffer_size', required=True),
        enable_profiling=mp_config.get('enable_profiling', False))
    else:
      mp_options = None

  logger.info(f"Using read options: {read_options}")
  dataset_iterator = dataset.to_iter_dataset(read_options=read_options)
  if mp_options is not None:
    logger.info(f"Using multiprocessing prefetch with options: {mp_options}")
    # When using multiprocessing prefetching, we need to wrap the xarray datasets to allow using grain
    # SharedMemoryArrays.
    dataset_iterator = (dataset
                        .map(lambda value: xarray_jax.wrap_data(value, to_jax=False, np_contiguous=True))
                        .to_iter_dataset(read_options=read_options)
                        .mp_prefetch(options=mp_options))

  return dataset_iterator


def get_first_sample_and_reset(iter: DatasetIterator[InputsTargetsForcings]) -> InputsTargetsForcings:
  state = iter.get_state()
  sample = next(iter)
  iter.set_state(state)
  return sample


def get_params(init_fn: Callable[..., hk.MutableParams], iterator: DatasetIterator, configs: Configs):
  seed = configs['seed']
  rng_key = jax.random.key(seed)
  sample_on_host = get_first_sample_and_reset(iterator)
  # FIXME: avoid grain workers failing (harmless, but produce lousy logs)
  sample = xarray_jax.device_put(sample_on_host)
  logger.info(f"Initializing parameters with {seed=}")
  return init_fn(rng_key, sample)


def get_checkpoint_manager(ckpt_path: epath.Path, configs: Configs) -> ocp.CheckpointManager:
  logger.info(f"Reading from and saving checkpoints to {ckpt_path}")
  ckpt_mngr_options = ocp.CheckpointManagerOptions(best_fn=lambda metrics: metrics['loss'],
                                                   best_mode='min',
                                                   **configs.get('checkpoints', {}))
  # `jax.multihost_utils.sync_global_devices` implements the barrier by calling
  # `jax.multihost_utils.broadcast_on_to_all`, which inside uses jax.sharding.Mesh declared for the purpose and
  # generally different from the context mesh, causing an error.
  # For this reason, orbax CheckpointManager (which uses such a barrier) needs to be called using `null_mesh`, or before
  # `jax.sharding.set_mesh()` is called. The same goes for save and restore operations.
  #  See: https://github.com/google/orbax/issues/2545
  mesh = jax.make_mesh((jax.process_count(), jax.local_device_count()), ('process', 'local_device'))
  with jax.sharding.use_mesh(mesh):
    ckpt_mngr = ocp.CheckpointManager(ckpt_path, options=ckpt_mngr_options, logger=OrbaxLogger())
  return ckpt_mngr


def pull_checkpoint(ckpt_mngr, params, opt_state, train_iterator, test_iterator):
  params_on_host = jax.device_get(params)
  opt_state_on_host = jax.device_get(opt_state)
  latest_step = ckpt_mngr.latest_step()
  logger.info(f"Restoring {latest_step=}")
  # TODO: check restore and save args options related to sharding and layout
  # FIXME: Orbax messes up with global mesh, see comment in get_checkpoint_manager. Check if new versions of Orbax fix the issue.
  restored = ckpt_mngr.restore(
    step=latest_step,
    args=ocp.args.Composite(
      train_iterator=GrainCkptRestore(train_iterator),
      test_iterator=GrainCkptRestore(test_iterator),
      params=ocp.args.StandardRestore(params_on_host),
      opt_state=ocp.args.StandardRestore(opt_state_on_host)))
  train_iterator.close()
  test_iterator.close()

  train_iterator = restored.train_iterator
  test_iterator = restored.test_iterator
  params = jax.device_put(restored.params)
  opt_state = jax.device_put(restored.opt_state)
  return params, opt_state, train_iterator, test_iterator


def push_checkpoint(ckpt_mngr, step, loss, params, opt_state, train_iterator, test_iterator):
  params_on_host = jax.device_get(params)
  opt_state_on_host = jax.device_get(opt_state)
  # FIXME: Orbax messes up with global mesh, see comment in get_checkpoint_manager. Check if new versions of Orbax fix the issue.
  ckpt_mngr.save(step,
                 args=ocp.args.Composite(
                   train_iterator=GrainCkptSave(train_iterator),
                   test_iterator=GrainCkptSave(test_iterator),
                   params=ocp.args.StandardSave(params_on_host),
                   opt_state=ocp.args.StandardSave(opt_state_on_host)),
                 metrics={'loss': loss})


# get_global_grad_fn supports an apply function which depends on PRNGkeys (after a haiku.transform).
# However, in the case of FSDS, there is the need to have a different rng key for each element of the batch.
# However, it turns out to be non-trivial. See:
# https://github.com/jax-ml/jax/discussions/22862
# https://github.com/jax-ml/jax/issues/22860
# Fold-in trick taken from
# https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/scaling/JAX/data_parallel_fsdp.html
# might not work when using a global mesh, and produce errors like the following.
#
#    File ".../jax/_src/prng.py", line 628, in random_fold_in
#      return random_fold_in_p.bind(keys, msgs)
#             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#   NotImplementedError: Closing over inputs to shard_map where the input is sharded on `Explicit` axes is not
#   implemented. As a workaround, please pass those inputs as an argument to shard_map.
#   Got input with shape key<fry>[]({Explicit: ('batch',)})
#
# See also: https://github.com/jax-ml/jax/issues/29162
#
# An alternative to fold-in might be to split the rng key and shard it, e.g.,
#   rngs = jax.random.split(rng, device_mesh.shape[sharding_dim])
#   rngs = jax.device_put(rngs, device_mesh)
# However, the previous might produce the following error related to addressable devices and require more thinking.

def fsdp_map(local_fn, mesh: jax.sharding.Mesh, batch_dim_name: str = 'batch'):
  def _local_grad_fn(params, rng_key, data, static_data):
    local_rng_key = jax.random.fold_in(rng_key, jax.lax.axis_index(batch_dim_name))
    return local_fn(params, local_rng_key, data, static_data)

  def _pmean_grad_fn(params, rng_key, data, static_data):
    return jax.lax.pmean(_local_grad_fn(params, rng_key, data, static_data), axis_name=batch_dim_name)

  def global_fn(params, rng_key, data, static_data):
    _global_grad_fn = shard_map(_pmean_grad_fn,
                                mesh=mesh,
                                in_specs=(P(), P(), P(batch_dim_name), P()),
                                out_specs=P(),
                                check_rep=False)
    return _global_grad_fn(params, rng_key, data, static_data)

  return global_fn


# As the dataloader calls extract_inputs_targets_forcings, which is missing the datetime coordinate, the following
# isn't needed anymore.
def reshard_data(dataset, sharding, datetime_coord_name='time'):
  """Utility to reshard datasets with datetime coordinates."""
  datetime_coord = dataset[datetime_coord_name]
  datetime_coord.data = datetime_coord.data.astype("datetime64[s]").astype(np.int64)
  dataset = dataset.drop(datetime_coord_name)
  dataset = dataset.map(
    lambda da: jax.tree_util.tree_map(
      lambda xs: jax.make_array_from_process_local_data(sharding=sharding, local_data=xs),
      da))
  dataset = xarray_jax.Dataset(data_vars={var: (data.dims, data.data) for var, data in dataset.data_vars.items()},
                               coords=dataset.coords,
                               jax_coords={datetime_coord_name: datetime_coord},
                               attrs=dataset.attrs)
  datetime_coordinate = multihost_utils.process_allgather(datetime_coord)
  # In the case of single-host setups it might happen that datetime is still a numpy array
  datetime_coordinate = xarray_jax.unwrap_data(datetime_coordinate, require_jax=False)
  datetime_coordinate = np.asarray(datetime_coordinate).astype('datetime64[s]')
  dataset = dataset.drop(datetime_coord_name)
  dataset = dataset.assign_coords({datetime_coord_name: (datetime_coord_name, datetime_coordinate)})
  return dataset


# # FIXME: this should be refactored, predictor cannot live outside of a haiku transform
# def save_checkpoint(predictor: Predictor, params, output_path: pathlib.Path, configs: Configs) -> None:
#   logger.info(f"Training finished, saving checkpoint to {output_path}")
#   with jax.sharding.use_mesh(_null_mesh):
#     if jax.process_index() == 0:
#       with output_path.open('wb') as ckpt_file:
#         graphcast_ckpt = predictor.checkpoint(params,
#                                               description=configs.get('description'),
#                                               license=configs.get('license'))
#         checkpoint.dump(ckpt_file, graphcast_ckpt)
#     sync_global_devices("save_checkpoint")


class SummaryWriter:

  def __init__(self, tb_path: pathlib.Path):

    # Tensorflow should be imported after jax initialization, see: https://github.com/google/flax/issues/4942
    if not jax.distributed.is_initialized():
      raise ValueError("Tensorboard summarywriter requires a distributed setup.")
    else:
      _ = jax.devices()
      import tensorflow as tf
      # TODO: could this be set using environment variables?
      tf.config.experimental.set_visible_devices([], 'GPU')
      from tensorflow import summary
      self._summary_writer = summary.create_file_writer(str(tb_path))

  def scalar(self, current_step, loss_and_diagnostics, suffix="/train"):
    from tensorflow import summary

    if jax.process_index() == 0:
      loss, diagnostics = loss_and_diagnostics
      with self._summary_writer.as_default():
        summary.scalar("loss" + suffix, loss, step=current_step)
        for key, value in diagnostics.items():
          summary.scalar(key + suffix, value, step=current_step)
