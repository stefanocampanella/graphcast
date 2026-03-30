import inspect
import logging
import os
import pathlib
import typing
from typing import Tuple, Union, Callable, Mapping

import grain.python as grain
import haiku as hk
import jax
import numpy as np
import optax
import orbax.checkpoint as ocp
import xarray as xr
from etils import epath
from grain.checkpoint import (CheckpointSave as IterDatasetSave,
                              CheckpointRestore as IterDatasetRestore)
from grain.experimental import pick_performance_config
from grain.python import IterDataset, DatasetIterator
from jax import checkpoint_policies as cp
from jax.experimental import multihost_utils
from jax.experimental.multihost_utils import sync_global_devices
from jax.experimental.shard_map import shard_map
from jax.sharding import PartitionSpec as P, Mesh
from orbax.checkpoint.args import PyTreeSave, PyTreeRestore

from graphcast import xarray_jax
from graphcast.casting import Bfloat16Cast
from graphcast.checkpoint import dump as ckpt_dump
from graphcast.cli_utils import Configs, OrbaxLogger
from graphcast.dataloader import ARCODataSource, InputsTargetsForcings
from graphcast.dataset_utils import Process
from graphcast.geospatial_mesh_utils import read_mesh_data
from graphcast.mask import Mask
from graphcast.mesh_graph import MeshData
from graphcast.model import ModelConfig, TaskConfig, GraphCast, CheckPoint
from graphcast.normalization import InputsAndResiduals
from graphcast.predictor_base import Predictor

logger = logging.getLogger(__name__)

Datasets = Tuple[xr.Dataset, ...]
MaybeDatasets = Tuple[Union[Datasets, None], ...]
DatasetsOrDataArrays = Tuple[Union[xr.Dataset, xr.DataArray], ...]
Paths = Tuple[epath.Path, ...]
InputsTargetsForcingsIterator = DatasetIterator[InputsTargetsForcings]
InputsTargetsForcingsIterDataset = IterDataset[InputsTargetsForcings]
Params = hk.Params | hk.MutableParams
JAXLossAndDiagnostics = Tuple[jax.Array, Mapping[str, jax.Array]]


def check_writable_paths(output: epath.Path,
                         train: epath.Path,
                         tb: epath.Path,
                         start_fresh: bool = False,
                         overwrite: bool = False) -> Paths:
  """Check preconditions for path-arguments to write to, safely creating and deleting directories if necessary."""

  def ensure_directory_exists(path: epath.Path):
    if jax.process_index() == 0 and not path.exists():
      path.mkdir(parents=True, exist_ok=True)
    sync_global_devices(f"ensure_{path}_exists")

  def empty_directory(path: epath.Path):
    if jax.process_index() == 0:
      if path.exists():
        path.rmtree()
      path.mkdir(parents=True, exist_ok=True)
    sync_global_devices(f"empty_{path}")

  mesh = jax.make_mesh((jax.device_count(), jax.local_device_count()), ('process', 'local_device'))
  with jax.sharding.set_mesh(mesh):

    # output is where the final checkpoint will be saved.
    # We ensure that output' parent directory exists (with write permission if created). Also:
    #   1. If overwrite=True, and output points to an existing file, then it should be writable
    #   2. If overwrite=True, and output doesn't point to an existing file, then the parent directory should be writable.
    #   3. If overwrite=False, then output should not point to an existing file.
    ensure_directory_exists(output.parent)
    if overwrite:
      if output.exists():
        if not os.access(output, os.W_OK):
          raise PermissionError(f"{output} is not writable.")
      else:
        if not os.access(output.parent, os.W_OK):
          raise PermissionError(f"{output.parent} is not writable.")
    else:
      if output.exists():
        raise FileExistsError(f"{output} already exists.")

    # train is the directory where checkpoints will be saved during training.
    # We ensure that the directory exists (with write permission if created) and is empty if start_fresh=True, also:
    #   1. It should be readable and writable.
    ensure_directory_exists(train)
    if start_fresh and any(train.iterdir()):
      empty_directory(train)
    if not os.access(train, os.R_OK & os.W_OK):
      raise PermissionError(f"{train} is not readable and/or writable.")

    # tb is where the summarywriter will write the logs during training:
    # We ensure that the directory exists (with write permission if created).
    #   1. It should be writable.
    ensure_directory_exists(tb)
    if not os.access(tb, os.W_OK):
      raise PermissionError(f"{tb} is not writable.")

  return output, train, tb


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


def get_artifacts(data_path: epath.Path, configs: Configs) -> MaybeDatasets:
  path = data_path / configs.get('artifacts.filepath', required=True)
  logger.info(f"Loading normalization artifacts from {path}")
  artifacts = xr.open_datatree(path, engine='zarr')

  def _get_ds(name):
    logger.info(f"Getting {name} dataset from normalization artifacts.")
    if ds_path := configs.get(f"artifacts.{name}.path"):
      ds = artifacts[ds_path].dataset
      postprocess = Process(steps=configs.get(f"artifacts.{name}.postprocess", None))
      ds = postprocess(ds)
      return ds
    else:
      return None

  return tuple(_get_ds(name) for name in ['mean_by_level', 'stddev_by_level', 'diffs_stddev_by_level'])


def get_model_config(configs: Configs) -> ModelConfig:

  return ModelConfig(
    latent_size=configs.get('model.latent_size', required=True),
    gnn_msg_steps=configs.get('model.gnn_msg_steps', required=True),
    hidden_layers=configs.get('model.hidden_layers', required=True),
    radius_query_fraction_edge_length=configs.get('model.radius_query_fraction_edge_length', required=True),
    per_variable_weights=configs.get('model.per_variable_weights', {}),
    learnable_fourier_features=configs.get('model.learnable_fourier_features', False),
    fourier_features_num_frequencies=configs.get('model.fourier_features_num_frequencies', 1),
    fourier_features_hidden_dim=configs.get('model.fourier_features_hidden_dim'),
    fourier_features_encoding_dim=configs.get('model.fourier_features_encoding_dim'))


def get_task_config(configs: Configs) -> TaskConfig:

  return TaskConfig(
    input_variables=configs.get('task.input_variables', required=True),
    target_variables=configs.get('task.target_variables', required=True),
    forcing_variables=configs.get('task.forcing_variables', required=True),
    levels=configs.get('task.levels', required=True),
    input_duration=configs.get('task.input_duration', required=True))


def get_policy(configs: Configs) -> Callable:

  return cp.save_and_offload_only_these_names(
    names_which_can_be_saved=configs.get("policy.save", []),
    names_which_can_be_offloaded=configs.get("policy.offload", []),
    offload_src=configs.get("policy.offload_src", "device"),
    offload_dst=configs.get("policy.offload_dst", "pinned_host")
  )


def get_predictor(configs: Configs,
                  mesh_data: MeshData,
                  grid_lat: np.ndarray,
                  grid_lon: np.ndarray,
                  grid_mask: np.ndarray,
                  mean_by_level: xr.Dataset,
                  stddev_by_level: xr.Dataset,
                  mask_da: xr.DataArray,
                  diffs_stddev_by_level: xr.Dataset | None = None,
                  ) -> Predictor:

  # Deeper one-step predictor.
  predictor = GraphCast(
    _model_config=get_model_config(configs),
    _task_config=get_task_config(configs),
    _grid_lat=grid_lat,
    _grid_lon=grid_lon,
    _grid_mask=grid_mask,
    _mesh_data=mesh_data,
    _scan=False,
    _remat=True,
    _policy=get_policy(configs),
    _prevent_cse=False,
  )

  # Modify inputs/outputs to `graphcast.GraphCast` to handle conversion to from/to float32 to/from BFloat16.
  predictor = Bfloat16Cast(predictor)

  # Modify inputs/outputs to `casting.Bfloat16Cast` so the casting to/from BFloat16 happens after applying
  # normalization to the inputs/targets.
  predictor = InputsAndResiduals(
    predictor,
    mean_by_level=mean_by_level,
    stddev_by_level=stddev_by_level,
    diffs_stddev_by_level=diffs_stddev_by_level,
    skip_names=configs.get('artifacts.skip_names', []))

  # Mask inputs/outputs. Notice, other not finite values (i.e., inf) are not filled with mask.fill_value.
  predictor = Mask(predictor, mask=mask_da, value=configs.get('mask.fill_value', required=True))

  return predictor


def get_optimizer(configs: Configs) -> optax.GradientTransformationExtraArgs:
  logger.info(f"Optimizing with {configs.get('optimizer')}")
  schedule_configs = configs.get('optimizer.schedule', [])
  if not schedule_configs:
    raise ValueError("No learning rate schedule specified.")
  scheduler_configs = configs.get('optimizer.schedule')
  scheduler_name = scheduler_configs.pop('name')
  scheduler_init = getattr(optax.schedules, scheduler_name)
  scheduler = scheduler_init(**scheduler_configs)

  gradient_transformation_configs = configs.get('optimizer.gradient_transformation', [])
  if not gradient_transformation_configs:
    raise ValueError("No gradient transformation specified.")
  gradient_transformations = []
  for gradient_transformation_config in gradient_transformation_configs:
    gradient_transformation_name = gradient_transformation_config.pop('name')
    gradient_transformation_init = getattr(optax, gradient_transformation_name)
    gradient_transformation_init_signature = inspect.signature(gradient_transformation_init)
    if 'learning_rate' in gradient_transformation_init_signature.parameters:
      gradient_transformation_config['learning_rate'] = scheduler
    gradient_transformation = gradient_transformation_init(**gradient_transformation_config)
    gradient_transformations.append(gradient_transformation)
  return optax.chain(*gradient_transformations)


def get_dataset_iterator(data_path: epath.Path,
                         configs: Configs,
                         train: bool = True,
                         ) -> InputsTargetsForcingsIterDataset:
  dataset_path = data_path / configs.get('dataset.filepath', required=True)
  logger.info(f"Loading datasource from {dataset_path}")

  task_config = get_task_config(configs)

  split_date = configs.get('dataset.split_date', required=True)
  datasource = ARCODataSource(dataset_path,
                              task=task_config,
                              target_lead_times=configs.get('dataset.target_lead_times', required=True),
                              from_date=None if train else split_date,
                              to_date=split_date if train else None)

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


def get_first_sample_and_reset(dataset_iterator: DatasetIterator[InputsTargetsForcings]) -> InputsTargetsForcings:
  state = dataset_iterator.get_state()
  sample = next(dataset_iterator)
  dataset_iterator.set_state(state)
  return sample


def get_params(init_fn: Callable[..., Params],
               iterator: DatasetIterator[InputsTargetsForcings],
               configs: Configs,
               seed: int | None = None
               ) -> Params:
  seed = seed or configs['seed']
  rng_key = jax.random.key(seed)
  sample_on_host = get_first_sample_and_reset(iterator)
  # FIXME: avoid grain workers failing (harmless, but produce lousy logs)
  sample = xarray_jax.device_put(sample_on_host)
  logger.info(f"Initializing parameters with {seed=}")
  return init_fn(rng_key, sample)


def get_checkpoint_manager(ckpt_path: epath.Path,
                           configs: Configs,
                           ) -> ocp.CheckpointManager:
  logger.info(f"Reading from and saving checkpoints to {ckpt_path}")
  ckpt_mngr_options = ocp.CheckpointManagerOptions(best_fn=lambda metrics: metrics[0],
                                                   best_mode='min',
                                                   **configs.get('checkpoints', {}))
  # `jax.multihost_utils.sync_global_devices` implements the barrier by calling
  # `jax.multihost_utils.broadcast_on_to_all`, which inside uses jax.sharding.Mesh declared for the purpose and
  # generally different from the context mesh, causing an error.
  # For this reason, orbax CheckpointManager (which uses such a barrier) needs to be called using `null_mesh`, or before
  # `jax.sharding.set_mesh()` is called. The same goes for save and restore operations.
  #  See: https://github.com/google/orbax/issues/2545
  mesh = jax.make_mesh((jax.process_count(), jax.local_device_count()), ('process', 'local_device'))
  with jax.sharding.set_mesh(mesh):
    ckpt_mngr = ocp.CheckpointManager(ckpt_path,
                                      options=ckpt_mngr_options,
                                      logger=OrbaxLogger())
  return ckpt_mngr


def push_checkpoint(ckpt_mngr: ocp.CheckpointManager,
                    step: int,
                    metrics: JAXLossAndDiagnostics,
                    train_iterator: InputsTargetsForcingsIterator,
                    test_iterator: InputsTargetsForcingsIterator,
                    params: Params,
                    opt_state: Params,
                    ) -> None:
  metrics = jax.tree_util.tree_map(lambda x: x.item(), metrics)
  params_on_host = jax.device_get(params)
  opt_state_on_host = jax.device_get(opt_state)
  # FIXME: Orbax messes up with global mesh, see comment in get_checkpoint_manager. Check if new versions of Orbax fix the issue.
  ckpt_mngr.save(step,
                 args=ocp.args.Composite(
                   train_iterator=IterDatasetSave(train_iterator),
                   test_iterator=IterDatasetSave(test_iterator),
                   params=PyTreeSave(params_on_host),
                   opt_state=PyTreeSave(opt_state_on_host)),
                 metrics=metrics)


def pull_latest_checkpoint(ckpt_mngr: ocp.CheckpointManager,
                           train_iterator: InputsTargetsForcingsIterator,
                           test_iterator: InputsTargetsForcingsIterator,
                           params: Params,
                           opt_state: Params,
                           device = None,
                           ) -> Tuple[InputsTargetsForcingsIterator, InputsTargetsForcingsIterDataset, Params, Params]:
  latest_step = ckpt_mngr.latest_step()
  logger.info(f"Restoring {latest_step=}")
  # TODO: check restore and save args options related to sharding and layout
  # FIXME: Orbax messes up with global mesh, see comment in get_checkpoint_manager. Check if new versions of Orbax fix the issue.
  restored = ckpt_mngr.restore(
    step=latest_step,
    args=ocp.args.Composite(
      train_iterator=IterDatasetRestore(train_iterator),
      test_iterator=IterDatasetRestore(test_iterator),
      params=PyTreeRestore(params),
      opt_state=PyTreeRestore(opt_state)))

  train_iterator = restored.train_iterator
  test_iterator = restored.test_iterator
  params = jax.device_put(restored.params, device)
  opt_state = jax.device_put(restored.opt_state, device)
  return train_iterator, test_iterator, params, opt_state


def next_batches_on_device(train_iterator: InputsTargetsForcingsIterator,
                           test_iterator: InputsTargetsForcingsIterator,
                           device_mesh: Mesh,
                           mp_prefetch: bool = False,
                           ) -> Tuple[InputsTargetsForcings, InputsTargetsForcings]:

  if mp_prefetch:
    try:
      batches_on_host = next(train_iterator), next(test_iterator)
      batches = xarray_jax.make_array_from_process_local_data(batches_on_host, mesh=device_mesh, spec=P('batch'))
      batch, batch_test = jax.block_until_ready(batches)
    finally:
      # Explicitly delete batch_on_host to trigger shared memory release in Grain.
      # It must be done after checkpointing, otherwise current SharedMemoryArrays could be released.
      del batches_on_host
  else:
    batches_on_host = next(train_iterator), next(test_iterator)
    batch, batch_test = xarray_jax.make_array_from_process_local_data(batches_on_host,
                                                                      mesh=device_mesh,
                                                                      spec=P('batch'))

  return batch, batch_test


class TensorboardLogger:

  def __init__(self, tb_path: pathlib.Path):

    # Tensorflow should be imported after jax initialization, see: https://github.com/google/flax/issues/4942
    if not jax.distributed.is_initialized():
      raise ValueError("JAX distributed should be initialized beforehand, "
                       "see: https://github.com/google/flax/issues/4942.")
    else:
      _ = jax.devices()
      import tensorflow as tf
      # TODO: could this be set using environment variables?
      tf.config.experimental.set_visible_devices([], 'GPU')
      from tensorflow import summary
      self._summary = summary
      self._summary_writer = summary.create_file_writer(str(tb_path))

  def log(self, current_step: int, train_metrics: JAXLossAndDiagnostics, test_metrics: JAXLossAndDiagnostics) -> None:
    with self._summary_writer.as_default():

      def _log(key, value):
        if jax.process_index() == 0:
          self._summary.scalar(key, value, step=current_step)

      for set_name, set_metrics in [('train', train_metrics), ('test', test_metrics)]:
        set_metrics = jax.tree_util.tree_map(lambda x: x.item(), set_metrics)
        loss, diagnostics = set_metrics
        _log(f'{set_name}/loss', loss)
        for key, value in diagnostics.items():
          _log(f'{set_name}/{key}', value)


def save_model(output_path: epath.Path,
               ckpt_mngr: ocp.CheckpointManager,
               configs: Configs,
               grid_lat: np.ndarray,
               grid_lon: np.ndarray,
               grid_mask: np.ndarray,
               mesh_data: MeshData,
               ) -> None:

  @hk.without_apply_rng
  @hk.transform
  def _get_model_checkpoint():

    @hk.without_apply_rng
    @hk.transform
    def _checkpoint_fn():
      predictor = GraphCast(
        _model_config=get_model_config(configs),
        _task_config=get_task_config(configs),
        _grid_lat=grid_lat,
        _grid_lon=grid_lon,
        _grid_mask=grid_mask,
        _mesh_data=mesh_data
      )
      predictor = Bfloat16Cast(predictor)
      description = configs.get('description', '')
      license = configs.get('license', '')
      return CheckPoint(
        model_config=predictor.model_config,
        task_config=predictor.task_config,
        mesh_data=predictor.mesh_data,
        grid_lat=predictor.grid_lat,
        grid_lon=predictor.grid_lon,
        grid_mask=predictor.grid_mask,
        description=description,
        license=license)

    rng = hk.next_rng_key() if hk.running_init() else None
    params = hk.transparent_lift(_checkpoint_fn.init)(rng)
    checkpoint = _checkpoint_fn.apply(params)
    checkpoint.params.clear()
    checkpoint.params.update(params)
    return checkpoint

  device_mesh = jax.make_mesh((jax.device_count(), jax.local_device_count()),
                              ('process', 'local_device'))
  with jax.sharding.set_mesh(device_mesh):
    best_step = ckpt_mngr.best_step()
    restored = ckpt_mngr.restore(step=best_step, args=ocp.args.Composite(params=None))
    graphcast_ckpt = _get_model_checkpoint.apply(restored.params)
    if jax.process_index() == 0:
      logger.info(f"Saving checkpoint to {output_path} ({best_step=}) ")
      with output_path.open('wb') as file:
        file = typing.cast(typing.BinaryIO, file)
        ckpt_dump(file, graphcast_ckpt)
    sync_global_devices("save_checkpoint")


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
