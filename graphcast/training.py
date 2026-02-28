# Copyright 2025 Stefano Campanella.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS-IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# TODO: don't use string interpolation in the log messages
import logging
import os
import pathlib
from functools import partial
from typing import Mapping, Any

import click
import gmsh
import grain
import haiku as hk
import jax
import numpy as np
import optax
import orbax.checkpoint as ocp
import xarray as xr
from grain.python import IndexSampler, DataLoader, Batch, ShardOptions, ReadOptions
from jax import checkpoint_policies as cp
from jax.experimental.shard import reshard
from jax.sharding import PartitionSpec, NamedSharding, AxisType

from graphcast import checkpoint, cli_utils, xarray_jax
from graphcast.casting import Bfloat16Cast
from graphcast.cli_utils import Configs, run_analysis_and_report
from graphcast.dataloader import ARCODataSource, FillNans, ExtractInputsTargetsForcings, AddLogDepthCoordinate, \
  WrapData
from graphcast.geospatial_mesh_utils import read_mesh
from graphcast.mask import MaskedPredictor
from graphcast.mesh_graph import MeshData, faces_to_edges, MeshGraph
from graphcast.model import TaskConfig, ModelConfig, GraphCast, CheckPoint
from graphcast.normalization import InputsAndResiduals
from graphcast.training_utils import get_optimizer, get_global_grad_fn

logger = logging.getLogger(__name__)

@click.group()
def cli():
  pass

@cli.command()
@click.argument("config_path",
                required=True,
                type=click.Path(path_type=pathlib.Path,
                                file_okay=True,
                                dir_okay=False,
                                readable=True,
                                resolve_path=True))
@click.argument("output_path",
                required=True,
                type=click.Path(path_type=pathlib.Path,
                                file_okay=True,
                                dir_okay=False,
                                writable=True,
                                resolve_path=True))
@click.option("--data-path",
              help="Path to the data directory.",
              type=click.Path(path_type=pathlib.Path,
                              file_okay=False,
                              dir_okay=True,
                              readable=True,
                              resolve_path=True))
@click.option("--other-configs",
              help="Other configs to override in the config file in the format 'key1:value1,key2:value2,...'",
              type=cli_utils.DictParamType())
@click.option("--overwrite/--no-overwrite",
              help="Whether to overwrite existing outputs",
              default=False,
              is_flag=True)
@click.option("--analysis-report/--no-analysis-report",
              "analysis",
              help="Whether log memory and cost analysis (requires `log-level` to be greater than `info`).",
              default=False,
              is_flag=True)
@click.option('--log-level',
              default='info',
              type=click.Choice(['debug', 'info', 'warning', 'error', 'critical'], case_sensitive=False))
def init(config_path: pathlib.Path,
         output_path: pathlib.Path,
         data_path: pathlib.Path | None = None,
         other_configs: Mapping[str, Any] | None = None,
         overwrite: bool = False,
         analysis = False,
         log_level: str = 'info'):

  logging.basicConfig(
    format='%(levelname)s - %(asctime)s: %(message)s',
    datefmt='%Y-%m-%dT%H:%M:%S',
    level=getattr(logging, log_level.upper()),
    force=True)

  logger.info("Initialize gmsh.")
  gmsh.initialize()
  gmsh.option.setNumber("General.Verbosity", 2)

  if output_path.exists() and not overwrite:
    raise ValueError(f"Output destination {output_path} already exists.")

  logger.info(f"Loading configs from {config_path}")
  configs = Configs.read(config_path)
  if other_configs is not None:
    configs.update(other_configs)

  task_config = TaskConfig(**configs.get('task', {}))
  target_lead_times = configs['target_lead_times']

  if data_path is None:
    data_path = pathlib.Path(os.getcwd())

  if (dataset_path := (data_path / configs.get('dataset.filepath'))) is None:
    raise ValueError("The dataset filepath must be specified in the config file.")

  logger.info(f"Loading dataset from %s", dataset_path)
  datasource = ARCODataSource(dataset_path,
                              timesteps=configs.get('dataset.timesteps', 3),
                              mask_name=configs.get('dataset.mask_name', 'glorys_mask'))
  sampler = IndexSampler(num_records=len(datasource))
  # FIXME: Should log-depth coordinate and other operation options be read from configs?
  operations = [FillNans(), AddLogDepthCoordinate(),
                ExtractInputsTargetsForcings(task=task_config, target_lead_times=target_lead_times)]
  dataloader = DataLoader(data_source=datasource, sampler=sampler, operations=operations)
  inputs, targets, forcings = next(iter(dataloader))

  if (mesh_path := (data_path / configs.get('mesh.filepath'))) is None:
    raise ValueError("The mesh filepath must be specified in the config file.")
  ocean_mesh, mesh_size = read_mesh(mesh_path=mesh_path,
                                    mesh_size_tag_name=configs.get('mesh.mesh_size_tag_name',
                                                                   'MeshSize'),
                                    step=configs.get('mesh.mesh_size_tag_step', 0))
  mesh_license = gmsh.model.getAttribute('license')
  mesh_description = gmsh.model.getAttribute('description')
  ocean_graph = MeshGraph(vertices=ocean_mesh.vertices, edges=faces_to_edges(ocean_mesh.faces), faces=ocean_mesh.faces,
                          boundary=ocean_mesh.boundary, spatial_reference_system=ocean_mesh.spatial_reference_system)
  logger.info("Mesh graph contains %d vertices and %d edges.",
              len(ocean_graph.vertices), len(ocean_graph.edges[0]))
  mesh_data = MeshData(mesh_graph=ocean_graph,
                       mesh_size=mesh_size,
                       description=mesh_license,
                       license=mesh_description)

  model_config = ModelConfig(
    latent_size=configs.get('model.latent_size'),
    gnn_msg_steps=configs.get('model.gnn_msg_steps'),
    hidden_layers=configs.get('model.hidden_layers'),
    radius_query_fraction_edge_length=configs.get('model.radius_query_fraction_edge_length'),
    per_variable_weights=configs.get('model.per_variable_weights'))

  policy = cp.save_and_offload_only_these_names(
    names_which_can_be_saved=["message_passing", "grid2mesh_gnn", "mesh_gnn", "mesh2grid_gnn"],
    names_which_can_be_offloaded=[],
    offload_src="device",  # Move from device memory
    offload_dst="pinned_host"  # To pinned host memory
  )
  predictor = GraphCast(model_config,
                        task_config,
                        grid_lat=datasource.mask['lat'].to_numpy(),
                        grid_lon=datasource.mask['lon'].to_numpy(),
                        grid_mask=datasource.mask,
                        mesh_graph=mesh_data.mesh_graph,
                        mesh_size=mesh_data.mesh_size,
                        remat=True,
                        policy=policy,
                        prevent_cse=False)
  predictor = Bfloat16Cast(predictor)

  @hk.without_apply_rng
  @hk.transform
  def run_forward(inputs, targets_template, forcings):
    return predictor(inputs, targets_template=targets_template, forcings=forcings)

  seed = configs['seed']
  logger.info(f"Initializing parameters with {seed=}")
  key = jax.random.key(seed)
  params = run_forward.init(rng=key, inputs=inputs, targets_template=targets, forcings=forcings)

  if analysis:
    logger.info("Running predictor memory and cost analysis")
    run_forward_jit = jax.jit(run_forward.apply)
    run_forward_compiled = run_forward_jit.trace(params,
                                                 inputs=inputs,
                                                 targets_template=targets,
                                                 forcings=forcings
                                                 ).lower().compile()
    run_analysis_and_report(run_forward_compiled)

  # noinspection PyTypeChecker
  graphcast_ckpt = CheckPoint(
    params=params,
    model_config=model_config,
    task_config=task_config,
    mesh_data=mesh_data,
    description=configs.get('description', ""),
    license=configs.get('license', ""))

  logger.info(f"Saving checkpoint to {output_path}")
  if not output_path.parent.exists():
    output_path.parent.mkdir(parents=True)
  with output_path.open('wb') as ckpt_file:
    checkpoint.dump(ckpt_file, graphcast_ckpt)


# FIXME: add docstring
@cli.command()
@click.argument("config_path",
                required=True,
                type=click.Path(path_type=pathlib.Path,
                                file_okay=True,
                                dir_okay=False,
                                readable=True,
                                resolve_path=True))
@click.argument("checkpoint_path",
                required=True,
                type=click.Path(path_type=pathlib.Path,
                                file_okay=True,
                                dir_okay=False,
                                readable=True,
                                resolve_path=True))
@click.option("--data-path",
              help="Path to the data directory.",
              type=click.Path(path_type=pathlib.Path,
                              file_okay=False,
                              dir_okay=True,
                              readable=True,
                              resolve_path=True))
@click.option("--other-configs",
              help="Other configs to override in the config file in the format 'key1:value1,key2:value2,...'",
              type=cli_utils.DictParamType())
@click.option("--start-fresh",
              help="Whether to start the training from scratch.",
              default=False,
              is_flag=True)
@click.option("--tensorboard-logdir",
              help="Tensorboard log directory.",
              type=click.Path(path_type=pathlib.Path,
                              file_okay=False,
                              dir_okay=True,
                              writable=True,
                              resolve_path=True))
@click.option("--analysis-report/--no-analysis-report",
              "analysis",
              help="Whether log memory and cost analysis (requires `log-level` to be greater than `info`).",
              default=False,
              is_flag=True)
@click.option('--log-level',
              default='info',
              type=click.Choice(['debug', 'info', 'warning', 'error', 'critical'], case_sensitive=False))
def launch(config_path: pathlib.Path,
           checkpoint_path: pathlib.Path,
           data_path: pathlib.Path | None = None,
           other_configs: Mapping[str, Any] | None = None,
           start_fresh: bool = False,
           tensorboard_logdir: pathlib.Path | None = None,
           analysis = False,
           log_level: str = 'info'):

  logging.basicConfig(
    format='%(levelname)s - %(asctime)s: %(message)s',
    datefmt='%Y-%m-%dT%H:%M:%S',
    level=getattr(logging, log_level.upper()),
    force=True)

  gmsh.initialize()

  logger.info(f"Loading configs from {config_path}")
  configs = Configs.read(config_path)
  if other_configs is not None:
    configs.update(other_configs)

  # Tensorflow should be imported after jax initialization, see: https://github.com/google/flax/issues/4942
  jax_backend = configs.get('jax_backend') or jax.default_backend()
  jax.distributed.initialize(local_device_ids=configs.get('local_devices', [0, 1, 2, 3]))
  _ = jax.devices()

  from tensorflow import summary
  import tensorflow as tf

  # TODO: could this be set using environment variables?
  tf.config.experimental.set_visible_devices([], 'GPU')

  # FIXME: use backend in jax.devices and jax.device_count,
  #  also check that the local batch size is divisible by the number of devices
  multi_host = jax.process_count() > 1
  logger.info(f"Setting up the device mesh with {jax.device_count()} devices "
              f"and backend {jax_backend} ({'multi-host setup' if multi_host else 'single-host setup'}).")

  device_mesh = jax.make_mesh((jax.device_count(),),
                              ('batch',),
                              devices=jax.devices(),
                              axis_types=(AxisType.Explicit,))

  jax.sharding.set_mesh(device_mesh)

  logger.info(f"Loading GraphCast init from {checkpoint_path}")
  with open(checkpoint_path, 'rb') as checkpoint_file:
    training_ckpt = checkpoint.load(checkpoint_file, CheckPoint)

  if data_path is None:
    data_path = pathlib.Path(os.getcwd())

  if (dataset_path := (data_path / configs.get('dataset.filepath'))) is None:
    raise ValueError("The dataset filepath must be specified in the config file.")

  logger.info(f"Loading dataset from {dataset_path}")
  datasource = ARCODataSource(dataset_path,
                              timesteps=configs.get('dataset.timesteps', 3),
                              mask_name=configs.get('dataset.mask_name', 'glorys_mask'))
  sampler = IndexSampler(num_records=len(datasource),
                         shard_options=ShardOptions(shard_count=jax.process_count(),
                                                    shard_index=jax.process_index()),
                         num_epochs=None,
                         shuffle=configs.get('sampler.shuffle_dataset', True),
                         seed=configs.get('sampler.seed'))

  def batch_fn(samples):
    return tuple(map(lambda datasets: xr.concat(datasets, dim='batch'), zip(*samples)))

  operations = [FillNans(),
                AddLogDepthCoordinate(),  # The negative logarithm of depth is used as a weight in loss calculations
                ExtractInputsTargetsForcings(task=training_ckpt.task_config, target_lead_times="1d"),
                Batch(batch_size=configs.get('local_batch_size', 1),
                      drop_remainder=True,
                      batch_fn=batch_fn),
                WrapData()]
  dataloader = DataLoader(data_source=datasource,
                          sampler=sampler,
                          operations=operations,
                          worker_count=configs.get('dataloader.worker_count', 0),
                          read_options=ReadOptions(num_threads=configs.get('dataloader.num_threads', 0)))

  mask = datasource.mask
  mesh_data = training_ckpt.mesh_data
  policy = cp.save_and_offload_only_these_names(
    names_which_can_be_saved=configs.get("policy.save", []),
    names_which_can_be_offloaded=configs.get("policy.offload", []),
    offload_src=configs.get("policy.offload_src", "device"),
    offload_dst=configs.get("policy.offload_dst", "pinned_host")
  )

  # Deeper one-step predictor.
  predictor = GraphCast(training_ckpt.model_config,
                        training_ckpt.task_config,
                        grid_lat=mask['lat'].to_numpy(),
                        grid_lon=mask['lon'].to_numpy(),
                        grid_mask=mask,
                        mesh_graph=mesh_data.mesh_graph,
                        mesh_size=mesh_data.mesh_size,
                        scan=False,
                        remat=True,
                        policy=policy,
                        prevent_cse=False)

  # Modify inputs/outputs to `graphcast.GraphCast` to handle conversion to from/to float32 to/from BFloat16.
  predictor = Bfloat16Cast(predictor)

  # TODO: Move artifacts loading code to training_utils.py, take care of zero residual scales, and put on device
  if (artifacts_path := (data_path / configs.get('artifacts.filepath'))) is None:
    raise ValueError("The normalization artifacts filepath must be specified in the config file.")
  logger.info(f"Loading normalization artifacts from {artifacts_path}")
  artifacts = xr.open_datatree(artifacts_path, engine='zarr')

  mean_by_level = artifacts['/inputs/location'].dataset
  mean_by_level = mean_by_level.fillna(0.0)

  stddev_by_level = artifacts['/inputs/scale'].dataset
  stddev_by_level = stddev_by_level.fillna(1.0)
  stddev_by_level = stddev_by_level.clip(min=1e-18)

  diffs_stddev_by_level = artifacts['/residuals/scale'].dataset
  diffs_stddev_by_level = diffs_stddev_by_level.fillna(1.0)
  diffs_stddev_by_level = diffs_stddev_by_level.clip(min=1e-18)

  # Modify inputs/outputs to `casting.Bfloat16Cast` so the casting to/from BFloat16 happens after applying
  # normalization to the inputs/targets.
  predictor = InputsAndResiduals(
    predictor,
    diffs_stddev_by_level=diffs_stddev_by_level,
    mean_by_level=mean_by_level,
    stddev_by_level=stddev_by_level)

  # Mask inputs/outputs replacing missing values with 0.0
  predictor = MaskedPredictor(predictor, mask=mask, value=0.0)

  global_grad_fn = get_global_grad_fn(predictor, device_mesh,
                                      levels_normalization_coord='log-depth', batch_dim_name='batch')

  optimizer = get_optimizer(configs["optimizer"])

  @partial(jax.jit, donate_argnums=(0, 1, 2))
  def train_step(params, rng_key, opt_state, sample):
    inputs, targets, forcings = sample
    rng_key, next_rng_key = jax.random.split(rng_key)
    (loss, diagnostics), grads = global_grad_fn(params, rng_key, inputs=inputs, targets=targets, forcings=forcings)
    updates, next_opt_state = optimizer.update(grads, opt_state, params)
    updated_params = optax.apply_updates(params, updates)

    return updated_params, next_rng_key, next_opt_state, loss, diagnostics

  params_checkpoint_path = data_path / configs.get("checkpoints.dirpath")
  logger.info(f"Reading and saving checkpoints from {params_checkpoint_path}")
  if start_fresh and any(params_checkpoint_path.iterdir()) and jax.process_index() == 0:
    params_checkpoint_path = ocp.test_utils.erase_and_create_empty(params_checkpoint_path)
  ckpt_mngr_options = ocp.CheckpointManagerOptions(best_fn=lambda metrics: metrics['loss'],
                                                   best_mode='min',
                                                   **configs.get('checkpoints.checkpoint_manager_options', {}))

  # FIXME: When checkpointing params as is, after a while the dataloader tries to retrieve a SharedMemoryArray whose
  #  memory has already been released, making a Grain worker fail and ultimately stopping the whole execution.
  #  The CheckpointManager has to be within a context using null_mesh, or before jax.sharding.set_mesh() is called.
  #  The reason is that within the CheckpointManager stack there is a call to jax.multihost_utils.broadcast_on_to_all
  #  (used to implement a barrier), which declares its own jax.sharding.Mesh which is generally different from the
  #  context mesh. The same goes for save and restore operations.
  #  See: https://github.com/google/orbax/issues/2545
  #  Also, it seems that converting params to a pytree of numpy arrays speeds things up.
  null_mesh = jax.make_mesh((), ())
  with jax.sharding.use_mesh(null_mesh):
    ckpt_mngr = ocp.CheckpointManager(params_checkpoint_path, options=ckpt_mngr_options)

  # Define checkpointables (dataloader iterator, params, rng, and opt_state), eventually restore them from the
  # checkpoint and move them to device.
  dataloader_iter = iter(dataloader)
  latest_step = 0
  rng_key = jax.random.key(configs['seed'])
  params = training_ckpt.params
  opt_state = optimizer.init(params)
  sharding_replicated = NamedSharding(device_mesh, PartitionSpec())

  if not start_fresh:
    latest_step = ckpt_mngr.lastest_step()
    logger.info(f"Restoring {latest_step=} from checkpoint")
    params_on_host = jax.tree_util.tree_map(np.array, params)
    opt_state_on_host = jax.tree_util.tree_map(np.array, opt_state)
    with jax.sharding.use_mesh(null_mesh):
      restored = ckpt_mngr.restore(
        step=latest_step,
        args=ocp.args.Composite(
          dataloader=grain.checkpoint.CheckpointRestore(dataloader_iter),
          params=ocp.args.StandardRestore(params_on_host, strict=True, support_layout=False),
          optimizer_state=ocp.args.StandardRestore(opt_state_on_host, strict=True, support_layout=False),
          rng=ocp.args.JaxRandomKeyRestore(restore_args=ocp.type_handlers.ArrayRestoreArgs(
            sharding=sharding_replicated))))
    dataloader_iter = restored.dataloader

    def replicate(tree):
      return jax.tree_util.tree_map(lambda local_data:
                                      jax.make_array_from_process_local_data(sharding=sharding_replicated,
                                                                             local_data=local_data),
                                    tree)

    params_on_host = restored.params
    params = replicate(params_on_host)
    rng_key = restored.rng
    opt_state_on_host = restored.optimizer_state
    opt_state = replicate(opt_state_on_host)

  # If params, rng, and opt_state have been restore from a checkpoint they should already have the correct sharding
  # and the following should be a no-op.
  params, rng, opt_state = reshard((params, rng_key, opt_state), out_shardings=sharding_replicated)

  # TODO: revise dataset put logic: could it be rewritten as a single tree_map?
  def device_put_dataset(dataset: xr.Dataset) -> xr.Dataset:
    # jax.block_until_ready is needed to ensure that shared memory arrays are converted to JAX arrays
    # while still existing (in the async case that is not guaranteed).
    sharding_along_batch = NamedSharding(device_mesh, PartitionSpec('batch'))
    dataset = dataset.map(
      lambda da: jax.block_until_ready(jax.tree_util.tree_map(
        lambda local_data: jax.make_array_from_process_local_data(sharding=sharding_along_batch,
                                                                  local_data=local_data),
        da)))
    dataset = xarray_jax.Dataset(data_vars={var: (data.dims, xarray_jax.wrap(data.data))
                                            for var, data in dataset.data_vars.items()},
                                 coords=dataset.coords,
                                 attrs=dataset.attrs)
    return dataset

  # TODO: Revise or deprecate analysis report, it could jeopardize dataloader iterator checkpoint restore.
  if analysis:
    logger.info("Running gradient memory and cost analysis")
    # Load a single minibatch from dataloader
    batch = next(dataloader_iter)
    inputs, targets, forcings = jax.tree_util.tree_map(device_put_dataset, batch,
                                                       is_leaf=lambda x: isinstance(x, xr.Dataset))
    global_grad_fn_jit = jax.jit(global_grad_fn)
    global_grad_fn_aot = global_grad_fn_jit.trace(params, rng_key,
                                                  inputs=inputs,
                                                  targets=targets,
                                                  forcings=forcings
                                                  ).lower().compile()
    run_analysis_and_report(global_grad_fn_aot)
    del batch

  tensorboard_logdir = tensorboard_logdir or (data_path / "logdir")
  summary_writer = summary.create_file_writer(str(tensorboard_logdir))
  training_steps = configs.get("training_steps")
  if training_steps is None:
    raise ValueError("The number of training steps must be specified in the config file.")
  logger.info(f"Training for {training_steps=} starting at step {latest_step=}.")
  for current_step in range(latest_step, training_steps):
    batch_on_host = next(dataloader_iter)
    try:
      batch = jax.tree_util.tree_map(device_put_dataset, batch_on_host,
                                      is_leaf=lambda x: isinstance(x, xr.Dataset))
      params, rng_key, opt_state, loss, diagnostics = train_step(params, rng_key, opt_state, batch)
      # FIXME: Orbax messes up with global mesh, see comments above. Check if new versions of Orbax fix the issue.
      with jax.sharding.use_mesh(null_mesh):
        params_on_host = jax.tree_util.tree_map(np.array, params)
        optimizer_state_on_host = jax.tree_util.tree_map(np.array, opt_state)
        ckpt_mngr.save(current_step,
                       args=ocp.args.Composite(
                         dataloader=grain.checkpoint.CheckpointSave(dataloader_iter),
                         params=ocp.args.StandardSave(params_on_host),
                         rng=ocp.args.JaxRandomKeySave(rng_key),
                         optimizer_state=ocp.args.StandardSave(optimizer_state_on_host)),
                       metrics={'loss': loss.item()})
      if jax.process_index() == 0:
        with summary_writer.as_default():
          summary.scalar("loss", loss, step=current_step)
          for key, value in diagnostics.items():
            summary.scalar(key, value, step=current_step)
    finally:
      # Explicitly delete batch_on_host to trigger shared memory release in Grain.
      # It must be done after checkpointing, otherwise current SharedMemoryArrays could be released.
      del batch_on_host

  jax.distributed.shutdown()


if __name__ == '__main__':
  cli()