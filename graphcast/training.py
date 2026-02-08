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
from typing import Mapping, Any

import click
import gmsh
import haiku as hk
import jax
import optax
import orbax.checkpoint as ocp
import xarray as xr
from grain.python import IndexSampler, DataLoader, Batch
from jax import checkpoint_policies as cp
from jax.experimental.shard_map import shard_map
from jax.sharding import PartitionSpec, NamedSharding, AxisType

from graphcast import checkpoint, cli_utils, xarray_tree
from graphcast.casting import Bfloat16Cast
from graphcast.cli_utils import Configs, run_analysis_and_report
from graphcast.dataloader import ARCODataSource, ToXarrayJax, RestoreDatetimeCoordinate, FillNans, \
  ExtractInputsTargetsForcings, AddLogDepthCoordinate, DevicePut, ShardOptions
from graphcast.geospatial_mesh_utils import read_mesh
from graphcast.mask import MaskedPredictor
from graphcast.mesh_graph import MeshData, faces_to_edges, MeshGraph
from graphcast.model import TaskConfig, ModelConfig, GraphCast, CheckPoint
from graphcast.normalization import InputsAndResiduals
from graphcast.training_utils import get_optimizer
from graphcast.xarray_jax import unwrap_data

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
  operations = [ToXarrayJax(),
                FillNans(),
                RestoreDatetimeCoordinate(multi_host=False),
                ExtractInputsTargetsForcings(task=task_config,
                                             target_lead_times=target_lead_times)]
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


  logger.info(f"Loading GraphCast checkpoint from {checkpoint_path}")
  with open(checkpoint_path, 'rb') as checkpoint_file:
    training_ckpt = checkpoint.load(checkpoint_file, CheckPoint)

  params_checkpoint_path = data_path / configs.get("checkpoints.dirpath")
  logger.info(f"Saving params checkpoint from {params_checkpoint_path}")
  params_checkpoint_path = params_checkpoint_path.resolve()
  if jax.process_index() == 0 and any(params_checkpoint_path.iterdir()):
    params_checkpoint_path = ocp.test_utils.erase_and_create_empty(params_checkpoint_path)

  if data_path is None:
    data_path = pathlib.Path(os.getcwd())

  if (dataset_path := (data_path / configs.get('dataset.filepath'))) is None:
    raise ValueError("The dataset filepath must be specified in the config file.")

  def sharding(*dims):
    return NamedSharding(device_mesh, PartitionSpec(*dims))

  logger.info(f"Loading dataset from {dataset_path}")
  datasource = ARCODataSource(dataset_path,
                              timesteps=configs.get('dataset.timesteps', 3),
                              mask_name=configs.get('dataset.mask_name', 'glorys_mask'))
  sampler = IndexSampler(num_records=len(datasource),
                         # FIXME:
                         # shard_options=BatchParallelShardOptions(sharding('batch')),
                         shard_options=ShardOptions(shard_count=jax.process_count(),
                                                    shard_index=jax.process_index()),
                         num_epochs=None,
                         shuffle=configs.get('sampler.shuffle_dataset', True),
                         seed=configs.get('sampler.seed'))
  # The order of operations is constrained by the following requirements:
  # 1. RestoreDatetimeCoordinate must be called before ExtractInputsTargetsForcings, as the datetime coordinate is used
  #    to determine the progress (e.g. day of the year) variables.
  # 2. MakeArrayFromProcessLocalData must be called before RestoreDatetimeCoordinate, as the latter assume that data is
  #    sharded (and it contains an all_gather communication).
  # 3. ToXarrayJax must be called before MakeArrayFromProcessLocalData, as the latter takes numpy arrays as inputs and
  #    returns jax arrays, and the former takes care of casting the datetime coordinate to a dtype that could be used in
  #    a jax array.
  # FIXME: IMPORTANT! Current implementation cannot use multiple workers as Device objects cannot be pickled. It might
  #  be the case that most post-processing steps should be moved to the training loop.
  operations = [Batch(batch_size=configs.get('local_batch_size', 1),
                      drop_remainder=True,
                      batch_fn=lambda datasets: xr.concat(datasets, dim='batch')),
                ToXarrayJax(),
                FillNans(),
                AddLogDepthCoordinate(),  # The negative logarithm of depth is used as a weight in loss calculations
                DevicePut(sharding=sharding('batch'), multi_host=multi_host),
                RestoreDatetimeCoordinate(multi_host=multi_host),
                ExtractInputsTargetsForcings(task=training_ckpt.task_config,
                                             target_lead_times="1d",
                                             derived_vars_device=sharding('batch'))]
  dataloader = DataLoader(data_source=datasource,
                          sampler=sampler,
                          operations=operations,
                          worker_count=configs.get('dataloader.worker_count', 0))

  # Load a single minibatch from dataloader
  if jax.process_count() > 1:
    inputs, targets, forcings = jax.tree_util.tree_map(lambda xs: xs.addressable_data(0), next(iter(dataloader)))
  else:
    inputs, targets, forcings = next(iter(dataloader))

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
                        remat=True,
                        policy=policy,
                        prevent_cse=False)

  # Modify inputs/outputs to `graphcast.GraphCast` to handle conversion to from/to float32 to/from BFloat16.
  predictor = Bfloat16Cast(predictor)

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

  # get params from checkpoint
  params = training_ckpt.params

  @hk.without_apply_rng
  @hk.transform
  def local_loss_fn(inputs, targets, forcings):
    loss, diagnostics = predictor.loss(inputs=inputs, targets=targets, forcings=forcings,
                                       levels_normalization_coord='log-depth')
    return xarray_tree.map_structure(
      lambda x: unwrap_data(x.mean(), require_jax=True),
      (loss, diagnostics))

  local_grad_fn = jax.value_and_grad(local_loss_fn.apply, has_aux=True)

  # Data parallel section

  params = jax.device_put(params, device=sharding())
  args_template = (params,) + next(iter(dataloader))
  args_template_value, args_template_structure = jax.tree.flatten(args_template)
  args_template_shapedtypestruct = jax.tree_util.tree_map(lambda x: jax.ShapeDtypeStruct(x.shape, x.dtype),
                                                          args_template)
  return_template = jax.eval_shape(local_grad_fn, *args_template_shapedtypestruct)

  in_specs = (jax.tree_util.tree_map(lambda x: x.sharding.spec, args_template_value),)
  out_specs = jax.tree_util.tree_map(lambda x: PartitionSpec(), return_template)

  def global_grad_fn(params, inputs, targets, forcings):

    # get pytrees from xarray.Dataset
    args_value, args_structure = jax.tree.flatten((params, inputs, targets, forcings))
    assert args_structure == args_template_structure

    def pmean_grad_fn(args_value):
      # reconstruct xarray.Dataset from pytrees
      params, inputs, targets, forcings = args_template_structure.unflatten(args_value)
      (loss, diagnostics), grads = jax.lax.pmean(local_grad_fn(params, inputs, targets, forcings), axis_name='batch')
      return (loss, diagnostics), grads

    # Eager evaluation of some function inside a `shard_map` isn't yet supported, hence the need for jit here.
    _global_grad_fn = shard_map(jax.jit(pmean_grad_fn),
                                mesh=device_mesh,
                                in_specs=in_specs,
                                out_specs=out_specs,
                                check_rep=False)

    return _global_grad_fn(args_value)

  if analysis:
    logger.info("Running gradient memory and cost analysis")
    global_grad_fn_jit = jax.jit(global_grad_fn)
    global_grad_fn_aot = global_grad_fn_jit.trace(params,
                                                  inputs=inputs,
                                                  targets=targets,
                                                  forcings=forcings
                                                  ).lower().compile()
    run_analysis_and_report(global_grad_fn_aot)

  optimizer = get_optimizer(configs["optimizer"])
  opt_state = optimizer.init(params)

  @jax.jit
  def train_step(params, opt_state, sample):
    inputs, targets, forcings = sample
    (loss, diagnostics), grads = global_grad_fn(params, inputs=inputs, targets=targets, forcings=forcings)
    updates, opt_state = optimizer.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)

    return params, opt_state, loss, diagnostics

  ckpt_mngr_options = ocp.CheckpointManagerOptions(max_to_keep=3, best_fn=lambda metrics: metrics['loss'],
                                                   best_mode='min')
  ckpt_mngr = ocp.CheckpointManager(params_checkpoint_path, options=ckpt_mngr_options)
  logdir = (data_path / configs.get('logging.filepath', 'logs')) / os.getenv('SLURM_JOB_ID', 'local')
  summary_writer = summary.create_file_writer(str(logdir))
  dataloader_iter = iter(dataloader)
  with summary_writer.as_default():
    training_steps = configs.get("training_steps", 1024)
    logger.info(f"Running {training_steps} training steps")
    for step in range(training_steps):
      params, opt_state, loss, diagnostics = train_step(params, opt_state, next(dataloader_iter))
      ckpt_mngr.save(step, args=ocp.args.StandardSave(params), metrics={'loss': loss.item()})
      if jax.process_index() == 0:
        summary.scalar("loss", loss, step=step)
        for key, value in diagnostics.items():
          summary.scalar(key, value, step=step)

  jax.distributed.shutdown()


if __name__ == '__main__':
  cli()