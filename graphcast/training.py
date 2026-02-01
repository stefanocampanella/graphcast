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
import xarray as xr
from grain.python import IndexSampler, DataLoader, Batch
from jax import checkpoint_policies as cp
from jax.experimental.shard_map import shard_map
from jax.sharding import PartitionSpec, NamedSharding, AxisType

from graphcast import checkpoint, cli_utils, xarray_tree
from graphcast.casting import Bfloat16Cast
from graphcast.cli_utils import run_analysis_and_report
from graphcast.dataloader import ARCODataSource, ToXarrayJax, RestoreDatetimeCoordinate, FillNans, \
  ExtractInputsTargetsForcings, AddLogDepthCoordinate, DevicePut, ShardOptions
from graphcast.dataset_utils import Configs
from graphcast.mask import MaskedPredictor
from graphcast.mesh_graph import MeshData, faces_to_edges, MeshGraph
from graphcast.model import TaskConfig, ModelConfig, GraphCast, CheckPoint
from graphcast.normalization import InputsAndResiduals
from graphcast.xarray_jax import unwrap_data
from graphcast.geospatial_mesh_utils import read_mesh

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
                       boundary_nodes=boundary_nodes,
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


if __name__ == '__main__':
  cli()