import logging
import os
import pathlib
from typing import Mapping, Any

import click
import haiku as hk
import jax
from grain.python import IndexSampler, DataLoader

from graphcast import checkpoint, cli_utils
from graphcast.cli_utils import analysis_report
from graphcast.dataloader import ARCODataSource, ToXarrayJax, RestoreDatetimeCoordinate, FillNans, \
  ExtractInputsTargetsForcings
from graphcast.dataset_utils import Configs
from graphcast.mesh_connectivity import get_connected_mesh_nodes, mask_mesh
from graphcast.mesh_graph import MeshData, faces_to_edges, MeshGraph
from graphcast.model import _get_max_edge_distance, TaskConfig, ModelConfig, GraphCast, CheckPoint
from graphcast.ocean_mesh_utils import read_mesh


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
  logger = logging.getLogger()

  if output_path.exists() and not overwrite:
    raise ValueError(f"Output destination {output_path} already exists")

  logger.info(f"Loading configs from {config_path}")
  configs = Configs.read(config_path)
  if other_configs is not None:
    configs.update(other_configs)

  if data_path is None:
    data_path = pathlib.Path(os.getcwd())

  if (dataset_path := (data_path / configs.get('dataset.filepath'))) is None:
    raise ValueError("The dataset filepath must be specified in the config file.")

  task_config = TaskConfig(**configs.get('task', {}))
  target_lead_times = configs['target_lead_times']

  logger.info(f"Loading dataset from {dataset_path}")
  datasource = ARCODataSource(dataset_path, timesteps=configs.get('dataset.timesteps', 3))
  sampler = IndexSampler(num_records=len(datasource))
  operations = [ToXarrayJax(),
                FillNans(),
                RestoreDatetimeCoordinate(gather_from_all_processes=False),
                ExtractInputsTargetsForcings(task=task_config,
                                             target_lead_times=target_lead_times)]
  dataloader = DataLoader(data_source=datasource, sampler=sampler, operations=operations)
  inputs, targets, forcings = next(iter(dataloader))

  logger.info(f"Extracting inputs, targets and forcings from dataset")

  if (mesh_path := (data_path / configs.get('mesh.filepath'))) is None:
    raise ValueError("The mesh filepath must be specified in the config file.")
  logger.info(f"Loading mesh from {mesh_path}")
  ocean_mesh, boundary_nodes = read_mesh(mesh_path)
  ocean_graph = MeshGraph(vertices=ocean_mesh.vertices, edges=faces_to_edges(ocean_mesh.faces), faces=ocean_mesh.faces)
  query_radius = configs.get('mesh.radius_query_fraction_edge_length', 1.0) * _get_max_edge_distance(ocean_graph)
  connected_mesh_nodes = get_connected_mesh_nodes(grid_lat=datasource.mask['lat'],
                                                  grid_lon=datasource.mask['lon'],
                                                  mesh_graph=ocean_graph,
                                                  grid_mask=datasource.mask,
                                                  query_radius=query_radius,
                                                  workers=-1)
  ocean_graph, _  = mask_mesh(connected_mesh_nodes, ocean_graph, mode='all')
  mesh_data = MeshData(mesh_graph=ocean_graph,
                       boundary_nodes=boundary_nodes,
                       description=configs.get('mesh.description', ""),
                       license=configs.get('mesh.license', ""))

  logger.info("Creating model")
  model_config = ModelConfig(
    latent_size=configs.get('model.latent_size'),
    gnn_msg_steps=configs.get('model.gnn_msg_steps'),
    hidden_layers=configs.get('model.hidden_layers'),
    radius_query_fraction_edge_length=configs.get('mesh.radius_query_fraction_edge_length'),
    per_variable_weights=configs.get('model.per_variable_weights', {}))

  seed = configs['seed']
  logger.info(f"Creating initializing parameters ({seed=})")
  predictor = GraphCast(model_config,
                        task_config,
                        grid_lat=datasource.mask['lat'].to_numpy(),
                        grid_lon=datasource.mask['lon'].to_numpy(),
                        grid_mask=datasource.mask,
                        mesh_graph=mesh_data.mesh_graph,
                        boundary_nodes=mesh_data.boundary_nodes)
  @hk.without_apply_rng
  @hk.transform
  def run_forward(inputs, targets_template, forcings):
    return predictor(inputs, targets_template=targets_template, forcings=forcings)
  key = jax.random.key(seed)
  params = run_forward.init(rng=key, inputs=inputs, targets_template=targets, forcings=forcings)

  if analysis:
    analysis_report(logger, run_forward.apply,
                    params=params, inputs=inputs, targets_template=targets, forcings=forcings)

  # noinspection PyTypeChecker
  graphcast_ckpt = CheckPoint(
    params=params,
    model_config=model_config,
    task_config=task_config,
    mesh_data=mesh_data,
    description=configs.get('description', ""),
    license=configs.get('license', ""))

  logger.info(f"Saving checkpoint to {output_path}")
  if output_path.parent.exists() is False:
    output_path.parent.mkdir(parents=True)
  with output_path.open('wb') as ckpt_file:
    checkpoint.dump(ckpt_file, graphcast_ckpt)


if __name__ == '__main__':
  cli()