# Copyright 2026 Stefano Campanella.
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
# Seamsh usage example are taken from: https://jlambrechts.git-page.immc.ucl.ac.be/seamsh/examples/6-stereographics.html
"""Tools for working with triangular ocean meshes."""

import logging
import pathlib

import click
import gmsh
import seamsh

from graphcast import cli_utils
from graphcast.geospatial_mesh_utils import (
  CompositeMeshSizeField,
  StereoMeshSizeField,
  coarsen_boundaries,
  load_domain,
)
from graphcast.gis_utils import CRSRegistry
from graphcast.mesh_config import MeshGenerationConfig

logger = logging.getLogger(__name__)


@click.command()
@click.argument(
  "config_path",
  required=True,
  type=click.Path(path_type=pathlib.Path, file_okay=True, readable=True),
)
@click.argument(
  "output_path",
  required=True,
  type=click.Path(path_type=pathlib.Path, dir_okay=True, writable=True),
)
@click.option(
  "--data-path",
  "data_path",
  help="Prefix to filepaths contained in config file.",
  default=None,
  type=click.Path(path_type=pathlib.Path, file_okay=False, dir_okay=True, readable=True),
)
@click.option(
  "--overwrite/--no-overwrite",
  help="Whether to overwrite existing outputs",
  default=False,
  is_flag=True,
)
@cli_utils.LOG_LEVEL_OPTION
def make_ocean_mesh(
  config_path: pathlib.Path,
  output_path: pathlib.Path,
  data_path: pathlib.Path | None = None,
  overwrite: bool = False,
  log_level: str = "info",
):
  """Mesh the ocean bounded by the coastline of CONFIG_PATH and write it, with its mesh size fields, to OUTPUT_PATH."""
  cli_utils.configure_logging(log_level)
  # Invalid settings and an existing output fail before meshing.
  configs = cli_utils.read_configs(config_path, schema=MeshGenerationConfig)
  if output_path.exists() and not overwrite:
    raise click.ClickException(f"{output_path} exists: pass --overwrite to replace it")
  logger.info(f"Using settings: {configs!r}")
  output_path.parent.mkdir(parents=True, exist_ok=True)
  write_ocean_mesh(configs, output_path, data_path)


def write_ocean_mesh(configs: MeshGenerationConfig, output_path: pathlib.Path, data_path: pathlib.Path | None) -> None:
  """Mesh the domain with seamsh and gmsh, and write the mesh with one view per mesh size field (and their minimum)."""
  # gmsh needs to be initialized before most seamsh functions can be used.
  if not gmsh.is_initialized():
    logger.info("Initialize gmsh.")
    gmsh.initialize()
  gmsh.option.setNumber("General.Verbosity", configs.gmsh.verbosity)
  gmsh.option.setNumber("PostProcessing.SaveMesh", 1)
  gmsh.option.setNumber("Mesh.Algorithm", configs.gmsh.algorithm)
  gmsh.option.setNumber("Mesh.MshFileVersion", configs.gmsh.msh_file_version)
  # Create a new model and set license and description attributes
  gmsh.model.add(configs.model_name)
  if configs.license:
    gmsh.model.setAttribute("license", [configs.license])
  if configs.description:
    gmsh.model.setAttribute("description", [configs.description])
  domain_path = pathlib.Path(configs.domain.filepath)
  if data_path is not None:
    domain_path = data_path / domain_path
  logger.info("Loading and coarsening boundary.")
  domain = load_domain(
    domain_path, physical_name_field=configs.domain.physical_name_field, curve_type=configs.domain.curve_type
  )
  boundary = coarsen_boundaries(
    domain,
    configs.coarsen_boundaries.mesh_size,
    x0=configs.coarsen_boundaries.x0,
    x0_projection_name=configs.coarsen_boundaries.x0_projection,
  )
  mesh_size = CompositeMeshSizeField(configs.fields, prefix=data_path)
  seamsh.gmsh.mesh(boundary, mesh_size, output_srs=CRSRegistry["cartesian"], smoothness=configs.mesh.smoothness)
  # Compute and add view containing mesh size field(s), finally save results
  node_tags, node_coords, _ = gmsh.model.mesh.getNodes(includeBoundary=True)
  node_coords = node_coords.reshape(-1, 3)
  logger.info(f"Writing mesh size field to {output_path}")

  def write_model_data(view_tag_name: str, field: StereoMeshSizeField, append=False) -> None:
    mesh_size_at_nodes = field.mesh_size_3d(node_coords, projection=CRSRegistry["cartesian"])
    view_tag = gmsh.view.add(view_tag_name)
    gmsh.view.addHomogeneousModelData(
      tag=view_tag,
      step=0,
      modelName=configs.model_name,
      dataType="NodeData",
      tags=node_tags,
      data=mesh_size_at_nodes.tolist(),
    )
    gmsh.view.write(view_tag, str(output_path), append=append)

  write_model_data(configs.view_tag_name, mesh_size)
  for field_name, field in mesh_size.fields.items():
    write_model_data(field_name, field, append=True)
