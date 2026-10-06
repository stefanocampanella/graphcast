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

from graphcast.cli_utils import Configs
from graphcast.geospatial_mesh_utils import (
  CompositeMeshSizeField,
  StereoMeshSizeField,
  coarsen_boundaries,
  load_domain,
)
from graphcast.gis_utils import CRSRegistry

logger = logging.getLogger(__name__)


@click.group()
def cli():
  pass


@cli.command()
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
@click.option(
  "--log-level",
  default="info",
  type=click.Choice(["debug", "info", "warning", "error", "critical"], case_sensitive=False),
)
def make(
  config_path: pathlib.Path,
  output_path: pathlib.Path,
  data_path: pathlib.Path | None = None,
  overwrite: bool = False,
  log_level: str = "info",
):
  logging.basicConfig(
    format="%(levelname)s - %(asctime)s: %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
    level=log_level.upper(),
    force=True,
  )

  if not gmsh.is_initialized():
    logger.info("Initialize gmsh.")
    gmsh.initialize()

  # Open the configuration file and load the TOML configs.
  configs = Configs.read(config_path)

  # If the destination exists and should not overwrite, raise and exit.
  if output_path.exists() and not overwrite:
    raise ValueError(f"Output destination {output_path} already exists")

  if not output_path.parent.exists():
    output_path.parent.mkdir(parents=True, exist_ok=True)

  # gmsh needs to be initialized before most seamsh functions can be used, so we do it here.
  # FIXME: These should be read from the config file.
  gmsh.option.setNumber("General.Verbosity", 2)
  gmsh.option.setNumber("PostProcessing.SaveMesh", 1)
  gmsh.option.setNumber("Mesh.Algorithm", 6)
  gmsh.option.setNumber("Mesh.MshFileVersion", 4.1)
  # Create a new model and set license and description attributes
  model_name = configs.get("model_name", "OceanMesh")
  gmsh.model.add(model_name)
  if mesh_license := configs.get("license"):
    gmsh.model.setAttribute("license", [mesh_license])
  if mesh_description := configs.get("description"):
    gmsh.model.setAttribute("description", [mesh_description])
  if (domain_path := pathlib.Path(configs.get("domain.filepath"))) is None:
    raise ValueError("Domain filepath must be specified in the config file.")
  if data_path is not None:
    domain_path = data_path / domain_path
  logger.info("Loading and coarsening boundary.")
  # Default values assume that the field name in the shapefile is "featurecla", and that the curve type is "bspline".
  domain = load_domain(
    domain_path,
    physical_name_field=configs.get("domain.physical_name_field", "featurecla"),
    curve_type=configs.get("domain.curve_type", "bspline"),
  )
  # Default values assume that the point (0.0, 0.0) in stereographic coordinates is within the domain.
  boundary = coarsen_boundaries(
    domain,
    configs.get("coarsen_boundaries.mesh_size"),
    x0=configs.get("coarsen_boundaries.x0", (0.0, 0.0)),
    x0_projection_name=configs.get("coarsen_boundaries.x0_projection", "stereographic"),
  )
  # Build mesh size field
  fields_config = configs.get("fields", [])
  if not fields_config:
    raise ValueError("At least one field must be specified in the config file.")
  mesh_size = CompositeMeshSizeField(fields_config, prefix=data_path)
  # Mesh using seamsh
  # noinspection PyTypeChecker
  seamsh.gmsh.mesh(
    boundary,
    mesh_size,
    output_srs=CRSRegistry["cartesian"],
    smoothness=configs.get("mesh.smoothness", 0.3),
  )
  # Compute and add view containing mesh size field(s), finally save results
  node_tags, node_coords, _ = gmsh.model.mesh.getNodes(includeBoundary=True)
  node_coords = node_coords.reshape(-1, 3)
  logger.info(f"Writing mesh size field to {output_path}")

  def write_model_data(view_tag_name: str, field: StereoMeshSizeField, append=False) -> None:
    mesh_size_at_nodes = field.mesh_size_3d(node_coords, projection=CRSRegistry["cartesian"])
    mesh_size_at_nodes = mesh_size_at_nodes.tolist()
    view_tag = gmsh.view.add(view_tag_name)
    gmsh.view.addHomogeneousModelData(
      tag=view_tag,
      step=0,
      modelName=model_name,
      dataType="NodeData",
      tags=node_tags,
      data=mesh_size_at_nodes,
    )
    gmsh.view.write(view_tag, str(output_path), append=append)

  view_tag_name = configs.get("view_tag_name", "MeshSize")
  write_model_data(view_tag_name, mesh_size)
  for field_name, field in mesh_size.fields.items():
    write_model_data(field_name, field, append=True)


if __name__ == "__main__":
  cli()
