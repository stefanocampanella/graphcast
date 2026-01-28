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
# Seamsh usage example are taken from: https://jlambrechts.git-page.immc.ucl.ac.be/seamsh/examples/6-stereographics.html
"""Tools for working with triangular ocean meshes."""
import logging
import pathlib

import click
import gmsh
import seamsh

from graphcast.dataset_utils import Configs
from graphcast.ocean_mesh_utils import load_domain, coarsen_boundaries, ShoreProximityField, ConstantField, \
  ProjectionRegistry

logger = logging.getLogger(__name__)

gmsh.initialize()


@click.group()
def cli():
  pass


@cli.command()
@click.argument("config_path",
                required=True,
                type=click.Path(path_type=pathlib.Path, file_okay=True, readable=True))
@click.argument("input_path",
                required=True,
                type=click.Path(path_type=pathlib.Path, file_okay=True, dir_okay=False, readable=True))
@click.argument("output_path",
                required=True,
                type=click.Path(path_type=pathlib.Path, dir_okay=True, writable=True))
@click.option("--overwrite/--no-overwrite",
              help="Whether to overwrite existing outputs",
              default=False,
              is_flag=True)
@click.option('--log-level',
              default='info',
              type=click.Choice(['debug', 'info', 'warning', 'error', 'critical'], case_sensitive=False))
def make(config_path: pathlib.Path,
         input_path: pathlib.Path,
         output_path: pathlib.Path,
         overwrite: bool = False,
         log_level: str = 'info'):
  logging.basicConfig(format='%(levelname)s - %(asctime)s: %(message)s',
                      datefmt='%Y-%m-%dT%H:%M:%S',
                      level=getattr(logging, log_level.upper()))

  # Open the configuration file and load the TOML configs.
  configs = Configs.read(config_path)

  # If the destination exists and should not overwrite, raise and exit.
  if output_path.exists() and not overwrite:
    raise ValueError(f"Output destination {output_path} already exists")

  # gmsh needs to be initialized before most seamsh functions can be used, so we do it here.
  gmsh.option.setNumber("General.Verbosity", 2)
  gmsh.option.setNumber("PostProcessing.SaveMesh", 1)

  # Default values assume that the field name in the shapefile is "featurecla", and that the curve type is "bspline".
  domain = load_domain(input_path,
                       physical_name_field=configs.get("domain.physical_name_field", "featurecla"),
                       curve_type=configs.get("domain.curve_type", "bspline"))

  # Default values assume that the point (0.0, 0.0) in stereographic coordinates is within the domain.
  coarse = coarsen_boundaries(domain, configs.get("coarsen_boundaries.mesh_size"),
                              x0=configs.get("coarsen_boundaries.x0", (0.0, 0.0)),
                              x0_projection=configs.get("coarsen_boundaries.x0_projection", "stereographic"))

  field_name = configs.get("field.name")
  if field_name == "ShoreProximityField":
    mesh_size = ShoreProximityField(domain=domain,
                                    sampling=configs.get("field.sampling"),
                                    field_min=configs.get("field.field_min"),
                                    field_max=configs.get("field.field_max"),
                                    size_min=configs.get("field.size_min"),
                                    size_max=configs.get("field.size_max"))
  elif field_name == "ConstantField":
    mesh_size = ConstantField(configs.get("field.value"))
  else:
    raise ValueError(f"Unknown field type: {field_name}")

  model_name = configs.get("model_name", "OceanMesh")
  gmsh.model.add(model_name)
  if mesh_license := configs.get('license'):
    gmsh.model.setAttribute('license', [mesh_license])
  if mesh_description := configs.get('description'):
    gmsh.model.setAttribute('description', [mesh_description])
  # Mesh using seamsh
  seamsh.gmsh.mesh(coarse, mesh_size, output_srs=ProjectionRegistry['cartesian'])
  # Compute and add view containing mesh size field, finally save results
  node_tags, node_coords, _ = gmsh.model.mesh.getNodes(includeBoundary=True)
  node_coords = node_coords.reshape(-1, 3)
  mesh_size_at_nodes = mesh_size.mesh_size_3d(node_coords, projection=ProjectionRegistry['cartesian'])
  mesh_size_at_nodes = mesh_size_at_nodes.tolist()
  view_tag_name = configs.get("view_tag_name", "MeshSize")
  view_tag = gmsh.view.add(view_tag_name)
  gmsh.view.addHomogeneousModelData(
    tag=view_tag,
    step=0,
    modelName=model_name,
    dataType="NodeData",
    tags=node_tags,
    data=mesh_size_at_nodes)
  logger.info(f"Writing mesh size field to {output_path}")
  gmsh.view.write(view_tag, str(output_path))


if __name__ == "__main__":
  cli()

gmsh.finalize()
