# Copyright 2025 OGS.
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
import seamsh

from graphcast.dataset_utils import Configs
from graphcast.ocean_mesh_utils import load_domain, coarsen_boundaries, ShoreProximityField, ConstantField, \
  ProjectionRegistry



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

  # If destination exists and should not overwrite, raise and exit.
  if output_path.exists() and not overwrite:
    raise ValueError(f"Output destination {output_path} already exists")

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

  seamsh.gmsh.mesh(coarse, str(output_path), mesh_size, output_srs=ProjectionRegistry[configs.get("save.projection")])


if __name__ == "__main__":
  cli()
