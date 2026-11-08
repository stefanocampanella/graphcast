# SPDX-FileCopyrightText: 2026 Stefano Campanella
# SPDX-License-Identifier: Apache-2.0
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
"""moby-dick compute-grid-basis: linear basis functions of a mesh at the ocean points of the training grid (draft)."""

import logging
import pathlib

import click
import gmsh
from scipy import sparse

from mobydick import cli_utils
from mobydick.basis_utils import compute_grid_basis
from mobydick.dataloader_utils import get_grid
from mobydick.geospatial_mesh_utils import read_mesh

logger = logging.getLogger(__name__)


@click.command("compute-grid-basis")
@click.argument("mesh_path", type=click.Path(path_type=pathlib.Path, exists=True, dir_okay=False))
@click.argument("mask_path", type=click.Path(path_type=pathlib.Path, exists=True))
@click.argument("output_path", type=click.Path(path_type=pathlib.Path, dir_okay=False, writable=True))
@click.option(
  "--overwrite/--no-overwrite",
  default=False,
  help="Whether to overwrite existing outputs",
  is_flag=True,
)
@cli_utils.LOG_LEVEL_OPTION
def compute_grid_basis_command(
  mesh_path: pathlib.Path,
  mask_path: pathlib.Path,
  output_path: pathlib.Path,
  overwrite: bool = False,
  log_level: str = "info",
):
  """Evaluate the linear basis functions of the mesh at MESH_PATH on the ocean points of the mask at MASK_PATH
  (written by moby-dick download-mask) and save them to OUTPUT_PATH, a sparse [ocean points, mesh vertices] matrix
  (scipy .npz)."""
  cli_utils.configure_logging(log_level)
  if output_path.exists() and not overwrite:
    raise click.ClickException(f"{output_path} exists: pass --overwrite to replace it")
  grid = get_grid(mask_path)
  if not gmsh.is_initialized():
    gmsh.initialize()
  gmsh.option.set_number("General.Verbosity", 2)
  mesh, _ = read_mesh(mesh_path, mesh_size_tag_name=None)
  basis = compute_grid_basis(mesh, latitudes=grid.lat, longitudes=grid.lon, mask=grid.mask)
  logger.info(f"Saving the basis ({basis.shape[0]} points, {basis.shape[1]} vertices) to {output_path}")
  output_path.parent.mkdir(parents=True, exist_ok=True)
  # Written to the path as given (save_npz would append .npz to a path without it).
  with output_path.open("wb") as file:
    sparse.save_npz(file, basis)
