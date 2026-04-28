import logging

import click
import numpy as np
from etils import epath
from scipy import sparse

from graphcast.basis_utils import compute_basis, load_mask
from graphcast.geospatial_mesh_utils import read_mesh_data

logger = logging.getLogger(__name__)

@click.group()
def cli():
  pass

@cli.command()
@click.argument("mesh_path",
                required=True,
                type=click.Path(path_type=epath.Path, file_okay=True, readable=True))
@click.argument("dataset_path",
                required=True,
                type=click.Path(path_type=epath.Path, file_okay=True, readable=True))
@click.argument("output_path",
                required=True,
                type=click.Path(path_type=epath.Path, dir_okay=True, writable=True))
@click.option("--mesh-size-tag-name",
              default="MeshSize",
              help="Name of the tag containing the mesh size",
              show_default=True)
@click.option("--mesh-size-tag-step",
              default=0,
              help="Step of the tag containing the mesh size",
              show_default=True)
@click.option("--mask-name",
              default="glorys_mask",
              help="Name of the mask variable")
@click.option("--longitude-dim",
              default="lon",
              help="Name of the longitude variable")
@click.option("--latitude-dim",
              default="lat",
              help="Name of the longitude variable")
@click.option("--overwrite/--no-overwrite",
              default=False,
              help="Whether to overwrite existing outputs",
              is_flag=True)
@click.option("--num-blocks",
              default=10,
              help="Number of blocks for the computation",
              show_default=True)
@click.option("--log-level",
              default='info',
              type=click.Choice(['debug', 'info', 'warning', 'error', 'critical'], case_sensitive=False),
              show_default=True,
              help="Logging level.")
@click.option("--progress/--no-progress",
              default=True,
              help="Whether to display a progress bar",
              is_flag=True)
def compute(mesh_path: epath.Path,
            dataset_path: epath.Path,
            output_path: epath.Path,
            mesh_size_tag_name: str,
            mesh_size_tag_step: int,
            mask_name: str,
            longitude_dim: str,
            latitude_dim: str,
            overwrite: bool,
            num_blocks: int,
            log_level: str,
            progress: bool = False):

  logging.basicConfig(format='%(levelname)s - %(asctime)s: %(message)s',
                      datefmt='%Y-%m-%dT%H:%M:%S',
                      level=log_level.upper(),
                      force=True)

  if output_path.exists() and not overwrite:
    raise ValueError(f"Output destination {output_path} already exists")

  mesh_data = read_mesh_data(mesh_path,
                             mesh_size_tag_name=mesh_size_tag_name,
                             mesh_size_tag_step=mesh_size_tag_step)

  mask = load_mask(dataset_path, mask_name, longitude_dim=longitude_dim)

  basis_values_on_grid = compute_basis(nodes=mesh_data.mesh_graph.vertices,
                                       mask=mask.to_numpy(),
                                       latitudes=mask[latitude_dim].to_numpy(),
                                       longitudes=mask[longitude_dim].to_numpy(),
                                       num_blocks=num_blocks,
                                       format='coo',
                                       dtype=np.float32,
                                       disable_progress=not progress)

  logger.info(f"Computed {basis_values_on_grid.shape[1]} basis values, saving to {output_path}")
  # noinspection PyTypeChecker
  sparse.save_npz(output_path, basis_values_on_grid)


if __name__ == "__main__":
  cli()