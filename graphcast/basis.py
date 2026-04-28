import logging

import click
import numpy as np
import tqdm
import xarray as xr
from etils import epath
from scipy import sparse
from scipy.interpolate import LinearNDInterpolator

from graphcast.data_utils import fix_longitude
from graphcast.geospatial_mesh_utils import read_mesh_data
from graphcast.gis_utils import get_transform, cartesian_srs, equirectangular_srs, stereographic_srs

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
@click.option("--overwrite/--no-overwrite",
              default=False,
              help="Whether to overwrite existing outputs",
              is_flag=True)
@click.option("--n-blocks",
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
            overwrite: bool,
            n_blocks: int,
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
  cart2stereo = get_transform(cartesian_srs, stereographic_srs)
  mesh_pts = cart2stereo(mesh_data.mesh_graph.vertices)
  num_mesh_pts = mesh_pts.shape[0]
  values = np.eye(num_mesh_pts, dtype=np.float32)
  logger.info(f"Building interpolator for {num_mesh_pts} basis functions.")
  interp = LinearNDInterpolator(mesh_pts, values, fill_value=0.0)

  logger.info(f"Loading mask ({mask_name}) from {dataset_path}")
  ds = xr.open_dataset(dataset_path, engine='zarr')
  da = ds[mask_name].isel(level=0)
  da = fix_longitude(da, longitude_dim=longitude_dim)
  mask = da.to_numpy()
  lats, lons = np.meshgrid(da['lat'], da['lon'], indexing='ij')
  valid_coords = (lons[mask], lats[mask])
  latlon2stereo = get_transform(equirectangular_srs, stereographic_srs)
  valid_stereo = latlon2stereo(valid_coords)
  grid_pts = np.stack(valid_stereo, axis=-1)

  num_grid_pts = grid_pts.shape[0]
  block_size = num_grid_pts // n_blocks
  logger.info(f"Found {num_grid_pts} valid grid points, splitting into {n_blocks} blocks of size {block_size}")
  # FIXME: This should be processed in parallel using multithreading (not multiprocessing, `values` is large, better
  #  not to serialize it, and, for the same reason, it is better to use just a few threads).
  #  However, scipy.interpolate.LinearNDInterpolator does not release the GIL, see:
  #    1. https://github.com/scipy/scipy/blob/f1f7a63f990660662841c326cf4951b44298d20d/scipy/interpolate/_interpnd.pyx#L356-L358
  #    2. https://github.com/scipy/scipy/issues/21885
  #  Hence some other workaround has to be found.
  basis_value_blocks = []
  for block_start in tqdm.trange(0, num_grid_pts, block_size, disable=not progress):
    block_end = min(num_grid_pts, block_start + block_size)
    grid_pts_block = grid_pts[block_start:block_end, :]
    basis_values_on_grid = interp(grid_pts_block)
    basis_values_on_grid = sparse.csr_matrix(basis_values_on_grid)
    basis_value_blocks.append(basis_values_on_grid)
  logger.info("Merging blocks")
  basis_values_on_grid = sparse.vstack(basis_value_blocks)

  logger.info(f"Computed {basis_values_on_grid.shape[1]} basis values, saving to {output_path}")
  sparse.save_npz(output_path, basis_values_on_grid)


if __name__ == "__main__":
  cli()