import logging
from typing import Literal

import numpy as np
import tqdm
import xarray as xr
from etils import epath
from scipy import sparse
from scipy.interpolate import LinearNDInterpolator

from graphcast.data_utils import fix_longitude
from graphcast.gis_utils import cartesian_crs, equirectangular_crs, get_transform, stereographic_crs

logger = logging.getLogger(__name__)


def load_mask(path: epath.PathLike, mask_name: str, longitude_dim: str = "longitude"):
  logger.info(f"Loading mask (named '{mask_name}') from {path}")
  ds = xr.open_dataset(path, engine="zarr")
  mask = ds[mask_name].isel(level=0)
  mask = fix_longitude(mask, longitude_dim=longitude_dim)
  return mask


# TODO: change the implementation to compute basis functions for the whole grid (not just masked values), taking care
#  of singular points at the south pole. Also, `fill_value` should be an argument of the function.
def compute_basis(
    nodes: np.ndarray,
    mask: np.ndarray,
    latitudes: np.ndarray,
    longitudes: np.ndarray,
    num_blocks: int | None,
    format: Literal["coo", "csr"] = "coo",
    dtype=np.float64,
    disable_progress: bool = False,
) -> sparse.spmatrix:
  cart2stereo = get_transform(cartesian_crs, stereographic_crs)
  mesh_pts = cart2stereo(nodes)
  num_mesh_pts = mesh_pts.shape[0]
  values = np.eye(num_mesh_pts, dtype=np.float32)
  logger.info(f"Building interpolator for {num_mesh_pts} basis functions.")
  interpolator = LinearNDInterpolator(mesh_pts, values, fill_value=0.0)

  latitudes, longitudes = np.meshgrid(latitudes, longitudes, indexing="ij")
  valid_coords = (longitudes[mask], latitudes[mask])
  latlon2stereo = get_transform(equirectangular_crs, stereographic_crs)
  valid_stereo = latlon2stereo(valid_coords)
  grid_pts = np.stack(valid_stereo, axis=-1)
  num_grid_pts = grid_pts.shape[0]
  logger.info(f"Found {num_grid_pts} (valid) grid points.")

  def _interp(xs: np.ndarray) -> sparse.csr_matrix:
    _interp_values = interpolator(xs)
    _interp_values = sparse.csr_matrix(_interp_values)
    return _interp_values

  logger.info("Computing basis values on grid.")
  if num_blocks is None:
    basis_values_on_grid = _interp(grid_pts)
    basis_values_on_grid = basis_values_on_grid.astype(dtype)
    if format == "coo":
      basis_values_on_grid = basis_values_on_grid.tocoo()
  else:
    if num_blocks <= 0:
      raise ValueError(f"num_blocks must be a positive integer, got {num_blocks}.")
    # FIXME: This should be processed in parallel using multithreading (not multiprocessing, `values` is large, better
    #  not to serialize it, and, for the same reason, it is better to use just a few threads).
    #  However, scipy.interpolate.LinearNDInterpolator does not release the GIL, see:
    #    1. https://github.com/scipy/scipy/blob/f1f7a63f990660662841c326cf4951b44298d20d/scipy/interpolate/_interpnd.pyx#L356-L358
    #    2. https://github.com/scipy/scipy/issues/21885
    #  Hence some other workaround has to be found.
    block_size = num_grid_pts // num_blocks
    logger.info(f"Splitting into {num_blocks} blocks of size {block_size}.")
    basis_value_blocks = []
    for block_start in tqdm.trange(0, num_grid_pts, block_size, disable=disable_progress):
      block_end = min(num_grid_pts, block_start + block_size)
      grid_pts_block = grid_pts[block_start:block_end, :]
      basis_values_on_grid = _interp(grid_pts_block)
      basis_value_blocks.append(basis_values_on_grid)
    basis_values_on_grid = sparse.vstack(basis_value_blocks, format=format, dtype=dtype)
  return basis_values_on_grid
