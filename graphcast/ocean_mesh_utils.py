# TODO:
#   1. Some StereoMeshSizeField derived classes have a pending implementation. These correspond to criterion fields
#      based on the bathymetry (in case of drag waves) and standard error analysis tools. At the end, it should be
#      possible to reproduce exactly the same mesh as in https://doi.org/10.1007/s10236-008-0148-3
#   2. Some StereoMeshSizeField classes require several samples to average otherwise noisy criterion fields. These are
#      The one based on the hessian norm (SST might be a good target) and the one leveraging the
#      Courant–Friedrichs–Lewy condition. These require both to be implemented and some other machinery (a command in
#      ocean_mesh.py, and possibly a slurm script in leonardo/scripts) to compute the relevant statistics before a
#      field object can be instantiated.
import functools
import pathlib
from typing import Literal

import gmsh
import numpy as np
import seamsh
import xarray as xr
from osgeo import osr
from pyproj import Transformer
from scipy.interpolate import RegularGridInterpolator

from graphcast.constants import EARTH_RADIUS
from graphcast.mesh_graph import TriangleMesh

osr.UseExceptions()
stereographic_proj = osr.SpatialReference("+proj=stere +ellps=WGS84 +lat_0=90")
cartesian_proj = osr.SpatialReference("+proj=cart +ellps=WGS84 +units=m +x_0=0 +y_0=0")
platecarree_proj = osr.SpatialReference("+proj=latlong +datum=WGS84 +no_defs")

ProjectionRegistry = {'stereographic': stereographic_proj, 'cartesian': cartesian_proj, 'platecarree': platecarree_proj}
Projection = Literal['stereographic', 'cartesian', 'platecarree']

def unpack_points(func, pack_back=True):
  """Decorator to unpack points in 2D or 3D space, apply a function and eventually pack the result back."""
  @functools.wraps(func)
  def wrapper(points: np.ndarray):
    # We assume that the coordinate dimension is the last one.
    if points.shape[-1] == 2:
      xx = points[..., 0]
      yy = points[..., 1]
      zz = None
    elif points.shape[-1] == 3:
      xx = points[..., 0]
      yy = points[..., 1]
      zz = points[..., 2]
    else:
      raise ValueError(f"Trailing dimension must be 2 or 3, got {points.shape[-1]}")
    result = func(xx, yy, zz)
    if pack_back:
      result = np.stack(result, axis=-1)
    return result

  return wrapper


def get_transform(source: osr.SpatialReference, destination: osr.SpatialReference, pack_back=True):
  """Gets a function that transforms points from one projection to another."""
  transformer = Transformer.from_proj(source.ExportToProj4(), destination.ExportToProj4())
  transform = unpack_points(transformer.transform, pack_back=pack_back)
  return transform


def map_on_grid(func, grid: xr.DataArray, longitude_dim='lon', latitude_dim='lat') -> xr.DataArray:
  """Maps a function expecting points on a regular grid in plate carree projection."""
  grid = grid.transpose(longitude_dim, latitude_dim)
  xx, yy = np.meshgrid(grid[longitude_dim], grid[latitude_dim], indexing='ij')
  xx = np.where(grid.astype(bool), xx, 0.0)
  yy = np.where(grid.astype(bool), yy, 0.0)
  xx = xx.flatten()
  yy = yy.flatten()
  points = np.stack([xx, yy], axis=-1)
  values = func(points, platecarree_proj)
  values = values.reshape(grid.shape)
  values = xr.DataArray(values, dims=grid.dims, coords=grid.coords)

  return values


class StereoMeshSizeField:
  """
  Base class that computes a mesh size field for a single criterion field in stereographic projection coordinates.

  This object depends on the reference system in two ways:
    1. how it interprets the coordinates of the points on which is evaluated,
    2. whether the mesh size (which is an edge length) is defined on the stereographic plane or in 3D.

  As the point coordinates are represented by numpy arrays, not holding information of the reference system, all the
  methods take a projection argument specifying it.

  About the second, as the meshing procedure is performed by gmsh on the stereographic plane, the `__call__` method
  return the value of the mesh size in that space.

  First the value in 3D space of the criterion field is computed, then it is clipped and rescaled to interpolate the values between
  `size_min` and `size_max`. Finally, the value of the mesh size field is rescaled to its value on the stereographic
  projection.

  See https://doi.org/10.1007/s10236-008-0148-3.
  """
  def __init__(self, size_min, size_max):
    self.size_min = size_min
    self.size_max = size_max

  def criterion(self, x: np.ndarray, projection: osr.SpatialReference) -> np.ndarray:
    """Value of the criterion field in 3D space."""
    pass

  def mesh_size_3d(self, x: np.ndarray, projection: osr.SpatialReference) -> np.ndarray:
    """Value of the mesh size field in 3D space."""
    alpha = self.criterion(x, projection)
    delta = self.size_min + (self.size_max - self.size_min) * alpha
    return delta

  def __call__(self, x: np.ndarray, projection: osr.SpatialReference) -> np.ndarray:
    """Value of the mesh size field in stereographic projection coordinates,
    possibly as a function of the coordinates in parametric space."""
    mesh_size = self.mesh_size_3d(x, projection)
    if not platecarree_proj.IsSame(projection):
      transform = get_transform(projection, stereographic_proj)
      x = transform(x)
    earth_radius_squared = EARTH_RADIUS * EARTH_RADIUS
    stereo_factor = (4 * earth_radius_squared) / (4 * earth_radius_squared + x[:, 0] ** 2 + x[:, 1] ** 2)
    return mesh_size / stereo_factor


class ConstantField(StereoMeshSizeField):
  """Stereographic mesh size field with a constant value."""
  def __init__(self, value: float):
    super().__init__(size_min=value, size_max=value)

  def criterion(self, x, projection):
    return np.full(x.shape[0], self.size_min)


class ShoreProximityField(StereoMeshSizeField):
  """Stereographic mesh size field based on the distance from the coast."""
  def __init__(self, domain: seamsh.geometry.Domain, sampling: float, field_min: float, field_max: float, size_min,
               size_max):
    super().__init__(size_min, size_max)
    self.field_min = field_min
    self.field_max = field_max
    self.distance_from_coast_f = seamsh.field.Distance(domain, sampling, projection=cartesian_proj)

  def criterion(self, x, projection):
    distance_from_coast = np.clip(self.distance_from_coast_f(x, projection), self.field_min, self.field_max)
    alpha = (distance_from_coast - self.field_min) / (self.field_max - self.field_min)
    return alpha


class GridField(StereoMeshSizeField):
  """Stereographic mesh size field whose criterion field is interpolated from values defined on a regular grid in
  plate carree projection. The grid is assumed to be an xarray DataArray with dimensions (latitude_dim, longitude_dim).
  The field extrema are computed from the quantiles q_low and q_high.
  """

  def __init__(self, grid: xr.DataArray, q_low: float, q_high: float, size_min, size_max,
               longitude_dim: str = 'lon', latitude_dim: str = 'lat'):
    super().__init__(size_min, size_max)
    self.field = RegularGridInterpolator((grid[latitude_dim], grid[longitude_dim]), grid)
    self.field_min = np.nanquantile(grid, q_low)
    self.field_max = np.nanquantile(grid, q_high)

  def criterion(self, x, projection):
    if not projection.IsSame(platecarree_proj):
      transform = get_transform(projection, platecarree_proj)
      x = transform(x)
    # noinspection PyTypeChecker
    alpha = (self.field(x) - self.field_min) / (self.field_max - self.field_min)
    return alpha


class HessianField(StereoMeshSizeField):
  pass


class BathymetryField(StereoMeshSizeField):
  pass


class CourantField(StereoMeshSizeField):
  pass


def norm_of_hessian(f: xr.DataArray, longitude_dim: str = 'lon', latitude_dim: str = 'lat') -> xr.DataArray:

  h = np.empty(f.shape + (2, 2), dtype=f.dtype)

  def diff(f: xr.DataArray, coord: str) -> xr.DataArray:
    grad = f.differentiate(coord=coord)
    # FIXME: document why this is needed, and in which approximation solves the problem.
    if coord == longitude_dim:
      grad = grad / np.cos(f[latitude_dim] * np.pi / 180.0)
    return grad

  for (i, coord_i) in enumerate([longitude_dim, latitude_dim]):
    for (j, coord_j) in enumerate([longitude_dim, latitude_dim]):
      h[..., i, j] = diff(diff(f, coord_i), coord_j)

  # FIXME: document why this is needed, and in which approximation solves the problem.
  h = 0.5 * (h + np.swapaxes(h, -1, -2))

  h_singular_values = np.linalg.svd(h, compute_uv=False, hermitian=True)
  h_norm = np.max(h_singular_values, axis=-1)
  h_norm = xr.DataArray(data=h_norm, coords=f.coords, dims=f.dims, name='norm_of_hessian')

  return h_norm


def compute_alpha(ds: xr.Dataset, eps: float = 1.0e-10, longitude_dim: str = 'lon', latitude_dim: str = 'lat') -> xr.Dataset:

  def alpha_f(da: xr.DataArray):
    # noinspection PyArgumentList
    alpha = np.sqrt(np.abs(da) / np.clip(norm_of_hessian(da), min=eps))
    return alpha

  def _compute_alpha(da: xr.DataArray):
    if longitude_dim in da.dims and latitude_dim in da.dims and np.issubdtype(da.dtype, np.floating):
      # TODO: why is `da = da.map_blocks(alpha, template=da)` slower?
      da = alpha_f(da)
    return da

  ds = ds.map(_compute_alpha)

  return ds


def read_mesh(mesh_path: pathlib.Path | str) -> TriangleMesh:
  """Returns the TriangleMesh corresponding to the given mesh file.

  Args:
    mesh_path: path to the gmsh file containing the mesh.
  Returns:
    The computed TriangleMesh

    """
  mesh_path = pathlib.Path(mesh_path)
  if not mesh_path.exists():
    raise ValueError(f"Input path {mesh_path} does not exist")
  gmsh.open(str(mesh_path.absolute()))

  node_tags, node_coords, _ = gmsh.model.mesh.get_nodes()
  node_tags_map = {tag: tag - 1 for tag in node_tags}
  node_coords = node_coords.reshape(-1, 3)

  element_tags, _ = gmsh.model.mesh.get_elements_by_type(2)
  element_nodes = []
  for element_tag in element_tags:
    _, element_node_tags, _, _ = gmsh.model.mesh.get_element(element_tag)
    element_node_indices = np.array([node_tags_map[tag] for tag in element_node_tags], dtype=int)
    element_nodes.append(element_node_indices)
  element_nodes = np.vstack(element_nodes)

  return TriangleMesh(vertices=node_coords, faces=element_nodes)


def load_domain(path: pathlib.Path, physical_name_field: str = 'featurecla',
                curve_type: str = 'bspline'):
  # Path must be a shapefile.
  if not path.name.endswith('.shp'):
    raise ValueError(f"Path must be a shapefile, got {path}")
  domain = seamsh.geometry.Domain(projection=stereographic_proj)
  domain.add_boundary_curves_shp(str(path), physical_name_field, getattr(seamsh.geometry.CurveType, curve_type.upper()))
  return domain


def coarsen_boundaries(domain: seamsh.geometry.Domain,
                       mesh_size: float,
                       x0: tuple[float, float] = (0.0, 0.0),
                       x0_projection: Projection = 'stereographic'):
  """ Creates a new Domain with the same projection and coarsened boundaries.
  """
  x0_projection = ProjectionRegistry[x0_projection]
  mesh_size_f = ConstantField(mesh_size)
  coarse = seamsh.geometry.coarsen_boundaries(domain, x0=x0, x0_projection=x0_projection, mesh_size=mesh_size_f)
  return coarse
