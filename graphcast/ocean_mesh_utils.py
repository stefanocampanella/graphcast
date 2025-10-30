# TODO:
#   1. Some StereoMeshSizeField derived classes have a pending implementation. These correspond to criterion fields
#      based on the bathymetry (in case of drag waves) and standard error analysis tools. At the end, it should be
#      possible to reproduce exactly the same mesh as in https://doi.org/10.1007/s10236-008-0148-3
#   2. Some StereoMeshSizeField classes require several samples to average otherwise noisy criterion fields. These are
#      The one based on the hessian norm (SST might be a good target) and the one leveraging the
#      Courant–Friedrichs–Lewy condition. These require both to be implemented and some other machinery (a command in
#      ocean_mesh.py, and possibly a slurm script in leonardo/scripts) to compute the relevant statistics before a
#      field object can be instantiated.
import pathlib

import gmsh
import numpy as np
import seamsh
import xarray as xr
from osgeo import osr
from scipy.interpolate import RegularGridInterpolator

from graphcast.constants import EARTH_RADIUS
from graphcast.mesh_graph import TriangleMesh, cartesian_proj, platecarree_proj, stereographic_proj, Projection, ProjectionRegistry, get_transform


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


def read_mesh(mesh_path: pathlib.Path | str) -> tuple[TriangleMesh, np.ndarray]:
  """Returns the TriangleMesh and the list of boundary node indices.

  Args:
    mesh_path: path to the gmsh file containing the mesh.
  Returns:
    A tuple (mesh, boundary_nodes) where:
      - mesh is the computed TriangleMesh
      - boundary_nodes is a 1D numpy array of unique 0-based node indices that lie on the boundary.

    """
  mesh_path = pathlib.Path(mesh_path)
  if not mesh_path.exists():
    raise ValueError(f"Input path {mesh_path} does not exist")
  gmsh.open(str(mesh_path.absolute()))

  # Get all nodes
  node_tags, node_coords, _ = gmsh.model.mesh.get_nodes()
  node_tags_map = {int(tag): int(tag) - 1 for tag in node_tags}
  node_coords = node_coords.reshape(-1, 3)

  # Get triangular elements (type=2)
  element_tags, _ = gmsh.model.mesh.get_elements_by_type(2)
  element_nodes = []
  for element_tag in element_tags:
    _, element_node_tags, _, _ = gmsh.model.mesh.get_element(int(element_tag))
    element_node_indices = np.array([node_tags_map[int(tag)] for tag in element_node_tags], dtype=int)
    element_nodes.append(element_node_indices)
  element_nodes = np.vstack(element_nodes) if element_nodes else np.empty((0, 3), dtype=int)

  # Determine boundary nodes from 1D elements (lines). All 1D elements in a 2D surface mesh are boundary edges.
  line_element_tags, line_node_tags = gmsh.model.mesh.get_elements_by_type(1)
  if line_element_tags.size > 0:
    # line_node_tags is a flat array of size 2 * num_lines
    boundary_nodes_indices = np.unique([node_tags_map[int(tag)] for tag in line_node_tags])
  else:
    boundary_nodes_indices = np.array([], dtype=int)

  # Meshes produced with seamsh might contain isolated points. Their existence, number and location are determined by
  # the resolution of the shapefile containing the coastlines, and the target mesh size used when coarsening the
  # coastlines. Here we remove those points: the mesh is valid only if all nodes are used by at least one element.
  valid_nodes = np.unique(element_nodes.reshape(-1))
  valid_nodes_mask = np.isin(range(node_coords.shape[0]), valid_nodes)
  valid_node_map = {v: n for (n, v) in enumerate(valid_nodes)}
  valid_node_map_f = np.vectorize(valid_node_map.get)
  node_coords = node_coords[valid_nodes_mask, :]
  element_nodes = valid_node_map_f(element_nodes)
  boundary_nodes_indices = boundary_nodes_indices[np.isin(boundary_nodes_indices, valid_nodes)]
  boundary_nodes_indices = valid_node_map_f(boundary_nodes_indices)
  mesh = TriangleMesh(vertices=node_coords, faces=element_nodes)
  return mesh, boundary_nodes_indices


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
