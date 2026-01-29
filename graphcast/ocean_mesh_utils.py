# TODO:
#   1. Some StereoMeshSizeField derived classes have a pending implementation. These correspond to criterion fields
#      based on the bathymetry (in case of drag waves) and standard error analysis tools. At the end, it should be
#      possible to reproduce exactly the same mesh as in https://doi.org/10.1007/s10236-008-0148-3
#   2. Some StereoMeshSizeField classes require several samples to average otherwise noisy criterion fields. These are
#      The one based on the hessian norm (SST might be a good target) and the one leveraging the
#      Courant–Friedrichs–Lewy condition. These require both to be implemented and some other machinery (a command in
#      ocean_mesh.py, and possibly a slurm script in leonardo/scripts) to compute the relevant statistics before a
#      field object can be instantiated.
#   3. Seamsh can ingest raster fields, use this feature to implement HessianField, BathymetryField and CourantField.
import atexit
import logging
import pathlib

import gmsh
import numpy as np
import seamsh
import xarray as xr
from osgeo import osr
from scipy.interpolate import RegularGridInterpolator

from graphcast.constants import EARTH_RADIUS
from graphcast.mesh_graph import TriangleMesh, cartesian_proj, platecarree_proj, stereographic_proj, Projection, \
  ProjectionRegistry, get_transform

logger = logging.getLogger(__name__)

@atexit.register
def _gmsh_finalize():
    logger.info(f"Finalize gmsh.")
    gmsh.finalize()

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


def read_mesh(mesh_path: pathlib.Path | str, mesh_size_tag_name: str | None, step: int = 0) \
    -> tuple[TriangleMesh, np.ndarray, np.ndarray | None]:
  """Returns the TriangleMesh and the list of boundary node indices. It assumes that gmsh has already been initialized.

  Args:
    mesh_path: path to the gmsh file containing the mesh.
  Returns:
    A tuple (mesh, boundary_nodes, mesh_size) where:
      - mesh is the computed TriangleMesh
      - boundary_nodes is a 1D numpy array of unique 0-based node indices that lie on the boundary.
      - mesh_size is the target mesh size at each node.
  """
  logger.info("Reading mesh from %s, with mesh size tag name %s and step %d",
              mesh_path, mesh_size_tag_name, step)
  mesh_path = pathlib.Path(mesh_path)
  if not mesh_path.exists():
    raise ValueError(f"Input path {mesh_path} does not exist")
  gmsh.open(str(mesh_path.absolute()))

  # Get all nodes
  node_tags, node_coords, _ = gmsh.model.mesh.get_nodes()
  node_coords = node_coords.reshape(-1, 3)

  # Determine boundary nodes from line elements (type=1).
  # As a line is bounded by two nodes, we directly get the node tags.
  line_tags, line_node_tags = gmsh.model.mesh.get_elements_by_type(1)
  if len(line_tags) > 0:
    # line_node_tags is a flat array of size 2 * num_lines
    boundary_node_tags = np.unique(line_node_tags)
  else:
    boundary_node_tags = np.array([], dtype=int)

  # Meshes produced with seamsh might contain isolated points. Their existence, number and location are determined by
  # the resolution of the shapefile containing the coastlines, and the target mesh size used when coarsening the
  # coastlines. Here we remove those points: the mesh is valid only if all nodes are used by at least one element.

  # Get valid nodes, i.e., those belonging to a triangle by querying all triangular elements (type=2).
  triangle_tags, _ = gmsh.model.mesh.get_elements_by_type(2)
  if len(triangle_tags):
    faces_node_tags = []
    for triangle_tag in triangle_tags:
      _, triangle_node_tags, _, _ = gmsh.model.mesh.get_element(triangle_tag)
      faces_node_tags.extend(triangle_node_tags)
    faces_node_tags = np.array(faces_node_tags, dtype=int)
    valid_node_tags = np.unique(faces_node_tags)
  else:
    faces_node_tags = np.empty((0,), dtype=int)
    valid_node_tags = np.empty((0,), dtype=int)

  def get_view_name(tag: int) -> str:
    return gmsh.option.getString(f"View[{gmsh.view.getIndex(tag)}].Name")

  # Get the data, i.e., the mesh size at each node.
  view_tags = gmsh.view.getTags()
  mesh_size_view_tag = None
  for view_tag in view_tags:
    name = get_view_name(view_tag)
    if name == mesh_size_tag_name:
      mesh_size_view_tag = view_tag
      break
  if mesh_size_view_tag is None:
    data_node_tags = np.empty((0,), dtype=int)
    data = np.empty((0,), dtype=float)
  else:
    # Here we need to do two things: filter the nodes, and ensure that they have the same order as in node_coords.
    data_type, data_node_tags, data, _, num_components = gmsh.view.getHomogeneousModelData(mesh_size_view_tag, step)
    assert data_type == 'NodeData' and num_components == 1

  # Node coords have the same order as in node_tags, so we can filter them directly.
  node_mask = np.isin(node_tags, valid_node_tags)
  node_tags = node_tags[node_mask] # [num_valid_node_tags]
  node_coords = node_coords[node_mask, :] # [num_valid_node_tags, 3]

  # node_tags_inv_map tells how to retrieve the index of a node in node_coords from its tag.
  node_tags_inv_map = {n: i for (i, n) in enumerate(node_tags)}

  # Faces contains only valid node tags, so there is no need to filter them.
  faces = [node_tags_inv_map[n] for n in faces_node_tags]
  faces = np.array(faces, dtype=int)
  faces = faces.reshape(-1, 3) # [num_faces, 3]

  # Boundary nodes need both filtering and reordering.
  boundary_node_mask = np.isin(boundary_node_tags, valid_node_tags)
  boundary_node_tags = boundary_node_tags[boundary_node_mask]
  boundary_nodes = [node_tags_inv_map[n] for n in boundary_node_tags] # [num_boundary_nodes]
  boundary_nodes = np.array(boundary_nodes, dtype=int)

  # Data needs both filtering and reordering.
  data_mask = np.isin(data_node_tags, valid_node_tags)
  data = data[data_mask] # [num_valid_node_tags]
  data_node_tags = data_node_tags[data_mask] # [num_valid_node_tags]
  # node_coords index i -> j = data_node_tag_inv_map[i] -> data[j]
  data_node_tags_inv_map = {n: i for (i, n) in enumerate(data_node_tags)}
  mesh_size = [data[data_node_tags_inv_map[n]] for n in node_tags]
  mesh_size = np.array(mesh_size)

  mesh = TriangleMesh(vertices=node_coords, faces=faces, node_tags=valid_node_tags)
  return mesh, boundary_nodes, mesh_size


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
