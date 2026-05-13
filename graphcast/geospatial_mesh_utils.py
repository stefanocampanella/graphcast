# TODO:
#   1. Some StereoMeshSizeField classes require several samples to average otherwise noisy criterion fields. These are
#      The one based on the hessian norm (SST might be a good target) and the one leveraging the
#      Courant–Friedrichs–Lewy condition. These require both to be implemented and some other machinery (a command in
#      ocean_mesh.py, and possibly a slurm script in leonardo/scripts) to compute the relevant statistics before a
#      field object can be instantiated.
#   3. Seamsh can ingest raster fields, use this feature to implement HessianField, BathymetryField and CourantField.
import atexit
import logging
import pathlib
from collections.abc import Iterable
from typing import Any, Literal

import gmsh
import numpy as np
import seamsh
import xarray as xr
from osgeo import osr
from scipy.interpolate import RectBivariateSpline
from scipy.ndimage import gaussian_filter

from graphcast.gis_utils import (
  CoordinateReferenceSystem,
  CRSName,
  CRSRegistry,
  cartesian_srs,
  get_transform,
  stereographic_srs,
  xarray_to_gdal_raster,
)
from graphcast.mesh_graph import MeshData, MeshGraph, TriangleMesh, faces_to_edges

logger = logging.getLogger(__name__)


@atexit.register
def _maybe_gmsh_finalize():
  if gmsh.is_initialized():
    logger.info("Finalize gmsh.")
    gmsh.finalize()


class StereoMeshSizeField:
  """
  Base class that computes a mesh size field in stereographic projection coordinates.

  This object depends on the reference system in two ways:
    1. how it interprets the coordinates of the points on which is evaluated, and
    2. whether the mesh size (which is an edge length) is defined on the stereographic plane or in 3D.

  As the point coordinates are represented by numpy arrays, not holding information of the reference system, all the
  methods take a projection argument specifying it.

  About the second point, as the meshing procedure is performed by gmsh on the stereographic plane, the `__call__`
  method returns the value of the mesh size in that space.

  First, the value in 3D space of the criterion field is computed; then the mesh size field is rescaled to its value on
  the stereographic projection.

  See https://doi.org/10.1007/s10236-008-0148-3.
  """

  def mesh_size_3d(self, x: np.ndarray, projection: CoordinateReferenceSystem) -> np.ndarray:
    """Value of the mesh size field in 3D space."""
    pass

  def __call__(self, x: np.ndarray, projection: CoordinateReferenceSystem) -> np.ndarray:
    """Value of the mesh size field in stereographic projection coordinates,
    possibly as a function of the coordinates in parametric space."""
    mesh_size = self.mesh_size_3d(x, projection)
    if not stereographic_srs.IsSame(projection):
      transform = get_transform(projection, stereographic_srs)
      x = transform(x)
    earth_radius_squared = stereographic_srs.GetSemiMajor() * stereographic_srs.GetSemiMinor()
    stereo_factor = (4 * earth_radius_squared) / (
      4 * earth_radius_squared + x[:, 0] ** 2 + x[:, 1] ** 2
    )
    return mesh_size / stereo_factor


class UniformField(StereoMeshSizeField):
  """Stereographic mesh size field with a constant value."""

  def __init__(self, value: float):
    self.value = value

  def mesh_size_3d(self, x, projection):
    return np.full(x.shape[0], self.value)


class BoundedStereoMeshSizeField(StereoMeshSizeField):
  """
  Base class that computes a mesh size field for using a criterion field in stereographic projection coordinates.
  Computes the criterion field in 3D space, then clips it, and finally rescale it to interpolate the values
  between `size_min` and `size_max`.

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


class BoundaryProximityField(BoundedStereoMeshSizeField):
  """Stereographic mesh size field based on the distance from the coast."""

  def __init__(
    self,
    filepath: str | pathlib.Path,
    physical_name_field: str,
    curve_type: str,
    sampling: float,
    field_min: float,
    field_max: float,
    size_min,
    size_max,
  ):
    super().__init__(size_min, size_max)
    if isinstance(filepath, str):
      filepath = pathlib.Path(filepath)
    domain = load_domain(filepath, physical_name_field=physical_name_field, curve_type=curve_type)

    self.field_min = field_min
    self.field_max = field_max
    self.distance_from_boundary = seamsh.field.Distance(domain, sampling, projection=cartesian_srs)

  def criterion(self, x, projection):
    value = np.clip(self.distance_from_boundary(x, projection), self.field_min, self.field_max)
    alpha = (value - self.field_min) / (self.field_max - self.field_min)
    return alpha


class RasterField(BoundedStereoMeshSizeField):
  """Stereographic mesh size field whose criterion field is interpolated from values defined on a regular grid in
  plate carree projection. The grid is assumed to be an xarray DataArray with dimensions (latitude_dim, longitude_dim).
  The field extrema are computed from the quantiles q_low and q_high.
  """

  def __init__(
    self,
    grid: xr.DataArray,
    size_min,
    size_max,
    q_low: float = 0.0,
    q_high: float = 1.0,
    longitude_dim: str = "lon",
    latitude_dim: str = "lat",
    srs_name: str = "cartesian",
  ):
    super().__init__(size_min, size_max)
    if q_low <= 0.0:
      self.field_min = np.nanmin(grid)
    else:
      self.field_min = np.nanquantile(grid, q_low)
    if q_high >= 1.0:
      self.field_max = np.nanmax(grid)
    else:
      self.field_max = np.nanquantile(grid, q_high)
    grid = grid.fillna(self.field_max)
    gdal_raster = xarray_to_gdal_raster(
      da=grid, latitude_dim=latitude_dim, longitude_dim=longitude_dim
    )
    self.field = seamsh.field.Raster(gdal_raster)

  # noinspection PyTypeChecker
  def criterion(self, x, projection):
    value = np.clip(self.field(x, projection), self.field_min, self.field_max)
    alpha = (value - self.field_min) / (self.field_max - self.field_min)
    return alpha


class BathymetryField(RasterField):
  """Stereographic mesh size field based on the bathymetry."""

  def __init__(self, filepath: str | pathlib.Path, var_name: str, size_min, size_max, **kwargs):
    """It assumes that variable contains positive depth values."""
    grid_ds = xr.open_dataset(filepath, engine="zarr")
    grid_da = grid_ds[var_name].load()
    grid_da = grid_da / grid_da.max()
    assert np.all(np.logical_or(grid_da.isnull(), grid_da >= 0.0)), (
      f"Variable {var_name} contain negative depth values."
    )
    bathy_sqrt = np.sqrt(grid_da)
    super().__init__(bathy_sqrt, size_min, size_max, **kwargs)


class BathymetryHessianField(RasterField):
  def __init__(
    self,
    filepath: str | pathlib.Path,
    var_name: str,
    size_min,
    size_max,
    eps: float = 1.0e-5,
    longitude_dim: str = "lon",
    latitude_dim: str = "lat",
    spline_kwargs: dict[str, Any] | None = None,
    filter_kwargs: dict[str, Any] | None = None,
    **kwargs,
  ):
    grid_ds = xr.open_dataset(filepath, engine="zarr")
    grid_da = grid_ds[var_name].load()
    grid_da = grid_da / grid_da.max()
    hnorm = self.norm_of_hessian(
      grid_da,
      longitude_dim=longitude_dim,
      latitude_dim=latitude_dim,
      filter_kwargs=filter_kwargs,
      spline_kwargs=spline_kwargs,
    )
    assert np.all(hnorm >= 0.0), "Computed Hessian contains negative values."
    hnorm_invsqrt = 1 / np.clip(np.sqrt(hnorm), a_min=eps, a_max=None)
    hnorm_invsqrt = xr.where(grid_da.isnull(), np.nan, hnorm_invsqrt)
    super().__init__(
      hnorm_invsqrt,
      size_min,
      size_max,
      longitude_dim=longitude_dim,
      latitude_dim=latitude_dim,
      **kwargs,
    )

  @staticmethod
  def norm_of_hessian(
    da: xr.DataArray,
    longitude_dim: str = "lon",
    latitude_dim: str = "lat",
    filter_kwargs: dict[str, Any] | None = None,
    spline_kwargs: dict[str, Any] | None = None,
  ) -> xr.DataArray:
    """Computes the norm of the Hessian matrix of a 2D array. The Hessian matrix is approximated by a spline,
    and assumes latitude and longitude are in degrees."""

    spline_kwargs = spline_kwargs or {}
    # Use canonical coordinates order and fill missing values.
    da = da.transpose(latitude_dim, longitude_dim)
    da = da.fillna(0.0)
    data = da.to_numpy()
    filter_kwargs = filter_kwargs or {}
    sigma = filter_kwargs.pop("sigma", 1.0)
    data = gaussian_filter(data, sigma, **filter_kwargs)
    latitudes = da[latitude_dim].to_numpy()
    longitudes = da[longitude_dim].to_numpy()
    hessian_matrix = np.empty(data.shape + (2, 2), dtype=data.dtype)

    def direction(coord: str) -> tuple[int, int]:
      if coord == longitude_dim:
        return 0, 1
      elif coord == latitude_dim:
        return 1, 0
      else:
        raise ValueError(f"Unknown coordinate {coord}.")

    def fix_for_latitudes(z_di: np.ndarray, eps=1e-10) -> np.ndarray:
      latitudes_grid = np.tile(latitudes.reshape(-1, 1), (1, len(longitudes)))
      corrective_factor = 1 / np.clip(np.cos(latitudes_grid * np.pi / 180.0), a_min=eps, a_max=None)
      return np.where(
        np.logical_or(np.isclose(latitudes_grid, 90.0), np.isclose(latitudes_grid, -90.0)),
        np.zeros_like(z_di),
        z_di * corrective_factor,
      )

    def grad(z: np.ndarray, coord: str) -> np.ndarray:
      z_spline = RectBivariateSpline(latitudes, longitudes, z, **spline_kwargs)
      z_di = z_spline.partial_derivative(*direction(coord))(latitudes, longitudes)
      # FIXME: document why this is needed, and in which approximation solves the problem.
      if coord == longitude_dim:
        z_di = fix_for_latitudes(z_di)
      return z_di

    for i, coord_i in enumerate([longitude_dim, latitude_dim]):
      for j, coord_j in enumerate([longitude_dim, latitude_dim]):
        hessian_matrix[..., i, j] = grad(grad(data, coord_i), coord_j)

    # FIXME: document why this is needed, and in which approximation solves the problem.
    hessian_matrix = 0.5 * (hessian_matrix + np.swapaxes(hessian_matrix, -1, -2))
    hessian_matrix_singular_values = np.linalg.svd(hessian_matrix, compute_uv=False, hermitian=True)
    hessian_matrix_norm = np.max(hessian_matrix_singular_values, axis=-1)
    hessian_matrix_norm = xr.DataArray(
      data=hessian_matrix_norm, coords=da.coords, dims=da.dims, name=f"{da.name}_hessian_norm"
    )

    return hessian_matrix_norm


FieldName = Literal["constant", "shore_proximity", "bathymetry", "bathymetry_hessian"]
FieldsRegistry = {
  "uniform": UniformField,
  "shore_proximity": BoundaryProximityField,
  "bathymetry": BathymetryField,
  "bathymetry_hessian": BathymetryHessianField,
}


class CompositeMeshSizeField(StereoMeshSizeField):
  """Stereographic mesh size field which takes the minimum of other fields."""

  def __init__(self, fields_config: Iterable[dict[str, Any]], prefix: pathlib.Path | None = None):
    fields = {}
    for config in fields_config:
      field_name = config.pop("name")
      if field_name not in FieldsRegistry:
        raise ValueError(
          f"Unknown field type {field_name}. Available types are {FieldsRegistry.keys()}."
        )
      if prefix is not None:
        for key in config.keys():
          if key == "filepath" or key.endswith("_path"):
            config[key] = prefix / config[key]
      logger.info("Creating field %s with config %s.", field_name, config)
      fields[field_name] = FieldsRegistry[field_name](**config)
    self.fields = fields

  def mesh_size_3d(self, x: np.ndarray, projection: osr.SpatialReference) -> np.ndarray:
    return np.minimum.reduce([field.mesh_size_3d(x, projection) for field in self.fields.values()])


# TODO: update implementation to save boundary elements into TriangleMesh, and read reference system from mesh file and
#  save it as well in the output file.
def read_mesh(
  mesh_path: pathlib.Path | str,
  mesh_size_tag_name: str | None,
  srs_attribute_name: str = "Projection",
  step: int = 0,
) -> tuple[TriangleMesh, np.ndarray | None]:
  """Returns the TriangleMesh and the list of boundary node indices. It assumes that gmsh has already been initialized.

  Args:
    mesh_path: path to the gmsh file containing the mesh.
  Returns:
    A tuple (mesh, boundary_nodes, mesh_size) where:
      - mesh is the computed TriangleMesh
      - boundary_nodes is a 1D numpy array of unique 0-based node indices that lie on the boundary.
      - mesh_size is the target mesh size at each node.
  """
  logger.info(
    "Reading mesh from %s, with mesh size tag name %s and step %d",
    mesh_path,
    mesh_size_tag_name,
    step,
  )
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
    boundary_node_tags = line_node_tags.reshape(-1, 2)
  else:
    boundary_node_tags = np.empty((0, 2), dtype=int)

  # Meshes produced with seamsh might contain isolated points. Their existence, number and location are determined by
  # the resolution of the shapefile containing the coastlines, and the target mesh size used when coarsening the
  # coastlines. Here we remove those points: the mesh is valid only if all nodes are used by at least one element.

  # Get valid nodes, i.e., those belonging to a triangle by querying all triangular elements (type=2).
  triangle_tags, triangle_node_tags = gmsh.model.mesh.get_elements_by_type(2)
  if len(triangle_tags):
    faces_node_tags = triangle_node_tags.reshape(-1, 3)
    valid_node_tags = np.unique(faces_node_tags)
  else:
    faces_node_tags = np.empty((0, 3), dtype=int)
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
    data_type, data_node_tags, data, _, num_components = gmsh.view.getHomogeneousModelData(
      mesh_size_view_tag, step
    )
    assert data_type == "NodeData" and num_components == 1

  # Node coords have the same order as in node_tags, so we can filter them directly.
  node_mask = np.isin(node_tags, valid_node_tags)
  node_tags = node_tags[node_mask]  # [num_valid_node_tags]
  node_coords = node_coords[node_mask, :]  # [num_valid_node_tags, 3]

  # node_tags_inv_map tells how to retrieve the index of a node in node_coords from its tag.
  node_tags_inv_map = {n: i for (i, n) in enumerate(node_tags)}
  node_tags_inv_f = np.vectorize(lambda n: node_tags_inv_map.get(n, -1))

  # Faces contains only valid node tags, so there is no need to filter them.
  faces = node_tags_inv_f(faces_node_tags)  # [num_faces, 3]

  # Compute the boundary nodes mask
  boundary_node_mask = np.isin(
    range(len(node_coords)), node_tags_inv_f(np.unique(boundary_node_tags))
  )  # [num_valid_node_tags]

  # Data might need both filtering and reordering.
  data_mask = np.isin(data_node_tags, valid_node_tags)
  data = data[data_mask]  # [num_valid_node_tags]
  data_node_tags = data_node_tags[data_mask]  # [num_valid_node_tags]
  # node_coords index i -> j = data_node_tag_inv_map[i] -> data[j]
  data_node_tags_inv_map = {n: i for (i, n) in enumerate(data_node_tags)}
  mesh_size = [data[data_node_tags_inv_map[n]] for n in node_tags]
  mesh_size = np.array(mesh_size)

  srs_string_type, srs_string = gmsh.model.get_attribute(srs_attribute_name)
  assert srs_string_type.upper() == "WKT"

  mesh = TriangleMesh(
    vertices=node_coords,
    faces=faces,
    boundary=boundary_node_mask,
    node_tags=valid_node_tags,
    spatial_reference_system=srs_string,
  )
  return mesh, mesh_size


def load_domain(
  path: pathlib.Path, physical_name_field: str = "featurecla", curve_type: str = "bspline"
):
  # Path must be a shapefile.
  if not path.name.endswith(".shp"):
    raise ValueError(f"Path must be a shapefile, got {path}")
  domain = seamsh.geometry.Domain(projection=stereographic_srs)
  domain.add_boundary_curves_shp(
    str(path), physical_name_field, getattr(seamsh.geometry.CurveType, curve_type.upper())
  )
  return domain


def coarsen_boundaries(
  domain: seamsh.geometry.Domain,
  mesh_size: float,
  x0: tuple[float, float] = (0.0, 0.0),
  x0_projection: CRSName = "stereographic",
):
  """Creates a new Domain with the same projection and coarsened boundaries."""
  x0_projection = CRSRegistry[x0_projection]
  mesh_size_f = UniformField(mesh_size)
  coarse = seamsh.geometry.coarsen_boundaries(
    domain, x0=x0, x0_projection=x0_projection, mesh_size=mesh_size_f
  )
  return coarse


def read_mesh_data(
  mesh_path,
  gmsh_verbosity: int = 2,
  mesh_size_tag_name: str | None = None,
  mesh_size_tag_step: int = 0,
) -> MeshData:
  if not gmsh.is_initialized():
    logger.info("Initialize gmsh.")
    gmsh.initialize()
  gmsh.option.setNumber("General.Verbosity", gmsh_verbosity)

  mesh_size_tag_name = mesh_size_tag_name or "MeshSize"
  mesh, mesh_size = read_mesh(
    mesh_path=mesh_path, mesh_size_tag_name=mesh_size_tag_name, step=mesh_size_tag_step
  )
  mesh_license = gmsh.model.getAttribute("license")
  mesh_description = gmsh.model.getAttribute("description")
  graph = MeshGraph(
    vertices=mesh.vertices,
    edges=faces_to_edges(mesh.faces),
    faces=mesh.faces,
    boundary=mesh.boundary,
    spatial_reference_system=mesh.spatial_reference_system,
  )
  logger.info(
    "Mesh graph contains %d vertices and %d edges.", len(graph.vertices), len(graph.edges[0])
  )
  mesh_data = MeshData(
    mesh_graph=graph, mesh_size=mesh_size, description=mesh_license, license=mesh_description
  )
  return mesh_data
