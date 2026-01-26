# Copyright 2023 DeepMind Technologies Limited.
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
"""Tools for converting from regular grids on a sphere, to triangular meshes."""
# TODO: change type annotations to use multimesh_graph types

import logging
from functools import partial
from typing import Union, Iterable, Literal, Tuple, Dict, NamedTuple

import numpy as np
import numpy.typing as npt
import scipy
import trimesh
import xarray

from graphcast.constants import EARTH_RADIUS
from graphcast.mesh_graph import TriangleMesh, MeshGraph, faces_to_edges, mesh_to_wgs
from graphcast.typed_graph import Context, NodeSet, EdgeSet, EdgeSetKey, EdgesIndices, TypedGraph

logger = logging.getLogger(__name__)


class Box(NamedTuple):
  lat_min: float
  lat_max: float
  lon_min: float
  lon_max: float


Mesh = Union[TriangleMesh, MeshGraph]


# TODO: update tests and usage in notebooks (e.g. mesh_comparison.ipynb) to take into account that radius of the Earth
#  is now used (no longer unit sphere)
# FIXME: implement this using pyproj
def _grid_lat_lon_to_coordinates(
    grid_latitude: np.ndarray, grid_longitude: np.ndarray) -> np.ndarray:
  """Lat [num_lat] lon [num_lon] to 3d coordinates [num_lat, num_lon, 3]."""
  # Convert to spherical coordinates phi and theta defined in the grid.
  # Each [num_latitude_points, num_longitude_points]
  phi_grid, theta_grid = np.meshgrid(
      np.deg2rad(grid_longitude),
      np.deg2rad(90 - grid_latitude))

  # [num_latitude_points, num_longitude_points, 3]
  # Note this assumes unit radius, since for now we model the earth as a
  # sphere of unit radius, and keep any vertical dimension as a regular grid.
  coordinates_on_unit_sphere = np.stack(
      [np.cos(phi_grid)*np.sin(theta_grid),
       np.sin(phi_grid)*np.sin(theta_grid),
       np.cos(theta_grid)], axis=-1)
  return EARTH_RADIUS * coordinates_on_unit_sphere


def radius_query_indices(
    *,
    grid_latitude: np.ndarray,
    grid_longitude: np.ndarray,
    mesh: Mesh,
    radius: float | np.ndarray,
    mask: xarray.DataArray | None = None,
    workers: int = 1) -> tuple[np.ndarray, np.ndarray]:
  """Returns mesh-grid edge indices for radius query.

  Args:
    grid_latitude: Latitude values for the grid [num_lat_points]
    grid_longitude: Longitude values for the grid [num_lon_points]
    mesh: Mesh object.
    radius: Radius of connectivity in R3. for a sphere of unit radius.
    mask: Boolean mask of shape [num_lat_points, num_lon_points] to filter.

  Returns:
    tuple with `grid_indices` and `mesh_indices` indicating edges between the
    grid and the mesh such that the distances in a straight line (not geodesic)
    are smaller than or equal to `radius`.
    * grid_indices: Indices of shape [num_edges], that index into a
      [num_lat_points, num_lon_points] grid, after flattening the leading axes.
    * mesh_indices: Indices of shape [num_edges], that index into mesh.vertices.
  """
  if mask is None:
    mask_data = np.ones((grid_latitude.shape[0], grid_longitude.shape[0]), dtype=bool)
  else:
    # FIXME: the following can probably have a more straightforward implementation, the crucial point is to be consistent
    #  with the order of lat/lon dimensions and indexing
    assert np.array_equal(mask['lat'].to_numpy(), grid_latitude)
    assert np.array_equal(mask['lon'].to_numpy(), grid_longitude)
    mask_data = mask.transpose('lat', 'lon').to_numpy()
  # [num_grid_points=num_lat_points * num_lon_points]
  grid_mask = mask_data.reshape([-1])
  # [num_grid_points=num_lat_points * num_lon_points, 3]
  grid_positions = _grid_lat_lon_to_coordinates(grid_latitude, grid_longitude).reshape([-1, 3])
  # [num_mesh_points, 3]
  mesh_positions = mesh.vertices
  # [num_valid_grid_points=sum(grid_mask)]
  valid_grid_positions = grid_positions[grid_mask]

  # NOTICE:
  #   1. the number of grid points per mesh point is not constant, so `query_ball_point` return an array of lists,
  #     rather than a 2d array,
  #   2. the ball is in 3D space, so the distances are not geodesic,
  #   3. if radius is a number, then building a KDTree of grid points and querying mesh nodes is the same
  #     as building a KDTree of mesh nodes and querying grid points: for varying radius this is not the case,
  #   4. the original implementation was more readable, but unfeasible for large grids/meshes.
  kd_tree = scipy.spatial.cKDTree(valid_grid_positions)
  # [num_grid_points, num_mesh_points_per_grid_point (variable)]
  # noinspection PyTypeChecker
  query_indices: npt.NDArray[list[int]] = kd_tree.query_ball_point(x=mesh_positions, r=radius, workers=workers)
  _, masked_to_unmasked_fn = get_masking_indices_fns(grid_mask)
  # noinspection PyTypeChecker
  grid_senders = np.concatenate(list(map(masked_to_unmasked_fn, query_indices)), axis=0).astype(int)
  mesh_receivers = np.repeat(np.arange(mesh_positions.shape[0], dtype=int),
                             np.fromiter(map(len, query_indices), dtype=int))
  return grid_senders, mesh_receivers


def get_mesh_to_grid_edges(
    *,
    grid_latitude: np.ndarray,
    grid_longitude: np.ndarray,
    mesh: Mesh,
    mask: None | xarray.DataArray = None) -> tuple[np.ndarray, np.ndarray]:
  """Returns mesh-grid edge indices for grid points contained in mesh triangles.

  Args:
    grid_latitude: Latitude values for the grid [num_lat_points]
    grid_longitude: Longitude values for the grid [num_lon_points]
    mesh: Mesh object.
    mask: Boolean mask of shape [num_lat_points, num_lon_points] to filter.

  Returns:
    tuple with `grid_indices` and `mesh_indices` indicating edges between the
    grid and the mesh vertices of the triangle that contain each grid point.
    The number of edges is always num_lat_points * num_lon_points * 3
    * grid_indices: Indices of shape [num_edges], that index into a
      [num_lat_points, num_lon_points] grid, after flattening the leading axes.
    * mesh_indices: Indices of shape [num_edges], that index into mesh.vertices.
  """
  if mask is None:
    mask_data = np.ones((grid_latitude.shape[0], grid_longitude.shape[0]), dtype=bool)
  else:
    assert np.array_equal(mask['lat'].to_numpy(), grid_latitude)
    assert np.array_equal(mask['lon'].to_numpy(), grid_longitude)
    mask_data = mask.transpose('lat', 'lon').to_numpy()
  # [num_grid_points=num_lat_points * num_lon_points]
  grid_mask = mask_data.reshape([-1])
  # [num_grid_points=num_lat_points * num_lon_points, 3]
  grid_positions = _grid_lat_lon_to_coordinates(grid_latitude, grid_longitude).reshape([-1, 3])
  # [num_valid_grid_points=sum(grid_mask)]
  valid_grid_positions = grid_positions[grid_mask, :]
  mesh_senders, valid_grid_receivers = get_mesh_to_points_edges(senders_mesh=mesh,
                                                                receivers_position=valid_grid_positions)
  _, masked_to_unmasked_fn = get_masking_indices_fns(grid_mask)
  # noinspection PyTypeChecker
  grid_receivers: npt.NDArray[int] = masked_to_unmasked_fn(valid_grid_receivers)

  return mesh_senders, grid_receivers


# TODO: add tests
# TODO: some notebooks might have used `get_mesh_to_mesh_edges` instead of `get_mesh_to_points_edges`,
#  both the signature and direction of edges have to be refactore.
def get_mesh_to_points_edges(
    *,
    senders_mesh: Mesh,
    receivers_position: np.ndarray) \
    -> tuple[np.ndarray, np.ndarray]:
  """Returns edges connecting each point in `receivers_potition` to the vertices of the `senders_mesh`
  whose face is closest to.

  Args:
    senders_mesh: Mesh object.
    receivers_position: Array of points in R3, shape [num_receivers_points, 3].

  Returns:
    senders, receivers tuple of indices indicating edges between the mesh and the points.
    The number of edges is always num_points * 3.
  """

  mesh_trimesh = trimesh.Trimesh(vertices=senders_mesh.vertices, faces=senders_mesh.faces)
  # [num_senders_mesh_vertices] with mesh face indices for each sender mesh vertex.
  _, _, query_face_indices = trimesh.proximity.closest_point(mesh_trimesh, receivers_position)
  # [3 * num_senders_mesh_vertices]
  senders = senders_mesh.faces[query_face_indices].reshape([-1])
  # [3 * num_senders_mesh_vertices]
  receivers = np.tile(np.arange(len(receivers_position), dtype=int).reshape([-1, 1]), [1, 3]).reshape([-1])

  return senders, receivers


# TODO: add tests
def get_connected_mesh_nodes(grid_lat: np.ndarray,
                             grid_lon: np.ndarray,
                             mesh_graph: Mesh,
                             mask: xarray.DataArray,
                             query_radius: float | np.ndarray,
                             workers: int = 1) -> set[int]:
  """Returns the set of mesh vertices connected to a valid grid point.

  It does so by excluding the mesh vertices that are not connected to a valid grid point by at least one edge of the
  Grid2Mesh and/or the Mesh2Grid-like graphs described in the GraphCast paper.

  Args:
    grid_lat: Latitude values for the grid [num_lat_points]
    grid_lon: Longitude values for the grid [num_lon_points]
    mesh_graph: MultiMeshGraph or TriangleMesh object.
    mask: Boolean mask of shape [num_lat_points, num_lon_points]
    query_radius: Radius of connectivity in R3 for a sphere of unit radius.
  Returns:
    Set of indices of mesh vertices connected to a valid grid point.
  """
  if isinstance(query_radius, np.ndarray):
    assert mesh_graph.vertices.shape[0] == query_radius.shape[0], \
      "The number of vertices in the mesh graph must match the number of query radii."

  (_, mesh_receivers) = radius_query_indices(
    grid_latitude=grid_lat,
    grid_longitude=grid_lon,
    mesh=mesh_graph,
    radius=query_radius,
    mask=mask,
    workers=workers)

  (mesh_senders, _) = get_mesh_to_grid_edges(
    grid_latitude=grid_lat,
    grid_longitude=grid_lon,
    mesh=mesh_graph,
    mask=mask)

  grid2mesh_connected_mesh_vertices = set(mesh_receivers)
  mesh2grid_connected_mesh_vertices = set(mesh_senders)
  connected_mesh_vertices = set.intersection(grid2mesh_connected_mesh_vertices, mesh2grid_connected_mesh_vertices)

  return connected_mesh_vertices


# TODO: add tests
# TODO: reimplement the following using get_masking_indices_fns
def mask_mesh(marked_vertices: Iterable[int], mesh: Mesh, mode: Literal['any', 'all'] = 'any') -> Tuple[Mesh, Dict[int, int]]:
  """Filters the mesh to include only vertices belonging to triangles with at least one marked vertex (when `mode='any'`),
  or with all marked vertices (when `mode='all'`).

  Args:
    marked_vertices: Set of vertices connected to a valid grid point.
    mesh: MeshGraph object.
  Returns:
    Tuple containint a masked multimesh graph, and a mapping from the old vertex indices to the new vertex indices.
  """
  predicate = np.any if mode == 'any' else np.all
  if isinstance(marked_vertices, set):
    marked_vertices = list(marked_vertices)
  valid_faces_mask = predicate(np.isin(mesh.faces, marked_vertices), axis=1)
  valid_faces = mesh.faces[valid_faces_mask, :]
  if len(valid_faces) > 0:
    valid_vertices = np.unique(np.concatenate(valid_faces))
    vertices = mesh.vertices[valid_vertices, :]
    valid_vertices_map = {v: i for (i, v) in enumerate(valid_vertices)}
    valid_vertices_map_f = np.vectorize(valid_vertices_map.get)
    faces = valid_vertices_map_f(valid_faces)
  else:
    valid_vertices_map = dict()
    faces = np.empty((0, 3), dtype=int)
    vertices = np.empty((0, 3), dtype=float)

  if isinstance(mesh, MeshGraph):
    masked_mesh = MeshGraph(vertices=vertices, edges=faces_to_edges(faces), faces=faces)
  else:
    masked_mesh = TriangleMesh(vertices=vertices, faces=faces)

  # The np.unique function can return an array containing additional information, i.e. made of tuples. This confuses the type checker.
  # noinspection PyTypeChecker
  return masked_mesh, valid_vertices_map


def get_masking_indices_fns(mask: npt.NDArray[np.bool], raise_error: bool = True, default_value=-1):
  """Returns two functions that convert between masked and unmasked indices.

  Args:
    mask: Boolean mask of shape [num_elements].
    raise_error: If True, raise an error if an index is invalid.
    default_value: Value to return if an invalid index is encountered and `raise_error=False`.

  Returns:
    Tuple of two vectorized functions (`unmasked_to_masked_fn`, `masked_to_unmasked_fn`),
    that convert between masked and unmasked indices.
  """
  assert mask.ndim == 1
  # Indices of an unmasked array
  unmasked_array_indices = np.arange(len(mask), dtype=int)
  # Array whose values are the indices of mask that are True, but also a mapping (via subscript operator)
  # from the index of a masked array to the corresponding index in the unmasked array
  valid_indices = unmasked_array_indices[mask]
  num_valid_indices = len(valid_indices)
  # Indices of a masked array
  masked_array_indices = np.arange(num_valid_indices, dtype=int)
  # Mapping from unmasked indices to masked indices
  valid_indices_inverse_map = {v: i for (i, v) in zip(masked_array_indices, valid_indices)}

  @partial(np.vectorize, otypes=[int])
  def masked_to_unmasked_fn(n: int) -> int:
    if n < num_valid_indices:
      return valid_indices[n]
    else:
      if raise_error:
        raise ValueError(f"There is no index of unmasked array corresponding to index {n} in masked array.")
      else:
        return default_value

  @partial(np.vectorize, otypes=[int])
  def unmasked_to_masked_fn(n: int) -> int:
    if index := valid_indices_inverse_map.get(n):
      return index
    else:
      if raise_error:
        raise ValueError(f"There is no index of masked array corresponding to index {n} in unmasked array.")
      else:
        return default_value

  return unmasked_to_masked_fn, masked_to_unmasked_fn


# FIXME: add docstring
def mask_mesh_from_grid(ocean_mesh: TriangleMesh,
                        boundary_nodes: np.ndarray,
                        mask: xarray.DataArray,
                        query_radius: float | np.ndarray,
                        latitude_dim_name='lat',
                        longitude_dim_name='lon',
                        mode: Literal['all', 'any'] = 'all',
                        workers: int = 1):

  num_boundary_nodes = boundary_nodes.shape[0]
  num_vertices = ocean_mesh.vertices.shape[0]
  num_faces = ocean_mesh.faces.shape[0]
  logger.info(f"Read mesh with {num_vertices} vertices, "
              f"{num_boundary_nodes} boundary nodes, "
              f"and {num_faces} faces")
  if isinstance(query_radius, np.ndarray):
    mean_radius = query_radius.mean()
    logger.info("Looking for mesh vertices connected to the grid "
                f"with an average search radius of {(mean_radius / 1e3):.2f} km.")
  else:
    logger.info("Looking for mesh vertices connected to the grid "
                f"within a radius of {(query_radius / 1e3):.2f} km.")
  connected_mesh_vertices = get_connected_mesh_nodes(grid_lat=mask[latitude_dim_name].to_numpy(),
                                                     grid_lon=mask[longitude_dim_name].to_numpy(),
                                                     mesh_graph=ocean_mesh,
                                                     mask=mask,
                                                     query_radius=query_radius,
                                                     workers=workers)
  logger.info(f"Extracted {len(connected_mesh_vertices)} mesh vertices connected to the grid.")
  ocean_mesh_mskd, valid_vertices_map = mask_mesh(connected_mesh_vertices, ocean_mesh, mode=mode)
  boundary_nodes_mskd = np.vectorize(lambda n: valid_vertices_map.get(n, -1), otypes=[np.int32])(boundary_nodes)
  boundary_nodes_mskd = boundary_nodes_mskd[boundary_nodes_mskd >= 0]
  num_boundary_nodes_mskd = boundary_nodes_mskd.shape[0]
  num_vertices_mskd = ocean_mesh_mskd.vertices.shape[0]
  num_faces_mskd = ocean_mesh_mskd.faces.shape[0]
  logger.info(f"Masked mesh contains {num_vertices_mskd} vertices ({num_vertices_mskd / num_vertices:.2%}), "
              f"{num_boundary_nodes_mskd} boundary nodes ({num_boundary_nodes_mskd / num_boundary_nodes:.2%}), "
              f"and {num_faces_mskd} faces ({num_faces_mskd / num_faces:.2%}).")

  return ocean_mesh_mskd, boundary_nodes_mskd


def get_mesh_within_box(mesh: Mesh, box: Box):

  def _is_within_bounds(lat, lon):
    return (box.lat_min < lat < box.lat_max) and (box.lon_min < lon < box.lon_max)

  wgs_graph = mesh_to_wgs(mesh)
  vertices_within_bounds = np.array([_is_within_bounds(lat, lon) for (lat, lon) in zip(*wgs_graph.vertices)])
  marked_vertices = np.nonzero(vertices_within_bounds)[0]
  new_mesh, vertices_map = mask_mesh(marked_vertices, mesh, mode='all')
  return new_mesh, vertices_map


# TODO: add tests
def get_dummy_xscaling_graph(mesh_a: TriangleMesh,
                             mesh_b: TriangleMesh,
                             name_a: str = "mesh_a",
                             name_b: str = "mesh_b") -> TypedGraph:
  """Returns a typed graph containing whose sets of edges represent the provided triangle meshes and connectivity
  between the two.

  As main ideas are taken from https://arxiv.org/abs/2210.00612, the two meshes are termed "coarse" and "fine", however
  any two meshes could do as far as the implementation is concerned.

  Args:
    mesh_a: First TriangleMesh.
    mesh_b: Second TriangleMesh.

  Returns:
    A TypedGraph object representing a two-level hierarchical triangle mesh.
  """
  node_set_a = NodeSet(n_node=mesh_a.vertices.shape[0], features=mesh_a.vertices)
  node_set_b = NodeSet(n_node=mesh_b.vertices.shape[0], features=mesh_b.vertices)
  nodes = {name_a: node_set_a, name_b: node_set_b}

  def _get_mesh_edge_set(mesh):
    senders, receivers = faces_to_edges(mesh.faces)
    edge_set = EdgeSet(n_edge=senders.shape[0], indices=EdgesIndices(senders=senders, receivers=receivers), features=())
    return edge_set

  def _get_mesh_to_mesh_edge_set(senders_mesh: Mesh, receivers_mesh: Mesh):
    # The edge orientation goes from the nodes of the senders mesh to vertices of the receivers mesh contained into
    # triangles of the sender mesh
    senders, receivers = get_mesh_to_points_edges(senders_mesh=senders_mesh,
                                                  receivers_position=receivers_mesh.vertices)
    edge_set = EdgeSet(n_edge=senders.shape[0], indices=EdgesIndices(senders=senders, receivers=receivers), features=())
    return edge_set

  edges = {EdgeSetKey(name_a, (name_a, name_a)): _get_mesh_edge_set(mesh_b),
           EdgeSetKey(name_b, (name_b, name_b)): _get_mesh_edge_set(mesh_a),
           EdgeSetKey(name_a + "_to_" + name_b, (name_a, name_b)): _get_mesh_to_mesh_edge_set(mesh_a, mesh_b),
           EdgeSetKey(name_b + "_to_" + name_a, (name_b, name_a)): _get_mesh_to_mesh_edge_set(mesh_b, mesh_a)}

  graph = TypedGraph(context=Context(n_graph=np.array([1]), features=()), nodes=nodes, edges=edges)

  return graph
