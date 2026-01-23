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

from typing import Union, Iterable, Literal, Tuple, Dict, NamedTuple
from graphcast.typed_graph import Context, NodeSet, EdgeSet, EdgeSetKey, EdgesIndices, TypedGraph
from graphcast.mesh_graph import TriangleMesh, MeshGraph, faces_to_edges, mesh_to_wgs
from graphcast.constants import EARTH_RADIUS
import numpy as np
import scipy
import trimesh
import xarray


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
    radius: float,
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

  # FIXME: the following can probably have a more straightforward implementation, the crucial point is to be consistent
  #  with the order of lat/lon dimensions and indexing
  # [num_grid_points=num_lat_points * num_lon_points, 3]
  meshgrid_positions = _grid_lat_lon_to_coordinates(
      grid_latitude, grid_longitude).reshape([-1, 3])
  meshgrid_latitude_indices, meshgrid_longitude_indices = (
    np.unravel_index(np.arange(meshgrid_positions.shape[0], dtype=int),
                     (grid_latitude.shape[0], grid_longitude.shape[0])))

  if mask is not None:
    assert np.array_equal(mask['lat'].to_numpy(), grid_latitude)
    assert np.array_equal(mask['lon'].to_numpy(), grid_longitude)
    mask_data = mask.transpose('lat', 'lon').to_numpy()
  else:
    mask_data = np.ones((grid_latitude.shape[0], grid_longitude.shape[0]), dtype=bool)

  # [num_mesh_points, 3]
  mesh_positions = mesh.vertices
  kd_tree = scipy.spatial.cKDTree(mesh_positions)
  # [num_grid_points, num_mesh_points_per_grid_point]
  # Notice: the number of grid points per mesh point is not constant, so `query_ball_point` return an array of arrays, rather than a 2d array.
  # Notice: the ball is in 3D space, so the distances are not geodesic.
  query_indices = kd_tree.query_ball_point(x=meshgrid_positions, r=radius, workers=workers)
  mask_values = mask_data[meshgrid_latitude_indices, meshgrid_longitude_indices]
  valid_query_indices = query_indices[mask_values]
  grid_senders_indices = np.repeat(mask_values.nonzero()[0], np.array(list(map(len, valid_query_indices))))
  mesh_receivers_indices = np.concatenate(valid_query_indices, axis=0).astype(int)

  return grid_senders_indices, mesh_receivers_indices


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

  # [num_grid_points=num_lat_points * num_lon_points, 3]
  grid_positions = _grid_lat_lon_to_coordinates(
      grid_latitude, grid_longitude).reshape([-1, 3])

  mesh_trimesh = trimesh.Trimesh(vertices=mesh.vertices, faces=mesh.faces)

  # [num_grid_points] with mesh face indices for each grid point.
  # Notice: there is no guarantee that faces contain the grid point specified by `grid_positions`.
  _, _, query_face_indices = trimesh.proximity.closest_point(
      mesh_trimesh, grid_positions)

  # [num_grid_points, 3] with mesh node indices for each grid point.
  mesh_edge_indices = mesh.faces[query_face_indices]

  # [num_grid_points, 3] with grid node indices, where every row simply contains
  # the row (grid_point) index.
  grid_indices = np.arange(grid_positions.shape[0])
  grid_edge_indices = np.tile(grid_indices.reshape([-1, 1]), [1, 3])

  # Filter masked points.
  # [num_edges=num_grid_points, 3]
  if mask is not None:
    flat_mask = mask.transpose('lat', 'lon').to_numpy().reshape([-1])
    mesh_edge_indices = mesh_edge_indices[flat_mask, :]
    grid_edge_indices = grid_edge_indices[flat_mask, :]
  
  # Flatten to get a regular list.
  # [num_edges=num_grid_points*3]
  mesh_edge_indices = mesh_edge_indices.reshape([-1])
  grid_edge_indices = grid_edge_indices.reshape([-1])

  return mesh_edge_indices, grid_edge_indices


# TODO: add tests
def get_mesh_to_mesh_edges(
    *,
    senders_mesh: Mesh,
    receivers_mesh: Mesh) \
    -> tuple[np.ndarray, np.ndarray]:
  """Returns edges connecting each vertex of `senders_mesh` to the vertices of the triangle of `receivers_mesh` it's
  contained within.

  Args:
    senders_mesh: Mesh object.
    receivers_mesh: Mesh object.

  Returns:
    senders, receivers tuple of indices indicating edges between the two meshes.
    The number of edges is always num_lat_points * num_lon_points * 3
    * grid_indices: Indices of shape [num_edges], that index into a
      [num_lat_points, num_lon_points] grid, after flattening the leading axes.
    * mesh_indices: Indices of shape [num_edges], that index into mesh.vertices.
  """

  mesh_trimesh = trimesh.Trimesh(vertices=receivers_mesh.vertices, faces=receivers_mesh.faces)

  # [num_senders_mesh_vertices] with mesh face indices for each senders mesh vertex.
  _, _, query_face_indices = trimesh.proximity.closest_point(
    mesh_trimesh, senders_mesh.vertices)

  senders_indices = np.arange(senders_mesh.vertices.shape[0], dtype=int)
  # [3 * num_senders_mesh_vertices, 3] with mesh node indices for each grid point.
  senders = np.tile(senders_indices.reshape([-1, 1]), [1, 3]).reshape([-1])
  # [3 * num_senders_mesh_vertices, 3] with mesh node indices for each grid point.
  receivers = receivers_mesh.faces[query_face_indices].reshape([-1])

  return senders, receivers


# TODO: add tests
def get_connected_mesh_nodes(grid_lat: np.ndarray,
                             grid_lon: np.ndarray,
                             mesh_graph: Mesh,
                             grid_mask: xarray.DataArray,
                             query_radius: float,
                             workers: int = 1) -> set[int]:
  """Returns the set of mesh vertices connected to a valid grid point.

  It does so by excluding the mesh vertices that are not connected to a valid grid point by at least one edge of the
  Grid2Mesh and/or the Mesh2Grid-like graphs described in the GraphCast paper.

  Args:
    grid_lat: Latitude values for the grid [num_lat_points]
    grid_lon: Longitude values for the grid [num_lon_points]
    mesh_graph: MultiMeshGraph or TriangleMesh object.
    grid_mask: Boolean mask of shape [num_lat_points, num_lon_points]
    query_radius: Radius of connectivity in R3 for a sphere of unit radius.
  Returns:
    Set of indices of mesh vertices connected to a valid grid point.
  """

  (_, mesh_receivers) = radius_query_indices(
    grid_latitude=grid_lat,
    grid_longitude=grid_lon,
    mesh=mesh_graph,
    radius=query_radius,
    mask=grid_mask,
    workers=workers)

  (mesh_senders, _) = get_mesh_to_grid_edges(
    grid_latitude=grid_lat,
    grid_longitude=grid_lon,
    mesh=mesh_graph,
    mask=grid_mask)

  grid2mesh_connected_mesh_vertices = set(mesh_receivers)
  mesh2grid_connected_mesh_vertices = set(mesh_senders)
  connected_mesh_vertices = set.intersection(grid2mesh_connected_mesh_vertices, mesh2grid_connected_mesh_vertices)

  return connected_mesh_vertices


#TODO: add tests
def mask_mesh(marked_vertices: Iterable[int], mesh: Mesh, mode: Literal['any', 'all'] = 'any') -> Tuple[Mesh, Dict[int, int]]:
  """Filters the mesh to include only vertices belonging to triangles with at least one marked vertex (when `mode='any'`),
  or with all marked vertices (when `mode='all'`).

  Args:
    marked_vertices: Set of vertices connected to a valid grid point.
    mesh: MeshGraph object.
  Returns:
    Tuple containint a masked multimesh graph, and a mapping from the old vertex indices to the new vertex indices.
  """
  num_vertices, _ = mesh.vertices.shape
  num_faces, _ = mesh.faces.shape
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


def get_mesh_within_box(mesh: Mesh, box: Box):

  def _is_within_bounds(lat, lon):
    return (box.lat_min < lat < box.lat_max) and (box.lon_min < lon < box.lon_max)

  wgs_graph = mesh_to_wgs(mesh)
  vertices_within_bounds = np.array([_is_within_bounds(lat, lon) for (lat, lon) in zip(*wgs_graph.vertices)])
  marked_vertices = np.nonzero(vertices_within_bounds)[0]
  new_mesh, vertices_map = mask_mesh(marked_vertices, mesh, mode='all')
  return new_mesh, vertices_map


# TODO: add tests
def get_dummy_xscaling_graph(coarse_mesh: TriangleMesh, fine_mesh: TriangleMesh) -> TypedGraph:
  """Returns a typed graph containing whose sets of edges represent the provided triangle meshes and connectivity
  between the two.

  As main ideas are taken from https://arxiv.org/abs/2210.00612, the two meshes are termed "coarse" and "fine", however
  any two meshes could do as far as the implementation is concerned.

  Args:
    coarse_mesh: First TriangleMesh (coarse).
    fine_mesh: Second TriangleMesh (fine).

  Returns:
    A TypedGraph object representing a two-level hierarchical triangle mesh.
  """
  coarse_mesh_node_set = NodeSet(n_node=coarse_mesh.vertices.shape[0], features=coarse_mesh.vertices)
  fine_mesh_node_set = NodeSet(n_node=fine_mesh.vertices.shape[0], features=fine_mesh.vertices)
  nodes = {"coarse_mesh_nodes": coarse_mesh_node_set, "fine_mesh_nodes": fine_mesh_node_set}

  def _get_mesh_edge_set(mesh):
    senders, receivers = faces_to_edges(mesh.faces)
    edge_set = EdgeSet(n_edge=senders.shape[0], indices=EdgesIndices(senders=senders, receivers=receivers), features=())
    return edge_set

  def _get_mesh_to_mesh_edge_set(senders_mesh, receivers_mesh):
    # get_mesh_to_mesh_edges is taken from the grid to mesh connectivity implementation in GraphCast, hence
    # the edge orientation goes from the nodes contained in a triangle of the receivers mesh towards the vertices
    # of the latter
    senders, receivers = get_mesh_to_mesh_edges(senders_mesh=senders_mesh, receivers_mesh=receivers_mesh)
    edge_set = EdgeSet(n_edge=senders.shape[0], indices=EdgesIndices(senders=senders, receivers=receivers), features=())
    return edge_set

  edges = {EdgeSetKey("fine", ("fine_mesh_nodes", "fine_mesh_nodes")): _get_mesh_edge_set(fine_mesh),
           EdgeSetKey("coarse", ("coarse_mesh_nodes", "coarse_mesh_nodes")): _get_mesh_edge_set(coarse_mesh),
           EdgeSetKey("fine2coarse", ("coarse_mesh_nodes", "fine_mesh_nodes")): _get_mesh_to_mesh_edge_set(coarse_mesh,
                                                                                                           fine_mesh),
           EdgeSetKey("coarse2fine", ("fine_mesh_nodes", "coarse_mesh_nodes")): _get_mesh_to_mesh_edge_set(fine_mesh,
                                                                                                           coarse_mesh)}

  graph = TypedGraph(context=Context(n_graph=np.array([1]), features=()), nodes=nodes, edges=edges)

  return graph
