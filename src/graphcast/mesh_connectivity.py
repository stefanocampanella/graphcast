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

import numpy as np
import numpy.typing as npt
import scipy
import trimesh

from graphcast.gis_utils import cartesian_crs, equirectangular_crs, get_transform
from graphcast.mesh_graph import Mesh, TriangleMesh, faces_to_edges
from graphcast.typed_graph import Context, EdgeSet, EdgeSetKey, EdgesIndices, NodeSet, TypedGraph

logger = logging.getLogger(__name__)


# TODO: update tests and usage in notebooks (e.g. mesh_comparison.ipynb) to take into account that radius of the Earth
#  is now used (no longer unit sphere)
# FIXME: implement this using pyproj
def _grid_lat_lon_to_coordinates(
  grid_latitude: np.ndarray, grid_longitude: np.ndarray
) -> np.ndarray:
  """Lat [num_lat] lon [num_lon] to 3d coordinates [num_lat, num_lon, 3]."""
  num_lat = len(grid_latitude)
  num_lon = len(grid_longitude)
  lon, lat = np.meshgrid(grid_longitude, grid_latitude, indexing="ij")
  lon = np.where(lon > 180, lon - 360, lon)
  lon = lon.reshape(-1)
  lat = lat.reshape(-1)
  lonlat_coordinates = np.stack([lon, lat], axis=-1)
  transform = get_transform(equirectangular_crs, cartesian_crs)
  cartesian_coordinates = transform(lonlat_coordinates)
  cartesian_coordinates = cartesian_coordinates.reshape(num_lon, num_lat, 3)
  cartesian_coordinates = cartesian_coordinates.transpose(1, 0, 2)
  return cartesian_coordinates


def radius_query_indices(
  *,
  grid_latitude: np.ndarray,
  grid_longitude: np.ndarray,
  mesh: Mesh,
  radius: float | np.ndarray,
  mask: np.ndarray | None = None,
  workers: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
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
    mask = np.ones((grid_latitude.shape[0], grid_longitude.shape[0]), dtype=bool)
  # [num_grid_points=num_lat_points * num_lon_points]
  mask = mask.reshape([-1])
  # [num_grid_points=num_lat_points * num_lon_points, 3]
  grid_positions = _grid_lat_lon_to_coordinates(grid_latitude, grid_longitude).reshape([-1, 3])
  # [num_mesh_points, 3]
  mesh_positions = mesh.vertices
  # [num_valid_grid_points=sum(mask)]
  valid_grid_positions = grid_positions[mask]

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
  query_indices: npt.NDArray[list[int]] = kd_tree.query_ball_point(
    x=mesh_positions, r=radius, workers=workers
  )
  _, masked_to_unmasked_fn = get_masking_indices_fns(mask)
  # noinspection PyTypeChecker
  grid_senders = np.concatenate(list(map(masked_to_unmasked_fn, query_indices)), axis=0).astype(int)
  mesh_receivers = np.repeat(
    np.arange(mesh_positions.shape[0], dtype=int),
    np.fromiter(map(len, query_indices), dtype=int),
  )
  return grid_senders, mesh_receivers


def get_mesh_to_grid_edges(
  *, grid_latitude: np.ndarray, grid_longitude: np.ndarray, mesh: Mesh, mask: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
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
  # [num_grid_points=num_lat_points * num_lon_points]
  grid_mask = mask.reshape([-1])
  # [num_grid_points=num_lat_points * num_lon_points, 3]
  grid_positions = _grid_lat_lon_to_coordinates(grid_latitude, grid_longitude).reshape([-1, 3])
  # [num_valid_grid_points=sum(grid_mask)]
  valid_grid_positions = grid_positions[grid_mask, :]
  mesh_senders, valid_grid_receivers = get_mesh_to_points_edges(
    senders_mesh=mesh, receivers_position=valid_grid_positions
  )
  _, masked_to_unmasked_fn = get_masking_indices_fns(grid_mask)
  # noinspection PyTypeChecker
  grid_receivers: npt.NDArray[int] = masked_to_unmasked_fn(valid_grid_receivers)

  return mesh_senders, grid_receivers


# TODO: add tests
# TODO: some notebooks might have used `get_mesh_to_mesh_edges` instead of `get_mesh_to_points_edges`,
#  both the signature and direction of edges have to be refactore.
def get_mesh_to_points_edges(
  *, senders_mesh: Mesh, receivers_position: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
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
  receivers = np.tile(
    np.arange(len(receivers_position), dtype=int).reshape([-1, 1]), [1, 3]
  ).reshape([-1])

  return senders, receivers


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
  valid_indices_inverse_map = {
    v: i for (i, v) in zip(masked_array_indices, valid_indices, strict=False)
  }

  @partial(np.vectorize, otypes=[int])
  def masked_to_unmasked_fn(n: int) -> int:
    if n < num_valid_indices:
      return valid_indices[n]
    else:
      if raise_error:
        raise ValueError(
          f"There is no index of unmasked array corresponding to index {n} in masked array."
        )
      else:
        return default_value

  @partial(np.vectorize, otypes=[int])
  def unmasked_to_masked_fn(n: int) -> int:
    if index := valid_indices_inverse_map.get(n):
      return index
    else:
      if raise_error:
        raise ValueError(
          f"There is no index of masked array corresponding to index {n} in unmasked array."
        )
      else:
        return default_value

  return unmasked_to_masked_fn, masked_to_unmasked_fn


# TODO: add tests
def get_dummy_xscaling_graph(
  mesh_a: TriangleMesh, mesh_b: TriangleMesh, name_a: str = "mesh_a", name_b: str = "mesh_b"
) -> TypedGraph:
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
    edge_set = EdgeSet(
      n_edge=senders.shape[0],
      indices=EdgesIndices(senders=senders, receivers=receivers),
      features=(),
    )
    return edge_set

  def _get_mesh_to_mesh_edge_set(senders_mesh: Mesh, receivers_mesh: Mesh):
    # The edge orientation goes from the nodes of the senders mesh to vertices of the receivers mesh contained into
    # triangles of the sender mesh
    senders, receivers = get_mesh_to_points_edges(
      senders_mesh=senders_mesh, receivers_position=receivers_mesh.vertices
    )
    edge_set = EdgeSet(
      n_edge=senders.shape[0],
      indices=EdgesIndices(senders=senders, receivers=receivers),
      features=(),
    )
    return edge_set

  edges = {
    EdgeSetKey(name_a, (name_a, name_a)): _get_mesh_edge_set(mesh_b),
    EdgeSetKey(name_b, (name_b, name_b)): _get_mesh_edge_set(mesh_a),
    EdgeSetKey(name_a + "_to_" + name_b, (name_a, name_b)): _get_mesh_to_mesh_edge_set(
      mesh_a, mesh_b
    ),
    EdgeSetKey(name_b + "_to_" + name_a, (name_b, name_a)): _get_mesh_to_mesh_edge_set(
      mesh_b, mesh_a
    ),
  }

  graph = TypedGraph(context=Context(n_graph=np.array([1]), features=()), nodes=nodes, edges=edges)

  return graph
