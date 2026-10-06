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
# NOTICE:
# The routines here are intended to prune processor (mesh-)graphs and will stay as legacy, as they are of limited use.
# While at the moment of writing VRAM is still precious, when using land-masking in the ocean version of GraphCast
# there is little practical interest in pruning.
# The reason is that typically such graphs are well-connected enough to leave them unpruned after a few message-passing
# steps. Here is a more formal definition of the problem:
#
#   Consider a typed graph G = (V_grid, V_mesh, E_enc, E_dec, E_proc) and also all paths going from V_grid to
#   V_mesh via an edge of E_enc, then doing L hops using edges from E_proc, and finally returning to V_grid via an edge
#   of E_dec. Remove from V_mesh all nodes that are not visited by any such path.
#
# After an insightful discussion with Davide Nuzzi (who is always a source of inspiration) and patient supervision of
# Claudio B. Caporusso, we came up with the following algorithm that breaks the paths in two:
#   1. Build the boolean adjacency matrix E of the encoder, P of processor, and D of decoder edges.
#   2. For each n in 1, ..., L compute mask_n as follows:
#       a. Compute the adjacency matrix E * P^n between V_grid and V_mesh of paths doing exactly one hop from V_grid to
#          V_mesh and n hops on V_mesh.
#       b. Compute the V_mesh mask as the `any` operator mapped onto the rows of this matrix, which equals to
#          ones(V_grid) * E * P^n, where ones(V_grid) is the vector (1, ..., 1)
#          with length equal to the number of grid points.
#       c. Compute the sum (i.e. the boolean OR) of the adjacency matrices P^(n-k) * D for k = 0, ..., L-n, taking into
#          account the trajectories coming back to V_grid in at most L-n hops on V_mesh and one hop to V_grid.
#       d. Compute the V_mesh mask similarly as in step b as (sum_{k = 0, ..., L-n} P^(n-k) * D) * ones(V_grid)
#       e. Compute mask_n as the Hadamard product of the two masks computed in steps b and d.
#   3. Compute the OR (sum) of all mask_n, which is the final V_mesh mask.
# The algorithm can be efficiently implemented using array operations, even in JAX. The matrices could be large, but
# one could use sparse matrices and keep only their matrix-vector products all along the algorithm.
# The algorithm is rather fancy, but for reasonable graphs and number of message-passing steps pruning is useless.
# However, this could be a suitable technique for training large GNN by masking the inputs and using smaller subgraphs,
# an idea worth exploring.
# The current implementation of pruning routines should return results roughly similar to the one above with 0
# message-passing steps on the mesh graph.
"""Tools for pruning meshes."""

import logging
from collections.abc import Iterable
from typing import Literal, NamedTuple

import numpy as np
import xarray as xr

from graphcast.mesh_connectivity import get_mesh_to_grid_edges, radius_query_indices
from graphcast.mesh_graph import Mesh, MeshGraph, TriangleMesh, faces_to_edges, mesh_to_latlon

logger = logging.getLogger(__name__)


class Box(NamedTuple):
  lat_min: float
  lat_max: float
  lon_min: float
  lon_max: float


# TODO: add tests
def get_connected_mesh_nodes(
  grid_lat: np.ndarray,
  grid_lon: np.ndarray,
  mesh_graph: Mesh,
  mask: np.ndarray,
  query_radius: float | np.ndarray,
  workers: int = 1,
) -> set[int]:
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
    assert mesh_graph.vertices.shape[0] == query_radius.shape[0], (
      "The number of vertices in the mesh graph must match the number of query radii."
    )

  (_, mesh_receivers) = radius_query_indices(
    grid_latitude=grid_lat,
    grid_longitude=grid_lon,
    mesh=mesh_graph,
    radius=query_radius,
    mask=mask,
    workers=workers,
  )

  (mesh_senders, _) = get_mesh_to_grid_edges(
    grid_latitude=grid_lat, grid_longitude=grid_lon, mesh=mesh_graph, mask=mask
  )

  grid2mesh_connected_mesh_vertices = set(mesh_receivers)
  mesh2grid_connected_mesh_vertices = set(mesh_senders)
  connected_mesh_vertices = set.intersection(
    grid2mesh_connected_mesh_vertices, mesh2grid_connected_mesh_vertices
  )

  return connected_mesh_vertices


# TODO: add tests
# TODO: reimplement the following using get_masking_indices_fns
def prune_mesh(
  marked_vertices: Iterable[int], mesh: Mesh, mode: Literal["any", "all"] = "any"
) -> tuple[Mesh, dict[int, int]]:
  """Filters the mesh to include only vertices belonging to triangles with at least one marked vertex (when `mode='any'`),
  or with all marked vertices (when `mode='all'`).

  Args:
    marked_vertices: Set of vertices connected to a valid grid point.
    mesh: MeshGraph object.
  Returns:
    Tuple containint a masked multimesh graph, and a mapping from the old vertex indices to the new vertex indices.
  """
  predicate = np.any if mode == "any" else np.all
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


# FIXME: add docstring
def prune_mesh_from_mask(
  mesh: TriangleMesh,
  boundary_nodes: np.ndarray,
  mask: xr.DataArray,
  query_radius: float | np.ndarray,
  latitude_dim_name="lat",
  longitude_dim_name="lon",
  mode: Literal["all", "any"] = "all",
  workers: int = 1,
):
  num_boundary_nodes = boundary_nodes.shape[0]
  num_vertices = mesh.vertices.shape[0]
  num_faces = mesh.faces.shape[0]
  logger.info(
    f"Read mesh with {num_vertices} vertices, "
    f"{num_boundary_nodes} boundary nodes, "
    f"and {num_faces} faces"
  )
  if isinstance(query_radius, np.ndarray):
    # TODO: add min and max radius to logger
    mean_radius = query_radius.mean()
    logger.info(
      "Looking for mesh vertices connected to the grid "
      f"with an average search radius of {(mean_radius / 1e3):.2f} km."
    )
  else:
    logger.info(
      "Looking for mesh vertices connected to the grid "
      f"within a radius of {(query_radius / 1e3):.2f} km."
    )
  connected_mesh_vertices = get_connected_mesh_nodes(
    grid_lat=mask[latitude_dim_name].to_numpy(),
    grid_lon=mask[longitude_dim_name].to_numpy(),
    mesh_graph=mesh,
    mask=mask.to_numpy(),
    query_radius=query_radius,
    workers=workers,
  )
  logger.info(f"Extracted {len(connected_mesh_vertices)} mesh vertices connected to the grid.")
  mesh_mskd, valid_vertices_map = prune_mesh(connected_mesh_vertices, mesh, mode=mode)
  boundary_nodes_mskd = np.vectorize(lambda n: valid_vertices_map.get(n, -1), otypes=[np.int32])(
    boundary_nodes
  )
  boundary_nodes_mskd = boundary_nodes_mskd[boundary_nodes_mskd >= 0]
  num_boundary_nodes_mskd = boundary_nodes_mskd.shape[0]
  num_vertices_mskd = mesh_mskd.vertices.shape[0]
  num_faces_mskd = mesh_mskd.faces.shape[0]
  logger.info(
    f"Masked mesh contains {num_vertices_mskd} vertices ({num_vertices_mskd / num_vertices:.2%}), "
    f"{num_boundary_nodes_mskd} boundary nodes ({num_boundary_nodes_mskd / num_boundary_nodes:.2%}), "
    f"and {num_faces_mskd} faces ({num_faces_mskd / num_faces:.2%})."
  )

  return mesh_mskd, boundary_nodes_mskd


def get_mesh_within_box(mesh: Mesh, box: Box):
  def _is_within_bounds(lat, lon):
    return (box.lat_min < lat < box.lat_max) and (box.lon_min < lon < box.lon_max)

  wgs_graph = mesh_to_latlon(mesh)
  vertices_within_bounds = np.array(
    [_is_within_bounds(lat, lon) for (lat, lon) in zip(*wgs_graph.vertices, strict=False)]
  )
  marked_vertices = np.nonzero(vertices_within_bounds)[0]
  new_mesh, vertices_map = prune_mesh(marked_vertices, mesh, mode="all")
  return new_mesh, vertices_map
