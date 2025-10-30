# Copyright 2025 OGS.
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
# TODO: move tests from icosahedral_mesh and add tests for new functions
# TODO: switch from np.ndarray to chex.Array in type hints
"""Utils for working with (multi-)mesh graphs and geospatial graphs."""
import functools
import itertools
from typing import Literal
from typing import Sequence, Tuple

import chex
import networkx as nx
import numpy as np
import xarray as xr
from osgeo import osr
from pyproj import Transformer

from graphcast import typed_graph

osr.UseExceptions()

stereographic_proj = osr.SpatialReference("+proj=stere +ellps=WGS84 +lat_0=90")
cartesian_proj = osr.SpatialReference("+proj=cart +ellps=WGS84 +units=m +x_0=0 +y_0=0")
cartesian_unit_sphere_proj = osr.SpatialReference("+proj=cart +a=1 +b=1 +units=m +x_0=0 +y_0=0")
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


@chex.dataclass(frozen=True, eq=True)
class Graph:
  """Data structure for a generic graph

  Attributes:
    vertices: spatial positions of the vertices of the graph of shape [num_vertices, num_dims].
    edges: tuple of senders, receivers nodes
  """
  vertices: np.ndarray
  edges: tuple[np.ndarray, np.ndarray]


@chex.dataclass(frozen=True, eq=True)
class WGSGraph:
  """Data structure for a World Geodetic System (WGS) graph,
   where vertices are stored as a (latitudes, longitudes) tuple.

  Attributes:
    vertices: tuple of (latitudes, longitudes) of the nodes
    edges: tuple of senders, receivers nodes
  """
  vertices: tuple[np.ndarray, np.ndarray]
  edges: tuple[np.ndarray, np.ndarray]


@chex.dataclass(frozen=True, eq=True)
class TriangleMesh:
  """Data structure for triangular meshes in 3D.

  Attributes:
    vertices: spatial positions of the vertices of the mesh of shape
        [num_vertices, num_dims].
    faces: triangular faces of the mesh of shape [num_faces, 3]. Contains
        integer indices into `vertices`.

  """
  vertices: np.ndarray
  faces: np.ndarray


@chex.dataclass(frozen=True, eq=True)
class MeshGraph:
  """Data structure for multi-mesh graphs in 3D.

  Attributes:
    vertices: same as TriangleMesh.vertices.
    faces: same as TriangleMesh.faces.
    edges: cumulated edges of all the triangular meshes used in building the multi-mesh graph.

  """
  vertices: np.ndarray
  faces: np.ndarray
  edges: tuple[np.ndarray, np.ndarray]


def merge_meshes(
    mesh_list: Sequence[TriangleMesh]) -> MeshGraph:
  """Merges all meshes into one. Assumes the last mesh is the finest.

  Args:
     mesh_list: Sequence of meshes, from coarse to fine refinement levels. The
       vertices and faces may contain those from preceding, coarser levels.

  Returns:
     `MultiMeshGraph` for which the vertices and faces correspond to the highest
     resolution mesh in the hierarchy, and the edges are computed from the join set of the
     faces at all levels of the hierarchy.
  """
  for mesh_i, mesh_ip1 in itertools.pairwise(mesh_list):
    num_nodes_mesh_i = mesh_i.vertices.shape[0]
    assert np.allclose(mesh_i.vertices, mesh_ip1.vertices[:num_nodes_mesh_i])

  all_faces = np.concatenate([mesh.faces for mesh in mesh_list], axis=0)
  all_edges = faces_to_edges(all_faces)
  finest_mesh_vertices = mesh_list[-1].vertices
  finest_mesh_faces = mesh_list[-1].faces

  return MeshGraph(
    vertices=finest_mesh_vertices,
    edges=all_edges,
    faces=finest_mesh_faces)


def faces_to_edges(faces: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
  """Transforms polygonal faces to sender and receiver indices.

  It does so by transforming every face into N_i edges. Such if the triangular
  face has indices [0, 1, 2], three edges are added 0->1, 1->2, and 2->0.

  If all faces have consistent orientation, and the surface represented by the
  faces is closed, then every edge in a polygon with a certain orientation
  is also part of another polygon with the opposite orientation. In this
  situation, the edges returned by the method are always bidirectional.

  Args:
     faces: Integer array of shape [num_faces, 3]. Contains node indices
        adjacent to each face.
  Returns:
     Tuple with sender/receiver indices, each of shape [num_edges=num_faces*3].

  """
  assert faces.ndim == 2
  assert faces.shape[-1] == 3
  senders = np.concatenate([faces[:, 0], faces[:, 1], faces[:, 2]])
  receivers = np.concatenate([faces[:, 1], faces[:, 2], faces[:, 0]])
  return senders, receivers


def _get_undirected_edges(edges: tuple[np.ndarray, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
  """Transforms directed edges to undirected edges by using standard ordering of vertices.

  Args:
     edges: Tuple with sender/receiver indices, each with shape [num_edges]
  Returns:
     Tuple with sender/receiver indices of vertices adjacent to undirected edges.

  """
  senders, receivers = edges
  undirected_edges = set((s, r) if s <= r else (r, s) for (s, r) in zip(senders, receivers))
  senders = np.array([edge[0] for edge in undirected_edges])
  receivers = np.array([edge[1] for edge in undirected_edges])
  return senders, receivers


# TODO: add and tests
def graph_to_wgs(graph: Graph, unit_sphere: bool = False) -> WGSGraph:
  """Gets the graph (WGS coordinates of vertices and (undirected) edges) from a 3D graph."""

  transformer = get_transform(cartesian_unit_sphere_proj if unit_sphere else cartesian_proj,
                              platecarree_proj, pack_back=False)
  longitudes, latitudes, _ = transformer(graph.vertices)
  longitudes = np.where(longitudes < 0, longitudes + 360, longitudes)
  # We use the convention used by graphcast coordinates are (lat, lon), in this order.
  vertices = (latitudes, longitudes)
  return WGSGraph(vertices=vertices, edges=graph.edges)


# TODO: add tests
def mesh_to_wgs(mesh: TriangleMesh | MeshGraph, unit_sphere: bool = False) -> WGSGraph:
  """Gets the graph (WGS coordinates of vertices and (undirected) edges) from a 3D mesh.

  Args:
     mesh: TriangleMesh or MultiMeshGraph representing a mesh.
        When mesh is a multi-mesh, it considers only edges from faces of the finest mesh.
     unit_sphere: whether the mesh is defined on a unit sphere.
  Returns:
     Tuple with vertices and undirected edges between them.
  """

  edges = _get_undirected_edges(faces_to_edges(mesh.faces))
  graph = Graph(vertices=mesh.vertices, edges=edges)
  wgs_graph = graph_to_wgs(graph, unit_sphere=unit_sphere)

  return wgs_graph


# TODO: add tests
def typed_to_wgs(graph: typed_graph.TypedGraph, edge_set_name: str) -> WGSGraph:
  """Gets the graph (WGS coordinates of vertices and (undirected) edges)
  from a particular edge-set of a typed graph in 3D.

  Args:
      graph: TypedGraph representing a multi-mesh.
      edge_set_name: name of the edge set to be used.
  Returns:
      Tuple with vertices and undirected edges between them.
  """
  # FIXME: bugged implementation, consider for example the `fine` or `coarse` graphs. The vertices are counted twice!
  edge_set_key = graph.edge_key_by_name(edge_set_name)
  senders_nodes_name, receivers_nodes_name = edge_set_key.node_sets
  senders_nodes = graph.nodes[senders_nodes_name]
  receivers_nodes = graph.nodes[receivers_nodes_name]
  vertices = np.concatenate((senders_nodes.features, receivers_nodes.features))
  edge_set = graph.edges[edge_set_key]
  senders, receivers = edge_set.indices
  receivers = senders_nodes.n_node + receivers
  edges = (senders, receivers)
  graph = Graph(vertices=vertices, edges=edges)
  wgs_graph = graph_to_wgs(graph)

  return wgs_graph


def wgs_to_nx(wgs_graph: WGSGraph) -> nx.DiGraph:
  graph = nx.DiGraph()
  latitudes, longitudes = wgs_graph.vertices
  vertices = [(node, {'latitude': latitude, 'longitude': longitude}) for node, (latitude, longitude) in enumerate(zip(longitudes, latitudes))]
  graph.add_nodes_from(vertices)
  senders, receivers = wgs_graph.edges
  edges = [(sender, receiver) for sender, receiver in zip(senders, receivers)]
  graph.add_edges_from(edges)

  return graph


def graph_summary(graph: nx.Graph) -> str:
  summary =  f"Nodes: {graph.number_of_nodes()}, " \
             f"Edges: {graph.number_of_edges()}, " \
             f"Average degree: {sum(d for _, d in graph.degree()) / graph.number_of_nodes()}"

  return summary

