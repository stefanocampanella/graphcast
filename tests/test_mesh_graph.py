# SPDX-FileCopyrightText: 2023 DeepMind Technologies Limited
# SPDX-FileCopyrightText: 2026 Stefano Campanella
# SPDX-License-Identifier: Apache-2.0
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
#
# The merging tests come from the absltest tests of the original GraphCast code (icosahedral_mesh), ported to
# pytest and updated for this fork.
"""Tests for mobydick.mesh_graph (the merging tests come from the legacy icosahedral_mesh tests)."""

import numpy as np
import pytest

from mobydick import icosahedral_mesh, mesh_graph, typed_graph
from mobydick.gis_utils import _cartesian_wkt


@pytest.mark.parametrize("splits", list(range(4)))
def test_merge_meshes(splits):
  """The merged multi-mesh keeps the finest level's vertices and faces, with the edges of the faces of all levels
  (upstream GraphCast kept the faces of all levels)."""
  mesh_hierarchy = icosahedral_mesh.get_hierarchy_of_triangular_meshes_for_sphere(splits=splits)
  mesh = mesh_graph.merge_meshes(mesh_hierarchy)

  # Vertices and faces of the finest mesh, edges of the faces of all levels.
  all_faces = np.concatenate([m.faces for m in mesh_hierarchy], axis=0)
  expected_senders, expected_receivers = mesh_graph.faces_to_edges(all_faces)
  assert isinstance(mesh, mesh_graph.MeshGraph)
  np.testing.assert_array_equal(mesh.vertices, mesh_hierarchy[-1].vertices)
  np.testing.assert_array_equal(mesh.faces, mesh_hierarchy[-1].faces)
  np.testing.assert_array_equal(mesh.edges[0], expected_senders)
  np.testing.assert_array_equal(mesh.edges[1], expected_receivers)
  assert mesh.spatial_reference_system == mesh_hierarchy[-1].spatial_reference_system


def test_merge_meshes_rejects_non_nested_vertices():
  """Levels must be nested: a finer level whose leading vertices are not the coarser level's is rejected."""
  coarse, fine = icosahedral_mesh.get_hierarchy_of_triangular_meshes_for_sphere(splits=1)
  shuffled = mesh_graph.TriangleMesh(vertices=fine.vertices[::-1], faces=fine.faces)
  with pytest.raises(AssertionError):
    mesh_graph.merge_meshes([coarse, shuffled])


def test_merge_meshes_rejects_mixed_spatial_reference_systems():
  """All levels must be in the same coordinates: a level in another spatial reference system is rejected."""
  coarse, fine = icosahedral_mesh.get_hierarchy_of_triangular_meshes_for_sphere(splits=1)
  relabelled = mesh_graph.TriangleMesh(
    vertices=fine.vertices, faces=fine.faces, spatial_reference_system=_cartesian_wkt
  )
  with pytest.raises(AssertionError):
    mesh_graph.merge_meshes([coarse, relabelled])


def test_graph_defaults_to_earth_centred_coordinates():
  """Like meshes, a graph's vertices default to WGS84 Earth-centred coordinates (the WKT, not a registry name)."""
  graph = mesh_graph.Graph(vertices=np.zeros((2, 3)), edges=(np.array([0]), np.array([1])))
  assert graph.spatial_reference_system == _cartesian_wkt


def test_faces_to_edges():
  """Each face gives its three directed edges, in the documented order: first edges of all faces, then second, …;
  the reverses of edges that belong to one face only (here all of them: two separate triangles) follow."""
  faces = np.array([[0, 1, 2], [3, 4, 5]])

  # This also documents the order of the edges returned by the method.
  face_edges = np.array([[0, 1], [3, 4], [1, 2], [4, 5], [2, 0], [5, 3]])
  expected_edges = np.concatenate([face_edges, face_edges[:, ::-1]])
  expected_senders = expected_edges[:, 0]
  expected_receivers = expected_edges[:, 1]

  senders, receivers = mesh_graph.faces_to_edges(faces)

  np.testing.assert_array_equal(senders, expected_senders)
  np.testing.assert_array_equal(receivers, expected_receivers)


def test_faces_to_edges_of_open_and_closed_meshes_go_both_ways():
  """On an open mesh (two triangles sharing an edge, like a coastline's last faces) the boundary edges get their
  reverse, so every edge goes both ways: messages can run along the boundary in both directions (they could not
  before Nov 2026). A closed mesh keeps exactly its 3 * num_faces face edges, as in GraphCast."""
  senders, receivers = mesh_graph.faces_to_edges(np.array([[0, 1, 2], [0, 2, 3]]))
  edges = set(zip(senders.tolist(), receivers.tolist(), strict=True))
  assert len(edges) == len(senders) == 10  # 5 undirected edges, the shared one 0-2 once per direction
  assert all((b, a) in edges for a, b in edges)
  octahedron = _octahedron()
  senders, receivers = mesh_graph.faces_to_edges(octahedron.faces)
  assert len(senders) == 3 * len(octahedron.faces)
  edges = set(zip(senders.tolist(), receivers.tolist(), strict=True))
  assert len(edges) == len(senders) and all((b, a) in edges for a, b in edges)


def test_longest_incident_edges():
  """Each vertex gets the length of its longest triangle edge (straight line): here a 3-4-5 right triangle; a vertex in
  no triangle gets 0 (edges outside the triangles, like a multi-mesh's coarser edges, do not count)."""
  vertices = np.array([[0.0, 0.0, 0.0], [3.0, 0.0, 0.0], [0.0, 4.0, 0.0], [10.0, 10.0, 10.0]])
  longest = mesh_graph.longest_incident_edges(vertices, np.array([[0, 1, 2]]))
  np.testing.assert_allclose(longest, [4.0, 5.0, 5.0, 0.0])


def test_faces_to_edges_keeps_each_edge_once_with_a_flipped_face():
  """Two faces running their shared edge in the same direction (a flipped face) give it once and its reverse once,
  not twice each, which would make messages across it count double; edge order is that of first appearance."""
  senders, receivers = mesh_graph.faces_to_edges(np.array([[0, 1, 2], [0, 1, 3]]))
  edges = list(zip(senders.tolist(), receivers.tolist(), strict=True))
  assert len(edges) == len(set(edges)) == 10
  assert all((b, a) in edges for a, b in edges)
  assert edges[:2] == [(0, 1), (1, 2)]


def _octahedron(radius: float = 1.0) -> mesh_graph.TriangleMesh:
  """Vertices at the poles and on the equator at longitudes 0, 90, 180, 270."""
  vertices = radius * np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0], [-1, 0, 0], [0, -1, 0], [0, 0, -1]], dtype=float)
  faces = np.array([[0, 1, 2], [0, 2, 3], [0, 3, 4], [0, 4, 1], [5, 2, 1], [5, 3, 2], [5, 4, 3], [5, 1, 4]])
  return mesh_graph.TriangleMesh(vertices=vertices, faces=faces)


EXPECTED_LATITUDES = [90.0, 0.0, 0.0, 0.0, 0.0, -90.0]
EXPECTED_LONGITUDES = [0.0, 0.0, 90.0, 180.0, 270.0]


def test_mesh_to_latlon():
  """Vertices on the unit sphere become (latitude, longitude) in degrees, longitudes in [0, 360); each edge of the
  faces appears once, undirected."""
  graph = mesh_graph.mesh_to_latlon(_octahedron(), unit_sphere=True)

  latitudes, longitudes = graph.vertices
  np.testing.assert_allclose(latitudes, EXPECTED_LATITUDES, atol=1e-9)
  np.testing.assert_allclose(longitudes[1:5], EXPECTED_LONGITUDES[1:], atol=1e-9)
  edges = set(zip(*graph.edges, strict=True))
  assert len(edges) == 12
  assert all(s < r for s, r in edges)


def test_typed_to_latlon():
  """An edge set between two node sets is drawn on their concatenated vertices (receivers after senders); within one
  node set, on its vertices."""
  # On the WGS84 ellipsoid: the equator at the semi-major axis, the poles at the semi-minor one.
  semi_major = 6_378_137.0
  semi_minor = semi_major * (1 - 1 / 298.257223563)
  octahedron = _octahedron()
  poles, equator = octahedron.vertices[[0, 5]] * semi_minor, octahedron.vertices[1:5] * semi_major

  def nodes(features):
    return typed_graph.NodeSet(n_node=np.array([len(features)]), features=features)

  def edges(senders, receivers):
    indices = typed_graph.EdgesIndices(senders=np.array(senders), receivers=np.array(receivers))
    return typed_graph.EdgeSet(n_edge=np.array([len(senders)]), indices=indices, features=np.zeros((len(senders), 0)))

  between = typed_graph.EdgeSetKey("pole2equator", ("poles", "equator"))
  within = typed_graph.EdgeSetKey("ring", ("equator", "equator"))
  graph = typed_graph.TypedGraph(
    context=typed_graph.Context(n_graph=np.array([1]), features=()),
    nodes={"poles": nodes(poles), "equator": nodes(equator)},
    edges={between: edges([0, 1], [0, 2]), within: edges([0, 1, 2, 3], [1, 2, 3, 0])},
  )

  pole2equator = mesh_graph.typed_to_latlon(graph, "pole2equator")
  latitudes, _ = pole2equator.vertices
  np.testing.assert_allclose(latitudes, [90.0, -90.0, 0.0, 0.0, 0.0, 0.0], atol=1e-9)
  np.testing.assert_array_equal(np.stack(pole2equator.edges), [[0, 1], [2, 4]])
  ring = mesh_graph.typed_to_latlon(graph, "ring")
  np.testing.assert_allclose(ring.vertices[1], EXPECTED_LONGITUDES[1:], atol=1e-9)
  np.testing.assert_array_equal(np.stack(ring.edges), [[0, 1, 2, 3], [1, 2, 3, 0]])


def test_latlon_to_nx():
  """The networkx graph has a node per vertex with its latitude and longitude, and the directed edges; the summary
  counts them."""
  wgs_graph = mesh_graph.EquirectangularGraph(
    vertices=(np.array([10.0, 20.0, 30.0]), np.array([100.0, 110.0, 120.0])), edges=(np.array([0, 1]), np.array([1, 2]))
  )
  graph = mesh_graph.latlon_to_nx(wgs_graph)

  assert graph.nodes[1] == {"latitude": 20.0, "longitude": 110.0}
  assert list(graph.edges) == [(0, 1), (1, 2)]
  assert mesh_graph.graph_summary(graph) == "Nodes: 3, Edges: 2, Average degree: 1.3333333333333333"
