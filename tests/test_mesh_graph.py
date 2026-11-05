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
"""Tests for graphcast.mesh_graph (the merging tests come from the legacy icosahedral_mesh tests)."""

import numpy as np
import pytest

from graphcast import icosahedral_mesh, mesh_graph


@pytest.mark.parametrize("splits", list(range(4)))
def test_merge_meshes(splits):
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


def test_merge_meshes_rejects_non_nested_vertices():
  coarse, fine = icosahedral_mesh.get_hierarchy_of_triangular_meshes_for_sphere(splits=1)
  shuffled = mesh_graph.TriangleMesh(vertices=fine.vertices[::-1], faces=fine.faces)
  with pytest.raises(AssertionError):
    mesh_graph.merge_meshes([coarse, shuffled])


def test_faces_to_edges():
  faces = np.array([[0, 1, 2], [3, 4, 5]])

  # This also documents the order of the edges returned by the method.
  expected_edges = np.array([[0, 1], [3, 4], [1, 2], [4, 5], [2, 0], [5, 3]])
  expected_senders = expected_edges[:, 0]
  expected_receivers = expected_edges[:, 1]

  senders, receivers = mesh_graph.faces_to_edges(faces)

  np.testing.assert_array_equal(senders, expected_senders)
  np.testing.assert_array_equal(receivers, expected_receivers)
