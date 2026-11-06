# Copyright 2023 DeepMind Technologies Limited.
# Copyright 2026 Stefano Campanella.
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
# Ported from the absltest tests of the original GraphCast code to pytest; the merging tests moved to
# test_mesh_graph.py.
"""Tests for icosahedral_mesh."""

import chex
import numpy as np
import pyproj
import pytest

from graphcast import icosahedral_mesh, mesh_graph
from graphcast.gis_utils import cartesian_crs, cartesian_unit_sphere_crs

# WGS84 ellipsoid.
SEMI_MAJOR_AXIS = 6_378_137.0
SEMI_MINOR_AXIS = SEMI_MAJOR_AXIS * (1 - 1 / 298.257223563)


def _get_mesh_spec(splits: int):
  """Returns size of the final icosahedral mesh resulting from the splitting."""
  num_vertices = 12
  num_faces = 20
  for _ in range(splits):
    # Each previous face adds three new vertices, but each vertex is shared
    # by two faces.
    num_vertices += num_faces * 3 // 2
    num_faces *= 4
  return num_vertices, num_faces


def test_icosahedron():
  """The base icosahedron has 12 unit vertices and 20 outward-oriented faces."""
  mesh = icosahedral_mesh.get_icosahedron()
  _assert_valid_mesh(mesh, num_expected_vertices=12, num_expected_faces=20)


@pytest.mark.parametrize("splits", list(range(5)))
def test_get_hierarchy_of_triangular_meshes_for_sphere(splits):
  """Each refinement has the expected numbers of vertices and faces, stays on the unit sphere with outward faces, and
  starts with the vertices of the coarser level (nesting, which multi-meshes rely on)."""
  meshes = icosahedral_mesh.get_hierarchy_of_triangular_meshes_for_sphere(splits=splits)
  prev_vertices = None
  for mesh_i, mesh in enumerate(meshes):
    # Check that `mesh` is valid.
    num_expected_vertices, num_expected_faces = _get_mesh_spec(mesh_i)
    _assert_valid_mesh(mesh, num_expected_vertices, num_expected_faces)

    # Check that the first N vertices from this mesh match all of the
    # vertices from the previous mesh.
    if prev_vertices is not None:
      leading_mesh_vertices = mesh.vertices[: prev_vertices.shape[0]]
      np.testing.assert_array_equal(leading_mesh_vertices, prev_vertices)

    # Increase the expected/previous values for the next iteration.
    if mesh_i < len(meshes) - 1:
      prev_vertices = mesh.vertices


def _same_crs(wkt: str, crs) -> bool:
  return pyproj.CRS(wkt) == pyproj.CRS(crs.to_wkt())


def _spherical_lat_lon(vertices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
  """Latitude and longitude (degrees) of points on the unit sphere."""
  vertices = vertices.astype(np.float64)
  lat = np.rad2deg(np.arcsin(vertices[:, 2] / np.linalg.norm(vertices, axis=-1)))
  lon = np.rad2deg(np.arctan2(vertices[:, 1], vertices[:, 0]))
  return lat, lon


@pytest.mark.parametrize("splits", [0, 2])
def test_get_multi_mesh_graph(splits):
  """The multi-mesh is the merged hierarchy with its vertices moved onto the WGS84 ellipsoid (Earth-centred, in
  metres), keeping each vertex's latitude and longitude, and says so in its spatial reference system; it has no
  boundary and no seamsh node tags."""
  hierarchy = icosahedral_mesh.get_hierarchy_of_triangular_meshes_for_sphere(splits)
  merged = mesh_graph.merge_meshes(hierarchy)

  mesh = icosahedral_mesh.get_multi_mesh_graph(splits)

  assert isinstance(mesh, mesh_graph.MeshGraph)
  np.testing.assert_array_equal(mesh.faces, merged.faces)
  np.testing.assert_array_equal(mesh.edges[0], merged.edges[0])
  np.testing.assert_array_equal(mesh.edges[1], merged.edges[1])
  assert mesh.boundary is None
  assert mesh.node_tags is None
  assert _same_crs(mesh.spatial_reference_system, cartesian_crs)
  # On the ellipsoid: (x² + y²) / a² + z² / b² = 1.
  x, y, z = mesh.vertices.T
  np.testing.assert_allclose((x**2 + y**2) / SEMI_MAJOR_AXIS**2 + z**2 / SEMI_MINOR_AXIS**2, 1.0, rtol=0, atol=1e-12)
  # At the geodetic latitude and longitude equal to the spherical ones of the unit-sphere vertices (millimetres).
  lat, lon = np.deg2rad(_spherical_lat_lon(merged.vertices))
  eccentricity_squared = 1 - (SEMI_MINOR_AXIS / SEMI_MAJOR_AXIS) ** 2
  n = SEMI_MAJOR_AXIS / np.sqrt(1 - eccentricity_squared * np.sin(lat) ** 2)
  expected = np.stack(
    [n * np.cos(lat) * np.cos(lon), n * np.cos(lat) * np.sin(lon), n * (1 - eccentricity_squared) * np.sin(lat)],
    axis=-1,
  )
  np.testing.assert_allclose(mesh.vertices, expected, rtol=0, atol=1e-3)


def test_get_multi_mesh_graph_on_the_unit_sphere():
  """Asked for the unit sphere, the multi-mesh keeps the vertices of the hierarchy."""
  merged = mesh_graph.merge_meshes(icosahedral_mesh.get_hierarchy_of_triangular_meshes_for_sphere(2))

  mesh = icosahedral_mesh.get_multi_mesh_graph(2, crs=cartesian_unit_sphere_crs)

  np.testing.assert_allclose(mesh.vertices, merged.vertices, rtol=0, atol=1e-7)
  assert _same_crs(mesh.spatial_reference_system, cartesian_unit_sphere_crs)


def _assert_valid_mesh(mesh, num_expected_vertices, num_expected_faces):
  vertices = mesh.vertices
  faces = mesh.faces
  chex.assert_shape(vertices, [num_expected_vertices, 3])
  chex.assert_shape(faces, [num_expected_faces, 3])

  # Vertices norm should be 1, and the mesh says so.
  vertices_norm = np.linalg.norm(vertices, axis=-1)
  np.testing.assert_allclose(vertices_norm, 1.0, rtol=1e-6)
  assert _same_crs(mesh.spatial_reference_system, cartesian_unit_sphere_crs)

  _assert_positive_face_orientation(vertices, faces)


def _assert_positive_face_orientation(vertices, faces):
  # Obtain a unit vector that points, in the direction of the face.
  face_orientation = np.cross(
    vertices[faces[:, 1]] - vertices[faces[:, 0]], vertices[faces[:, 2]] - vertices[faces[:, 1]]
  )
  face_orientation /= np.linalg.norm(face_orientation, axis=-1, keepdims=True)

  # And a unit vector pointing from the origin to the center of the face.
  face_centers = vertices[faces].mean(1)
  face_centers /= np.linalg.norm(face_centers, axis=-1, keepdims=True)

  # Positive orientation means those two vectors should be parallel
  # (dot product, 1), and not anti-parallel (dot product, -1).
  dot_center_orientation = np.einsum("ik,ik->i", face_orientation, face_centers)

  # Check that the face normal is parallel to the vector that joins the center
  # of the face to the center of the sphere. Note we need a small tolerance
  # because some discretizations are not exactly uniform, so it will not be
  # exactly parallel.
  np.testing.assert_allclose(dot_center_orientation, 1.0, atol=6e-4)
