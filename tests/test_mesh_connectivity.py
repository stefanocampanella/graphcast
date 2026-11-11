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
# Ported from the absltest tests of the original GraphCast code to pytest and rewritten for meshes and
# grids in Earth-centred coordinates.
"""Tests for mobydick.mesh_connectivity: grid and mesh in WGS84 Earth-centred coordinates (metres)."""

import numpy as np
import pytest

from mobydick import icosahedral_mesh, mesh_connectivity
from mobydick.mesh_graph import MeshGraph, TriangleMesh, faces_to_edges

# WGS84 ellipsoid.
SEMI_MAJOR_AXIS = 6_378_137.0
FLATTENING = 1 / 298.257223563
ECCENTRICITY_SQUARED = FLATTENING * (2 - FLATTENING)


def _wgs84_to_ecef(latitude: np.ndarray, longitude: np.ndarray) -> np.ndarray:
  """Earth-centred, Earth-fixed coordinates in metres of points on the ellipsoid (degrees in, [..., 3] out)."""
  lat, lon = np.deg2rad(latitude), np.deg2rad(longitude)
  # Prime vertical radius of curvature.
  n = SEMI_MAJOR_AXIS / np.sqrt(1 - ECCENTRICITY_SQUARED * np.sin(lat) ** 2)
  return np.stack(
    [n * np.cos(lat) * np.cos(lon), n * np.cos(lat) * np.sin(lon), n * (1 - ECCENTRICITY_SQUARED) * np.sin(lat)],
    axis=-1,
  )


def _earth_mesh(splits: int = 3) -> MeshGraph:
  """The icosahedral multi-mesh with its vertices on the WGS84 ellipsoid, in the coordinates of the grid."""
  return icosahedral_mesh.get_multi_mesh_graph(splits)


@pytest.fixture
def grid() -> tuple[np.ndarray, np.ndarray]:
  return np.linspace(-75, 75, 6), np.arange(12) * 30.0


@pytest.fixture
def partial_mask() -> np.ndarray:
  mask = np.ones((6, 12), dtype=bool)
  mask[:2, :] = False
  mask[:, 5] = False
  return mask


def test_grid_lat_lon_to_coordinates():
  """Grid points map to WGS84 Earth-centred coordinates in metres, as the meshes are written, with longitudes beyond
  180° wrapped (millimetre agreement with the closed-form formulas)."""
  grid_latitude = np.array([-45.0, 0.0, 45.0, 90.0])
  # Longitudes beyond 180 degrees are wrapped.
  grid_longitude = np.array([0.0, 90.0, 180.0, 270.0, 359.5])

  coordinates = mesh_connectivity._grid_lat_lon_to_coordinates(grid_latitude, grid_longitude)

  lat, lon = np.meshgrid(grid_latitude, grid_longitude, indexing="ij")
  assert coordinates.shape == (4, 5, 3)
  # Millimetre agreement with the closed-form formulas.
  np.testing.assert_allclose(coordinates, _wgs84_to_ecef(lat, lon), rtol=0, atol=1e-3)


def _grid_positions(grid_latitude, grid_longitude) -> np.ndarray:
  lat, lon = np.meshgrid(grid_latitude, grid_longitude, indexing="ij")
  return _wgs84_to_ecef(lat, lon).reshape([-1, 3])


@pytest.mark.parametrize("masked", [False, True], ids=["no_mask", "partial_mask"])
@pytest.mark.parametrize("varying", [False, True], ids=["uniform_radius", "radius_per_vertex"])
def test_radius_query_indices(grid, partial_mask, masked, varying):
  """Encoder edges are exactly the (grid point, mesh vertex) pairs whose distance (straight line) is within the radius
  of the mesh vertex, as a brute-force search finds them; masked points are left out. With GraphCast's radius (0.6 of
  the longest finest edge, more than the circumradius of a face) every valid grid point is reached. A radius per
  vertex is centred on the mesh vertices. (The legacy test's unit-sphere mesh found no edges at all.)"""
  grid_latitude, grid_longitude = grid
  mesh_data = icosahedral_mesh.get_multi_mesh_data(3)
  mesh = mesh_data.mesh_graph
  assert mesh_data.mesh_size is not None
  radius = 0.6 * mesh_data.mesh_size
  if varying:
    # Larger balls around the northern vertices.
    radius = np.where(mesh.vertices[:, 2] > 0, 1.5 * radius, radius)
  mask = partial_mask if masked else None

  grid_senders, mesh_receivers = mesh_connectivity.radius_query_indices(
    grid_latitude=grid_latitude, grid_longitude=grid_longitude, mesh=mesh, radius=radius, mask=mask
  )

  distances = np.linalg.norm(
    _grid_positions(grid_latitude, grid_longitude)[:, None, :] - mesh.vertices[None, :, :], axis=-1
  )
  within = distances <= radius[None, :]
  if masked:
    within &= partial_mask.reshape([-1, 1])
  expected = set(zip(*np.nonzero(within), strict=True))
  actual = set(zip(grid_senders.tolist(), mesh_receivers.tolist(), strict=True))
  assert len(actual) == len(grid_senders)
  assert actual == expected
  # Every valid grid point is reached.
  valid = np.flatnonzero(partial_mask) if masked else np.arange(grid_latitude.size * grid_longitude.size)
  np.testing.assert_array_equal(np.unique(grid_senders), valid)


def test_radius_query_does_not_depend_on_the_workers():
  """The k-d tree query uses all cores by default: the edges, and their order, are the same with one thread, so the
  default does not change the model's graphs."""
  mesh_data = icosahedral_mesh.get_multi_mesh_data(3)
  assert mesh_data.mesh_size is not None
  points = icosahedral_mesh.get_multi_mesh_graph(4).vertices
  edges = [
    mesh_connectivity.get_points_to_mesh_edges(
      senders_position=points, receivers_mesh=mesh_data.mesh_graph, radius=0.6 * mesh_data.mesh_size, workers=workers
    )
    for workers in (1, -1)
  ]
  assert edges[0][0].size > 0
  for one_thread, all_cores in zip(*edges, strict=True):
    np.testing.assert_array_equal(one_thread, all_cores)


@pytest.mark.parametrize("masked", [False, True], ids=["full_mask", "partial_mask"])
def test_get_mesh_to_grid_edges(grid, partial_mask, masked):
  """Decoder edges: each valid grid point receives 3 edges from the vertices of a mesh face containing it (seen from the
  Earth's centre); masked points receive none."""
  grid_latitude, grid_longitude = grid
  mesh = _earth_mesh()
  mask = partial_mask if masked else np.ones_like(partial_mask)

  mesh_senders, grid_receivers = mesh_connectivity.get_mesh_to_grid_edges(
    grid_latitude=grid_latitude, grid_longitude=grid_longitude, mesh=mesh, mask=mask
  )

  # Each valid grid point receives 3 edges, from the vertices of one face of the mesh; masked points none.
  valid = np.flatnonzero(mask)
  assert mesh_senders.shape == grid_receivers.shape == (3 * valid.size,)
  np.testing.assert_array_equal(grid_receivers, np.repeat(valid, 3))
  senders = mesh_senders.reshape([-1, 3])
  faces = {tuple(face) for face in mesh.faces.tolist()}
  assert all(tuple(face) in faces for face in senders.tolist())
  # The face contains the point, seen from the Earth's centre: the point is a non-negative combination of the face's
  # vertices (a small tolerance, as the closest face of a point near an edge may be its neighbour).
  positions = _grid_positions(grid_latitude, grid_longitude)[valid]
  weights = np.linalg.solve(mesh.vertices[senders].transpose(0, 2, 1), positions[..., None])[..., 0]
  weights /= weights.sum(axis=1, keepdims=True)
  assert weights.min() > -0.05


def test_masking_index_maps():
  """Indices into the unmasked points map to indices into all points and back. Indices without a counterpart raise,
  or get the default.

  Regression test for a bug fixed in b8286ea (Nov 2026): `unmasked_to_masked` looked up the index of a point among the
  unmasked ones with `if index := inverse_map.get(n):`, a truth test, so the first unmasked point, whose index is 0,
  was taken as missing: it raised "no index of masked array", or got the default (-1) with `raise_error=False`. Only
  `masked_to_unmasked` is used to build the encoder and decoder edges, so trained models were not affected."""
  mask = np.array([False, True, False, True, True])
  unmasked_to_masked, masked_to_unmasked = mesh_connectivity.get_masking_indices_fns(mask)
  np.testing.assert_array_equal(masked_to_unmasked(np.array([0, 1, 2])), [1, 3, 4])
  np.testing.assert_array_equal(unmasked_to_masked(np.array([1, 3, 4])), [0, 1, 2])
  with pytest.raises(ValueError, match="no index of masked array"):
    unmasked_to_masked(np.array([2]))
  with pytest.raises(ValueError, match="no index of unmasked array"):
    masked_to_unmasked(np.array([3]))
  lenient_to_masked, lenient_to_unmasked = mesh_connectivity.get_masking_indices_fns(mask, raise_error=False)
  np.testing.assert_array_equal(lenient_to_masked(np.array([0, 1])), [-1, 0])
  np.testing.assert_array_equal(lenient_to_unmasked(np.array([2, 3])), [4, -1])


def test_dummy_xscaling_graph():
  """Two meshes as one typed graph: each mesh's edges on its own nodes, and edges from the vertices of the face of
  one mesh closest to each vertex of the other (3 per receiving vertex)."""
  coarse = icosahedral_mesh.get_multi_mesh_graph(0)
  fine = icosahedral_mesh.get_multi_mesh_graph(1)
  coarse = TriangleMesh(vertices=coarse.vertices, faces=coarse.faces)
  fine = TriangleMesh(vertices=fine.vertices, faces=fine.faces)

  graph = mesh_connectivity.get_dummy_xscaling_graph(coarse, fine, name_a="coarse", name_b="fine")

  for name, mesh in (("coarse", coarse), ("fine", fine)):
    edges = graph.edges[graph.edge_key_by_name(name)].indices
    np.testing.assert_array_equal(np.stack([edges.senders, edges.receivers]), np.stack(faces_to_edges(mesh.faces)))
  for senders_name, senders_mesh, receivers_mesh in (("coarse", coarse, fine), ("fine", fine, coarse)):
    receivers_name = "fine" if senders_name == "coarse" else "coarse"
    key = graph.edge_key_by_name(f"{senders_name}_to_{receivers_name}")
    assert key.node_sets == (senders_name, receivers_name)
    edges = graph.edges[key].indices
    np.testing.assert_array_equal(edges.receivers, np.repeat(np.arange(len(receivers_mesh.vertices)), 3))
    faces = {tuple(face) for face in senders_mesh.faces.tolist()}
    assert all(tuple(face) in faces for face in edges.senders.reshape(-1, 3).tolist())
