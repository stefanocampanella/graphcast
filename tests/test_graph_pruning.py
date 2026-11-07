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
"""Tests of mesh pruning (graph_pruning.py) on a hand-made mesh and on the icosahedral mesh on the Earth."""

import numpy as np
import pytest
import xarray as xr

from graphcast import graph_pruning, icosahedral_mesh, mesh_connectivity
from graphcast.gis_utils import cartesian_crs, equirectangular_crs, get_transform
from graphcast.mesh_graph import MeshGraph, TriangleMesh, faces_to_edges

LAT = np.arange(-85.0, 90.0, 10.0)
LON = np.arange(0.0, 360.0, 10.0)


def _two_triangles() -> TriangleMesh:
  """Faces (0, 1, 2) and (1, 3, 2) sharing the edge 1-2."""
  vertices = np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 1.0], [0.0, 1.0, 1.0], [1.0, 1.0, 1.0]])
  return TriangleMesh(vertices=vertices, faces=np.array([[0, 1, 2], [1, 3, 2]]))


@pytest.mark.parametrize(
  ("marked", "mode", "kept_vertices", "faces"),
  [
    ({0}, "any", [0, 1, 2], [[0, 1, 2]]),
    ({1}, "any", [0, 1, 2, 3], [[0, 1, 2], [1, 3, 2]]),
    ({0, 1, 2}, "all", [0, 1, 2], [[0, 1, 2]]),
    ({1, 2}, "all", [], []),
  ],
)
def test_prune_mesh(marked, mode, kept_vertices, faces):
  """Faces with any (all) marked vertex are kept, with their vertices renumbered in order; the map gives the new index
  of each kept vertex."""
  mesh = _two_triangles()
  pruned, vertices_map = graph_pruning.prune_mesh(marked, mesh, mode=mode)

  assert isinstance(pruned, TriangleMesh)
  assert vertices_map == {old: new for new, old in enumerate(kept_vertices)}
  np.testing.assert_array_equal(pruned.vertices, mesh.vertices[kept_vertices].reshape(-1, 3))
  expected_faces = np.array([[vertices_map[v] for v in face] for face in faces], dtype=int).reshape(-1, 3)
  np.testing.assert_array_equal(pruned.faces, expected_faces)


def test_prune_mesh_graph_rebuilds_the_edges():
  """A pruned MeshGraph gets the edges of its remaining faces."""
  mesh = _two_triangles()
  graph = MeshGraph(vertices=mesh.vertices, faces=mesh.faces, edges=faces_to_edges(mesh.faces))
  pruned, _ = graph_pruning.prune_mesh({0}, graph)
  assert isinstance(pruned, MeshGraph)
  np.testing.assert_array_equal(np.stack(pruned.edges), np.stack(faces_to_edges(pruned.faces)))


@pytest.fixture(scope="module")
def earth_mesh():
  """The icosahedral mesh refined twice, on the WGS84 ellipsoid, with GraphCast's query radius."""
  mesh_data = icosahedral_mesh.get_multi_mesh_data(2)
  graph = mesh_data.mesh_graph
  assert mesh_data.mesh_size is not None
  return TriangleMesh(vertices=graph.vertices, faces=graph.faces), 0.6 * mesh_data.mesh_size


@pytest.fixture(scope="module")
def mask() -> np.ndarray:
  """Land north of 30°N."""
  return np.broadcast_to(LAT[:, None] <= 30.0, (LAT.size, LON.size)).copy()


def _latitudes(vertices: np.ndarray) -> np.ndarray:
  return np.asarray(get_transform(cartesian_crs, equirectangular_crs)(vertices))[:, 1]


def test_connected_mesh_nodes_are_both_encoder_receivers_and_decoder_senders(earth_mesh, mask):
  """The mesh vertices kept are those the encoder connects to an ocean point and the decoder connects to one."""
  mesh, radius = earth_mesh
  connected = graph_pruning.get_connected_mesh_nodes(LAT, LON, mesh, mask, radius)

  _, receivers = mesh_connectivity.radius_query_indices(
    grid_latitude=LAT, grid_longitude=LON, mesh=mesh, radius=radius, mask=mask
  )
  senders, _ = mesh_connectivity.get_mesh_to_grid_edges(grid_latitude=LAT, grid_longitude=LON, mesh=mesh, mask=mask)
  assert connected == set(receivers) & set(senders)
  # Far from the ocean points, in the north, no vertex is connected.
  assert _latitudes(mesh.vertices[sorted(connected)]).max() < 60.0
  assert 0 < len(connected) < len(mesh.vertices)


def test_prune_mesh_from_mask(earth_mesh, mask):
  """The mesh keeps the faces whose vertices are all connected to ocean points; boundary nodes are renumbered, and those
  pruned dropped."""
  mesh, radius = earth_mesh
  boundary = np.arange(0, len(mesh.vertices), 7)
  grid_mask = xr.DataArray(mask, dims=("lat", "lon"), coords={"lat": LAT, "lon": LON})

  pruned, pruned_boundary = graph_pruning.prune_mesh_from_mask(mesh, boundary, grid_mask, radius)

  connected = graph_pruning.get_connected_mesh_nodes(LAT, LON, mesh, mask, radius)
  expected, vertices_map = graph_pruning.prune_mesh(connected, mesh, mode="all")
  np.testing.assert_array_equal(pruned.vertices, expected.vertices)
  np.testing.assert_array_equal(pruned.faces, expected.faces)
  np.testing.assert_array_equal(pruned_boundary, [vertices_map[int(n)] for n in boundary if n in vertices_map])
  assert 0 < len(pruned_boundary) < len(boundary)


def test_mesh_within_box(earth_mesh):
  """Only faces with all vertices strictly inside the latitude-longitude box are kept."""
  mesh, _ = earth_mesh
  box = graph_pruning.Box(lat_min=0.0, lat_max=60.0, lon_min=0.0, lon_max=120.0)

  pruned, vertices_map = graph_pruning.get_mesh_within_box(mesh, box)

  lon, lat = np.asarray(get_transform(cartesian_crs, equirectangular_crs)(pruned.vertices)).T
  assert len(pruned.faces) > 0
  assert np.all((lat > box.lat_min) & (lat < box.lat_max) & (lon > box.lon_min) & (lon < box.lon_max))
  np.testing.assert_allclose(pruned.vertices, mesh.vertices[sorted(vertices_map)])
