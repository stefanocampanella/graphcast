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
"""Linear (finite element) basis functions of a triangular mesh, evaluated at points or on a latitude-longitude grid."""

import logging

import numpy as np
import trimesh
from scipy import sparse

from mobydick.mesh_connectivity import _grid_lat_lon_to_coordinates
from mobydick.mesh_graph import Mesh

logger = logging.getLogger(__name__)


def compute_basis(mesh: Mesh, points: np.ndarray, dtype: type[np.floating] = np.float32) -> sparse.csr_matrix:
  """Values of the mesh's linear basis functions at arbitrary points: grid points (mesh to grid) or the vertices of
  another mesh (mesh to mesh).

  Each point is taken to its closest point on the mesh (both in the same coordinates). The basis functions of that
  face's three vertices have the barycentric coordinates of the closest point there, and all others are zero. For grid
  points, this face is the one the decoder connects the point to (`mesh_connectivity.get_mesh_to_grid_edges`).

  On a flat mesh this is scipy's `LinearNDInterpolator` on the same triangles. On the sphere it differs from
  interpolating in a map projection (the draft used a stereographic one) by O(edge²), edges in radians.

  Use coordinates in metres, as the meshes are written: trimesh's closest-point search compares squared distances with
  an absolute tolerance (`trimesh.tol.merge`, 1e-8), so on the unit sphere it returns wrong points for edges below
  about 1°.

  Args:
    mesh: Triangular mesh.
    points: Points [num_points, 3], in the coordinates of the mesh.
    dtype: Type of the values.

  Returns:
    Sparse [num_points, num_mesh_vertices] matrix: 3 stored non-negative values per row, on the vertices of the face
    (some are 0 at its edges and vertices), summing to 1.
  """
  logger.info(f"Evaluating {len(mesh.vertices)} basis functions at {len(points)} points.")
  # Without processing, trimesh keeps the order of the faces (it would merge vertices and drop degenerate faces).
  surface = trimesh.Trimesh(vertices=mesh.vertices, faces=mesh.faces, process=False)
  closest, _, face_indices = trimesh.proximity.closest_point(surface, points)
  faces = mesh.faces[face_indices]
  weights = trimesh.triangles.points_to_barycentric(mesh.vertices[faces], closest)
  # A closest point on an edge or a vertex can get weights a rounding error below zero.
  weights = np.clip(weights, 0.0, None)
  weights /= weights.sum(axis=1, keepdims=True)
  rows = np.repeat(np.arange(len(points)), 3)
  # Exactly 3 entries per row (the vertices of the face, a weight may be 0), like the decoder's edges.
  return sparse.csr_matrix(
    (weights.reshape(-1).astype(dtype), (rows, faces.reshape(-1))), shape=(len(points), len(mesh.vertices))
  )


def compute_grid_basis(
  mesh: Mesh,
  latitudes: np.ndarray,
  longitudes: np.ndarray,
  mask: np.ndarray,
  dtype: type[np.floating] = np.float32,
) -> sparse.csr_matrix:
  """Values of the mesh's linear basis functions at the unmasked points of a latitude-longitude grid.

  Args:
    mesh: Triangular mesh in WGS84 Earth-centred coordinates (as the meshes are written).
    latitudes: Grid latitudes [num_lat].
    longitudes: Grid longitudes [num_lon].
    mask: Boolean [num_lat, num_lon], true at the grid points where the basis is evaluated.
    dtype: Type of the values.

  Returns:
    Sparse [num_unmasked_grid_points, num_mesh_vertices] matrix (see `compute_basis`), rows in the row-major order of
    the unmasked grid points.
  """
  points = _grid_lat_lon_to_coordinates(latitudes, longitudes).reshape(-1, 3)[mask.reshape(-1)]
  return compute_basis(mesh, points, dtype=dtype)
