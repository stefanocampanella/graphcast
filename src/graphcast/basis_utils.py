"""Linear (finite element) basis functions of a triangular mesh, evaluated on the points of a latitude-longitude grid."""

import logging

import numpy as np
import trimesh
from scipy import sparse

from graphcast.mesh_connectivity import _grid_lat_lon_to_coordinates
from graphcast.mesh_graph import Mesh

logger = logging.getLogger(__name__)


def compute_basis(
  mesh: Mesh,
  latitudes: np.ndarray,
  longitudes: np.ndarray,
  mask: np.ndarray,
  dtype: type[np.floating] = np.float32,
) -> sparse.csr_matrix:
  """Values of the mesh's linear basis functions at the unmasked grid points.

  Each grid point is taken to its closest point on the mesh (the mesh and grid are in the same Earth-centred
  coordinates), which lies in the face the decoder connects the point to (`mesh_connectivity.get_mesh_to_grid_edges`).
  The basis functions of that face's three vertices have the barycentric coordinates of the closest point there, and
  all others are zero.

  Args:
    mesh: Triangular mesh in WGS84 Earth-centred coordinates.
    latitudes: Grid latitudes [num_lat].
    longitudes: Grid longitudes [num_lon].
    mask: Boolean [num_lat, num_lon], true at the grid points where the basis is evaluated.
    dtype: Type of the values.

  Returns:
    Sparse [num_unmasked_grid_points, num_mesh_vertices] matrix, rows in the (row-major) order of the unmasked grid
    points: 3 non-negative values per row, summing to 1.
  """
  points = _grid_lat_lon_to_coordinates(latitudes, longitudes).reshape(-1, 3)[mask.reshape(-1)]
  logger.info(f"Evaluating {len(mesh.vertices)} basis functions at {len(points)} grid points.")
  # Without processing, trimesh keeps the order of the faces (it would merge vertices and drop degenerate faces).
  surface = trimesh.Trimesh(vertices=mesh.vertices, faces=mesh.faces, process=False)
  closest, _, face_indices = trimesh.proximity.closest_point(surface, points)
  faces = mesh.faces[face_indices]
  weights = trimesh.triangles.points_to_barycentric(mesh.vertices[faces], closest)
  # A closest point on an edge or a vertex can get weights a rounding error below zero.
  weights = np.clip(weights, 0.0, None)
  weights /= weights.sum(axis=1, keepdims=True)
  rows = np.repeat(np.arange(len(points)), 3)
  return sparse.csr_matrix(
    (weights.reshape(-1).astype(dtype), (rows, faces.reshape(-1))), shape=(len(points), len(mesh.vertices))
  )
