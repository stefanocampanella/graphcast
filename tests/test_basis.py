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
"""Tests of the linear basis functions of a mesh at points and on the grid (basis_utils, compute-grid-basis)."""

import numpy as np
import pytest
from click.testing import CliRunner
from scipy import sparse
from scipy.interpolate import LinearNDInterpolator
from scipy.spatial import ConvexHull, Delaunay

from mobydick import basis_utils, icosahedral_mesh, mesh_connectivity
from mobydick.basis import compute_grid_basis_command
from mobydick.dataloader_utils import get_grid
from mobydick.geospatial_mesh_utils import write_multi_mesh
from mobydick.gis_utils import cartesian_crs, cartesian_unit_sphere_crs, equirectangular_crs, get_transform
from mobydick.mesh_graph import TriangleMesh

LAT = np.arange(-85.0, 90.0, 10.0)
LON = np.arange(0.0, 360.0, 10.0)
# Mean Earth radius in metres: a sphere of the size of the meshes, which are in metres.
EARTH_RADIUS = 6.371e6


@pytest.fixture(scope="module")
def mesh():
  return icosahedral_mesh.get_multi_mesh_graph(2)


@pytest.fixture(scope="module")
def mask() -> np.ndarray:
  # Land north of 60°N and on a band of longitudes.
  mask = np.ones((LAT.size, LON.size), dtype=bool)
  mask[LAT > 60.0, :] = False
  mask[:, (LON >= 100.0) & (LON < 140.0)] = False
  return mask


def test_basis_is_the_linear_interpolation_in_the_decoder_face(mesh, mask):
  """Each ocean grid point gets the barycentric weights of its closest point in the mesh face that contains it (the face
  the decoder connects it to): 3 non-negative weights per row summing to 1, on the vertices of that face, which give the
  closest point back. The draft interpolated a dense identity matrix (2 TB for the 25 km mesh) on a new triangulation of
  a stereographic projection."""
  # In float64, to check the geometry (float32 weights place the closest point within centimetres on these faces).
  basis = basis_utils.compute_grid_basis(mesh, latitudes=LAT, longitudes=LON, mask=mask, dtype=np.float64)

  num_valid = int(mask.sum())
  assert basis.shape == (num_valid, len(mesh.vertices))
  assert isinstance(basis, sparse.csr_matrix)
  np.testing.assert_array_equal(np.diff(basis.indptr), 3)
  assert basis.data.min() >= 0.0
  np.testing.assert_allclose(basis @ np.ones(basis.shape[1]), 1.0, rtol=1e-6)
  # The same faces as the decoder edges.
  senders, receivers = mesh_connectivity.get_mesh_to_grid_edges(
    grid_latitude=LAT, grid_longitude=LON, mesh=mesh, mask=mask
  )
  np.testing.assert_array_equal(np.repeat(np.arange(num_valid), 3), np.searchsorted(np.flatnonzero(mask), receivers))
  for row, face in enumerate(senders.reshape(-1, 3)):
    assert set(basis.indices[basis.indptr[row] : basis.indptr[row + 1]]) == set(face)
  # The weighted vertices are the closest point of the face: the offset to the grid point is normal to the face where
  # the closest point is inside it (all weights positive).
  points = mesh_connectivity._grid_lat_lon_to_coordinates(LAT, LON).reshape(-1, 3)[mask.ravel()]
  closest = basis @ mesh.vertices
  faces = senders.reshape(-1, 3)
  a, b, c = (mesh.vertices[faces[:, i]] for i in range(3))
  normals = np.cross(b - a, c - a)
  normals /= np.linalg.norm(normals, axis=1, keepdims=True)
  offsets = points - closest
  inside = np.asarray((basis.toarray() > 1e-6).sum(axis=1)).ravel() == 3
  assert inside.mean() > 0.8
  tangential = offsets[inside] - (offsets[inside] * normals[inside]).sum(axis=1, keepdims=True) * normals[inside]
  np.testing.assert_allclose(tangential, 0.0, atol=1e-3)


def test_basis_at_the_mesh_vertices_is_the_identity(mesh):
  """At the mesh's own vertices each basis function is 1 at its vertex and 0 elsewhere."""
  basis = basis_utils.compute_basis(mesh, mesh.vertices, dtype=np.float64)
  np.testing.assert_allclose(basis.toarray(), np.eye(len(mesh.vertices)), atol=1e-9)


def test_basis_at_the_vertices_of_a_finer_mesh():
  """Mesh to mesh: at the vertices of the next refinement of an icosahedral mesh (on the unit sphere, where the new
  vertices lie radially above the midpoints of the coarse edges), the coarse basis is 1 at the shared vertices and (1/2,
  1/2) on the two ends of the split edge, e.g. for hierarchical meshes."""
  _, fine = icosahedral_mesh.get_hierarchy_of_triangular_meshes_for_sphere(1)
  coarse = icosahedral_mesh.get_multi_mesh_graph(0, crs=cartesian_unit_sphere_crs)
  basis = basis_utils.compute_basis(coarse, fine.vertices.astype(np.float64), dtype=np.float64).toarray()

  num_coarse = len(coarse.vertices)
  np.testing.assert_allclose(basis[:num_coarse], np.eye(num_coarse), atol=1e-6)
  new = basis[num_coarse:]
  assert np.all((new > 1e-6).sum(axis=1) == 2)
  np.testing.assert_allclose(np.sort(new, axis=1)[:, -2:], 0.5, atol=1e-6)
  # The two ends are a coarse edge whose midpoint, pushed to the sphere, is the new vertex.
  ends = np.argsort(new, axis=1)[:, -2:]
  midpoints = coarse.vertices[ends].mean(axis=1)
  np.testing.assert_allclose(
    midpoints / np.linalg.norm(midpoints, axis=1, keepdims=True), fine.vertices[num_coarse:], atol=1e-6
  )


def _stereographic(xyz: np.ndarray) -> np.ndarray:
  """Projection from the south pole onto the equatorial plane: the north pole maps to the origin."""
  radius = np.linalg.norm(xyz, axis=1, keepdims=True)
  return 2.0 * xyz[:, :2] / (radius + xyz[:, 2:3])


def _cap(num_points: int, radius_deg: float, rng: np.random.Generator | None = None) -> np.ndarray:
  """Points on the Earth (in metres) within radius_deg of the north pole: a Fibonacci spiral, quasi-uniform, or uniform
  random points if rng is given."""
  cos_max = np.cos(np.deg2rad(radius_deg))
  if rng is None:
    i = np.arange(num_points) + 0.5
    z, phi = 1.0 - (1.0 - cos_max) * i / num_points, np.pi * (1.0 + 5**0.5) * i
  else:
    z, phi = rng.uniform(cos_max, 1.0, num_points), rng.uniform(0.0, 2.0 * np.pi, num_points)
  r = np.sqrt(1.0 - z**2)
  return EARTH_RADIUS * np.stack([r * np.cos(phi), r * np.sin(phi), z], axis=1)


def test_basis_equals_scipy_linear_interpolation_on_a_flat_mesh():
  """On a flat triangulation, the basis equals scipy's `LinearNDInterpolator` on the same triangles applied to the
  identity matrix (the values of every basis function), which the draft computed. Points slightly off the plane get the
  values at their orthogonal projection."""
  rng = np.random.default_rng(0)
  xy = rng.uniform(0.0, 1e6, size=(60, 2))
  triangulation = Delaunay(xy)
  mesh = TriangleMesh(vertices=np.column_stack([xy, np.zeros(len(xy))]), faces=triangulation.simplices)
  query = rng.uniform(1e5, 9e5, size=(500, 2))
  query = query[triangulation.find_simplex(query) >= 0]
  offsets = rng.uniform(-1e3, 1e3, size=len(query))

  basis = basis_utils.compute_basis(mesh, np.column_stack([query, offsets]), dtype=np.float64).toarray()

  expected = LinearNDInterpolator(triangulation, np.eye(len(xy)))(query)
  np.testing.assert_allclose(basis, expected, atol=1e-9)


def test_basis_equals_scipy_linear_interpolation_in_cartesian_coordinates():
  """On a closed mesh on the Earth, the basis equals scipy's `LinearNDInterpolator` in 3D Cartesian coordinates.

  scipy interpolates in the tetrahedra of a 3D Delaunay triangulation of the vertices. The mesh is convex, so its faces
  are the boundary facets of that triangulation (checked first), and on a facet the tetrahedral interpolation is the
  barycentric interpolation in the face, as in the basis. Points on the sphere between the vertices lie outside the
  polyhedron (scipy gives NaN there), so the query points are random points on the faces, which the basis gets moved
  along the face normal by up to 1 km. scipy gets them moved inwards by 1e-9 of the Earth's radius (6 mm): exactly on
  the boundary, rounding puts some outside. The interior vertex of their tetrahedron then weighs up to about 1e-6."""
  mesh = icosahedral_mesh.get_multi_mesh_graph(3)
  vertices = np.asarray(mesh.vertices, dtype=np.float64)
  faces = np.asarray(mesh.faces)
  hull_faces = {tuple(sorted(face)) for face in ConvexHull(vertices).simplices}
  assert hull_faces == {tuple(sorted(face)) for face in faces}

  rng = np.random.default_rng(0)
  face_ids = rng.integers(0, len(faces), 2000)
  corners = vertices[faces[face_ids]]
  on_faces = (rng.dirichlet(np.ones(3), size=len(face_ids))[:, :, None] * corners).sum(axis=1)
  normals = np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0])
  normals /= np.linalg.norm(normals, axis=1, keepdims=True)
  offsets = rng.uniform(-1e3, 1e3, size=(len(face_ids), 1))

  basis = basis_utils.compute_basis(mesh, on_faces + offsets * normals, dtype=np.float64).toarray()

  expected = LinearNDInterpolator(Delaunay(vertices), np.eye(len(vertices)))(on_faces * (1.0 - 1e-9))
  np.testing.assert_allclose(basis, expected, atol=1e-5)


@pytest.mark.parametrize("cap_radius", [20.0, 5.0, 2.5])
def test_basis_against_the_stereographic_draft(cap_radius):
  """The draft interpolated in a stereographic projection (scipy's `LinearNDInterpolator`), the basis on the flat faces
  of the mesh: on the same triangles, they differ by the curvature of the projection, O(edge²) (edges in radians).
  The caps have edges of 1.9°, 0.47° and 0.24° (the 25 km mesh: 0.22°). The difference is far below the interpolation
  error itself.

  In metres, as the meshes are written: trimesh's closest point compares squared distances with an absolute tolerance
  (1e-8), so on the unit sphere it picks wrong points for edges below about 1°."""
  vertices = _cap(400, cap_radius)
  faces = Delaunay(_stereographic(vertices)).simplices
  mesh = TriangleMesh(vertices=vertices, faces=faces)
  points = _cap(2000, 0.6 * cap_radius, rng=np.random.default_rng(0))

  basis = basis_utils.compute_basis(mesh, points, dtype=np.float64).toarray()

  draft = LinearNDInterpolator(_stereographic(vertices), np.eye(len(vertices)))(_stereographic(points))
  edge = np.median(np.linalg.norm(vertices[faces[:, 0]] - vertices[faces[:, 1]], axis=1)) / EARTH_RADIUS
  # Also near an edge, where the two may pick the two faces on either side, with O(edge²) weights on their third
  # vertices.
  assert np.abs(basis - draft).max() < 1.5 * edge**2

  def smooth(xyz):
    x, y, z = (xyz / EARTH_RADIUS).T
    return np.sin(3.0 * x) * np.cos(2.0 * y) + z

  difference = np.abs(basis @ smooth(vertices) - draft @ smooth(vertices)).max()
  assert difference < 0.5 * np.abs(basis @ smooth(vertices) - smooth(points)).max()


def test_compute_cli(tmp_path, mask_path):
  """`moby-dick compute-grid-basis` reads the mesh and the mask written by moby-dick download-mask, and saves the basis
  on the ocean points of that grid; an existing output is replaced only with --overwrite."""
  mesh_path = tmp_path / "mesh.msh"
  write_multi_mesh(icosahedral_mesh.get_multi_mesh_data(1), mesh_path)
  output = tmp_path / "basis.npz"
  arguments = [str(mesh_path), mask_path, str(output), "--log-level", "warning"]

  result = CliRunner().invoke(compute_grid_basis_command, arguments)

  assert result.exit_code == 0, result.output
  basis = sparse.load_npz(output)
  grid = get_grid(mask_path)
  assert basis.shape == (int(grid.mask.sum()), 42)
  np.testing.assert_allclose(basis @ np.ones(basis.shape[1]), 1.0, rtol=1e-6)  # ty: ignore[no-matching-overload]
  result = CliRunner().invoke(compute_grid_basis_command, arguments)
  assert result.exit_code != 0 and "--overwrite" in result.output
  assert CliRunner().invoke(compute_grid_basis_command, [*arguments, "--overwrite"]).exit_code == 0


def _smooth_function(lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
  """A smooth function on the Earth, of geodetic latitude and longitude in degrees."""
  lat, lon = np.deg2rad(lat), np.deg2rad(lon)
  return np.sin(lat) ** 2 + np.cos(lat) * np.sin(2 * lon) + 0.5 * np.cos(lat) ** 2 * np.cos(3 * lon)


def test_grid_basis_interpolates_a_function_given_at_the_mesh_vertices(mask):
  """The basis times the values of a smooth function at the mesh vertices gives its values at the ocean grid points
  within the error of linear interpolation, which shrinks as the square of the edge length (4 times per refinement)."""
  lat_grid, lon_grid = np.meshgrid(LAT, LON, indexing="ij")
  expected = _smooth_function(lat_grid, lon_grid)[mask]
  to_lat_lon = get_transform(cartesian_crs, equirectangular_crs)
  errors = []
  for splits in (3, 4, 5):
    mesh = icosahedral_mesh.get_multi_mesh_graph(splits)
    lon_mesh, lat_mesh = np.asarray(to_lat_lon(mesh.vertices)).T
    basis = basis_utils.compute_grid_basis(mesh, latitudes=LAT, longitudes=LON, mask=mask, dtype=np.float64)
    errors.append(np.abs(basis @ _smooth_function(lat_mesh, lon_mesh) - expected).max())
  # Finest edges about 950, 480 and 240 km: maximum errors 0.071, 0.023, 0.0058 (the function's second derivatives
  # reach about 4.5).
  assert errors[-1] < 1e-2
  ratios = np.array(errors[:-1]) / np.array(errors[1:])
  assert np.all((ratios > 3.0) & (ratios < 5.5)), (errors, ratios)
