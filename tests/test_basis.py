"""Tests of the linear basis functions of a mesh on the grid (basis_utils.py, graphcast compute)."""

import os
import pathlib
import subprocess
import sys

import numpy as np
import pytest
from click.testing import CliRunner
from scipy import sparse

from graphcast import basis_utils, icosahedral_mesh, mesh_connectivity
from graphcast.basis import compute
from graphcast.dataloader_utils import get_grid
from graphcast.geospatial_mesh_utils import write_multi_mesh

LAT = np.arange(-85.0, 90.0, 10.0)
LON = np.arange(0.0, 360.0, 10.0)


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
  """Each ocean grid point gets the barycentric weights of its closest point in the mesh face that contains it (the
  face the decoder connects it to): 3 non-negative weights per row summing to 1, on the vertices of that face, which
  give the closest point back. The draft interpolated a dense identity matrix (2 TB for the 25 km mesh) on a new
  triangulation of a stereographic projection."""
  # In float64, to check the geometry (float32 weights place the closest point within centimetres on these faces).
  basis = basis_utils.compute_basis(mesh, latitudes=LAT, longitudes=LON, mask=mask, dtype=np.float64)

  num_valid = int(mask.sum())
  assert basis.shape == (num_valid, len(mesh.vertices))
  assert isinstance(basis, sparse.csr_matrix)
  np.testing.assert_array_equal(np.diff(basis.indptr), 3)
  assert basis.data.min() >= 0.0
  np.testing.assert_allclose(np.asarray(basis.sum(axis=1)).ravel(), 1.0, rtol=1e-6)
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


def test_compute_cli(tmp_path, mask_path):
  """`graphcast compute` reads the mesh and the mask written by graphcast download-mask, and saves the basis on the
  ocean points of that grid; an existing output is replaced only with --overwrite."""
  mesh_path = tmp_path / "mesh.msh"
  write_multi_mesh(icosahedral_mesh.get_multi_mesh_data(1), mesh_path)
  output = tmp_path / "basis.npz"
  arguments = [str(mesh_path), mask_path, str(output), "--log-level", "warning"]

  result = CliRunner().invoke(compute, arguments)

  assert result.exit_code == 0, result.output
  basis = sparse.load_npz(output)
  grid = get_grid(mask_path)
  assert basis.shape == (int(grid.mask.sum()), 42)
  np.testing.assert_allclose(np.asarray(basis.sum(axis=1)).ravel(), 1.0, rtol=1e-6)
  result = CliRunner().invoke(compute, arguments)
  assert result.exit_code != 0 and "--overwrite" in result.output
  assert CliRunner().invoke(compute, [*arguments, "--overwrite"]).exit_code == 0


def test_basis_module_runs_as_a_script():
  """`python -m graphcast.basis` runs the command (it called an undefined `cli`)."""
  source = pathlib.Path(__file__).parents[1] / "src"
  result = subprocess.run(
    [sys.executable, "-m", "graphcast.basis", "--help"],
    env={**os.environ, "PYTHONPATH": str(source)},
    capture_output=True,
    text=True,
    check=False,
  )
  assert result.returncode == 0, result.stderr
  assert "MESH_PATH MASK_PATH OUTPUT_PATH" in result.stdout
