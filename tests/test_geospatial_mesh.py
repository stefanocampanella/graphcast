"""Tests of the icosahedral multi-mesh files: writing (`graphcast make-icosahedral-mesh`) and reading them back."""

import dataclasses
import pathlib

import gmsh
import numpy as np
import pyproj
import pytest
from click.testing import CliRunner

from graphcast import icosahedral_mesh
from graphcast.geospatial_mesh import make_icosahedral_mesh
from graphcast.geospatial_mesh_utils import MULTI_MESH_EDGES_GROUP, read_mesh, read_mesh_data, write_multi_mesh

SPLITS = 2


def _same_crs(wkt_a: str, wkt_b: str) -> bool:
  return pyproj.CRS(wkt_a) == pyproj.CRS(wkt_b)


def _write_config(tmp_path: pathlib.Path, extra: str = "") -> pathlib.Path:
  path = tmp_path / "icosahedral.toml"
  path.write_text(f'description = "A test mesh"\nlicense = "A test license"\nsplits = {SPLITS}\n' + extra)
  return path


def test_make_icosahedral_mesh(tmp_path):
  """The CLI writes the multi-mesh of the config so that reading it gives it back: vertices, finest faces, the edges
  of all levels in order (also those that are not edges of the finest faces), GraphCast's mesh size, no boundary, the
  spatial reference system, description and license."""
  output = tmp_path / "out" / "icosahedral.msh"
  result = CliRunner().invoke(
    make_icosahedral_mesh, [str(_write_config(tmp_path)), str(output), "--log-level", "warning"]
  )
  assert result.exit_code == 0, result.output

  mesh_data = read_mesh_data(output, gmsh_verbosity=0)

  expected = icosahedral_mesh.get_multi_mesh_data(SPLITS)
  graph, expected_graph = mesh_data.mesh_graph, expected.mesh_graph
  np.testing.assert_allclose(graph.vertices, expected_graph.vertices, rtol=0, atol=1e-6)
  np.testing.assert_array_equal(graph.faces, expected_graph.faces)
  np.testing.assert_array_equal(graph.edges[0], expected_graph.edges[0])
  np.testing.assert_array_equal(graph.edges[1], expected_graph.edges[1])
  # The coarser levels add edges that the finest faces do not have.
  assert len(graph.edges[0]) > 3 * len(graph.faces)
  assert graph.boundary is not None and not graph.boundary.any()
  assert _same_crs(graph.spatial_reference_system, expected_graph.spatial_reference_system)
  assert mesh_data.mesh_size is not None and expected.mesh_size is not None
  np.testing.assert_allclose(mesh_data.mesh_size, expected.mesh_size)
  assert mesh_data.description == ["A test mesh"] and mesh_data.license == ["A test license"]


def test_make_icosahedral_mesh_refuses_existing_output(tmp_path):
  """An existing output is replaced only with --overwrite."""
  output = tmp_path / "icosahedral.msh"
  output.write_text("existing")
  arguments = [str(_write_config(tmp_path)), str(output), "--log-level", "warning"]
  result = CliRunner().invoke(make_icosahedral_mesh, arguments)
  assert result.exit_code != 0 and "--overwrite" in result.output
  assert output.read_text() == "existing"
  assert CliRunner().invoke(make_icosahedral_mesh, [*arguments, "--overwrite"]).exit_code == 0


@pytest.mark.parametrize(
  "settings", ["splits = 2\nunknown = 1\n", "splits = -1\n"], ids=["unknown_key", "negative_splits"]
)
def test_invalid_icosahedral_config_is_a_click_error(tmp_path, settings):
  """Unknown keys and a negative number of refinements are reported as a CLI error before anything is written."""
  config = tmp_path / "icosahedral.toml"
  config.write_text(settings)
  output = tmp_path / "icosahedral.msh"
  result = CliRunner().invoke(make_icosahedral_mesh, [str(config), str(output)])
  assert result.exit_code != 0 and "Invalid configuration" in result.output
  assert not output.exists()


def test_write_multi_mesh_rejects_a_boundary(tmp_path):
  """A boundary cannot be stored next to the multi-mesh edges (both are line elements): meshes with one are rejected."""
  mesh_data = icosahedral_mesh.get_multi_mesh_data(0)
  graph = mesh_data.mesh_graph
  with_boundary = dataclasses.replace(
    mesh_data, mesh_graph=dataclasses.replace(graph, boundary=np.arange(len(graph.vertices)) == 0)
  )
  with pytest.raises(ValueError, match="boundary"):
    write_multi_mesh(with_boundary, tmp_path / "mesh.msh")


def test_read_mesh_data_with_unsorted_node_tags(tmp_path):
  """Node tags in a file need not be sorted: faces, edges and the mesh size refer to the right vertices."""
  mesh_data = icosahedral_mesh.get_multi_mesh_data(1)
  graph = mesh_data.mesh_graph
  num_nodes = len(graph.vertices)
  tags = np.random.default_rng(0).permutation(num_nodes) + 1
  # A different mesh size per vertex, to check its order too.
  mesh_size = np.arange(num_nodes, dtype=float)
  path = tmp_path / "unsorted.msh"
  gmsh.initialize()
  gmsh.clear()
  gmsh.model.add("unsorted")
  gmsh.model.set_attribute("Projection", ["WKT", graph.spatial_reference_system])
  surface = gmsh.model.add_discrete_entity(2)
  gmsh.model.mesh.add_nodes(2, surface, tags, graph.vertices.reshape(-1))
  gmsh.model.mesh.add_elements_by_type(surface, 2, [], tags[graph.faces].reshape(-1))
  curve = gmsh.model.add_discrete_entity(1)
  gmsh.model.mesh.add_elements_by_type(curve, 1, [], tags[np.stack(graph.edges, axis=1)].reshape(-1))
  gmsh.model.add_physical_group(2, [surface], name="mesh")
  gmsh.model.add_physical_group(1, [curve], name=MULTI_MESH_EDGES_GROUP)
  view = gmsh.view.add("MeshSize")
  gmsh.view.add_homogeneous_model_data(view, 0, "unsorted", "NodeData", tags, mesh_size)
  gmsh.option.set_number("PostProcessing.SaveMesh", 1)
  gmsh.view.write(view, str(path))

  read = read_mesh_data(path, gmsh_verbosity=0)

  # Compare through coordinates, whatever order the vertices are read in.
  vertices = read.mesh_graph.vertices
  np.testing.assert_allclose(vertices[read.mesh_graph.faces], graph.vertices[graph.faces], rtol=0, atol=1e-6)
  for read_indices, indices in zip(read.mesh_graph.edges, graph.edges, strict=True):
    np.testing.assert_allclose(vertices[read_indices], graph.vertices[indices], rtol=0, atol=1e-6)
  nearest = np.argmin(np.linalg.norm(vertices[:, None] - graph.vertices[None], axis=-1), axis=1)
  np.testing.assert_array_equal(read.mesh_size, mesh_size[nearest])


@pytest.fixture
def icosahedral_file(tmp_path) -> pathlib.Path:
  path = tmp_path / "icosahedral.msh"
  write_multi_mesh(icosahedral_mesh.get_multi_mesh_data(1), path)
  return path


def test_read_mesh_data_names_a_missing_mesh_size_view(icosahedral_file):
  """A mesh size view that is not in the file is reported by name, with the views that are there (it was a KeyError
  on the first node)."""
  with pytest.raises(ValueError, match=r"No view 'Missing'.*MeshSize"):
    read_mesh_data(icosahedral_file, gmsh_verbosity=0, mesh_size_tag_name="Missing")


def test_read_mesh_without_mesh_size(icosahedral_file):
  """Without a view name, the mesh is read without mesh size, as documented."""
  mesh, mesh_size = read_mesh(icosahedral_file, mesh_size_tag_name=None)
  assert mesh_size is None
  assert len(mesh.vertices) == 42
