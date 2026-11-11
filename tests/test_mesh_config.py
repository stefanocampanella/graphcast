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
"""Tests of the settings of the ocean mesh generator (mesh_config.py), of the shipped mesh configs, and of
moby-dick make-ocean-mesh on a synthetic coastline (a ring at 70°S, meshed with a coarse uniform field)."""

import itertools
import pathlib

import networkx as nx
import numpy as np
import pytest
from click.testing import CliRunner
from osgeo import ogr, osr
from pydantic import ValidationError

from mobydick import cli_utils
from mobydick.geospatial_mesh import make_ocean_mesh
from mobydick.geospatial_mesh_utils import PointRefinementField, read_mesh_data
from mobydick.gis_utils import (
  cartesian_crs,
  cartesian_to_lat_lon,
  get_transform,
  lat_lon_to_cartesian,
  stereographic_crs,
)
from mobydick.mesh_config import (
  IcosahedralMeshConfig,
  MeshGenerationConfig,
  ShoreProximityFieldConfig,
  field_arguments,
)
from mobydick.mesh_graph import faces_to_edges

MESH_CONFIGS = pathlib.Path(__file__).parents[1] / "configs" / "mesh"
COAST_LATITUDE = -70.0
MINIMAL = {
  "domain": {"filepath": "coast.shp"},
  "coarsen_boundaries": {"mesh_size": 200_000.0},
  "fields": [{"name": "uniform", "value": 1_000_000.0}],
}


REFINEMENT = {"name": "point_refinement", "points": [[41.12, 29.07]], "size": 250.0, "growth": 0.02}


def _settings(**changes) -> dict:
  return {**MINIMAL, **changes}


def test_defaults():
  """Defaults: the gmsh options and smoothness used before they were configurable, one MeshSize view."""
  configs = MeshGenerationConfig.model_validate(MINIMAL)
  assert (configs.gmsh.verbosity, configs.gmsh.algorithm, configs.gmsh.msh_file_version) == (2, 6, 4.1)
  assert configs.mesh.smoothness == 0.3
  assert configs.view_tag_name == "MeshSize" and configs.model_name == "OceanMesh"
  assert configs.domain.physical_name_field == "featurecla" and configs.domain.curve_type == "bspline"
  assert configs.coarsen_boundaries.x0 == (0.0, 0.0) and configs.coarsen_boundaries.x0_projection == "stereographic"


@pytest.mark.parametrize(
  "path", sorted(set(MESH_CONFIGS.glob("*.toml")) - set(MESH_CONFIGS.glob("icosahedral_*.toml"))), ids=lambda p: p.name
)
def test_shipped_configs_are_valid(path):
  """Every shipped ocean mesh config validates, and a uniform config's name and description give its value in km."""
  configs = cli_utils.read_configs(path, schema=MeshGenerationConfig)
  if path.name.startswith("uniform_") and path.stem.endswith("km"):
    (field,) = [field for field in configs.fields if field.name == "uniform"]
    km = int(path.stem.removeprefix("uniform_").removesuffix("km"))
    assert field.value == km * 1000
    assert f"value of {km} km" in " ".join(configs.description.split())
  # Every ocean mesh keeps the Black Sea: coarsening and mesh size refined at the Turkish straits and Kerch.
  for fields in (configs.coarsen_boundaries.fields, configs.fields):
    (refinement,) = [field for field in fields if field.name == "point_refinement"]
    assert len(refinement.points) == 3
  assert "Black Sea" in configs.description


@pytest.mark.parametrize("path", sorted(MESH_CONFIGS.glob("icosahedral_*.toml")), ids=lambda path: path.name)
def test_shipped_icosahedral_configs_are_valid(path):
  """Every shipped icosahedral mesh config validates, and its name and description give its number of refinements."""
  configs = cli_utils.read_configs(path, schema=IcosahedralMeshConfig)
  assert configs.splits == int(path.stem.removeprefix("icosahedral_"))
  assert f"refined {configs.splits} times" in " ".join(configs.description.split())
  assert configs.license


@pytest.mark.parametrize(
  ("settings", "message"),
  [
    (_settings(unknown=1), "Extra inputs"),
    (_settings(domain={"filepath": "coast.shp", "sampling": 5000.0}), "Extra inputs"),
    (_settings(domain={"filepath": "coast.gpkg"}), "shapefile"),
    (_settings(domain={"filepath": "coast.shp", "curve_type": "spiral"}), "curve_type"),
    (_settings(fields=[]), "at least 1"),
    (_settings(fields=[{"name": "constant", "value": 1.0}]), "name"),
    (_settings(fields=[{"name": "uniform", "value": -1.0}]), "greater than 0"),
    (_settings(fields=[{"name": "uniform", "value": 1.0}, {"name": "uniform", "value": 2.0}]), "repeated"),
    (_settings(view_tag_name="uniform"), "also the name of a field"),
    (_settings(coarsen_boundaries={"mesh_size": 0.0}), "greater than 0"),
    (_settings(coarsen_boundaries={"mesh_size": 1.0, "x0_projection": "mercator"}), "x0_projection"),
    (_settings(gmsh={"algorithm": 0}), "greater than or equal to 1"),
    (
      _settings(
        coarsen_boundaries={"mesh_size": 1.0, "fields": [{**REFINEMENT, "points": [[95.0, 0.0]]}]},
      ),
      "latitude",
    ),
    (_settings(coarsen_boundaries={"mesh_size": 1.0, "fields": [{**REFINEMENT, "points": []}]}), "at least 1"),
    (_settings(coarsen_boundaries={"mesh_size": 1.0, "fields": [REFINEMENT, REFINEMENT]}), "repeated"),
  ],
)
def test_invalid_settings(settings, message):
  """Unknown keys, a non-shapefile domain, missing or repeated fields and out-of-range values are rejected."""
  with pytest.raises(ValidationError, match=message):
    MeshGenerationConfig.model_validate(settings)


SHORE = {"name": "shore_proximity", "filepath": "coast.shp", "sampling": 5000.0, "field_min": 0.0}
BATHYMETRY = {"name": "bathymetry", "filepath": "static", "var_name": "deptho"}


@pytest.mark.parametrize(
  ("field", "message"),
  [
    ({**SHORE, "field_max": 500_000.0, "size_min": 2.0, "size_max": 1.0}, "must not exceed size_max"),
    ({**SHORE, "field_max": 0.0, "size_min": 1.0, "size_max": 2.0}, "greater than 0"),
    (
      {**SHORE, "field_min": 600_000.0, "field_max": 500_000.0, "size_min": 1.0, "size_max": 2.0},
      "less than field_max",
    ),
    ({**BATHYMETRY, "q_low": 0.9, "q_high": 0.1, "size_min": 1.0, "size_max": 2.0}, "less than q_high"),
    ({**BATHYMETRY, "q_high": 1.5, "size_min": 1.0, "size_max": 2.0}, "less than or equal to 1"),
    ({**BATHYMETRY, "size_min": 1.0}, "size_max"),
  ],
)
def test_invalid_field_settings(field, message):
  """Bounded fields need size_min <= size_max; distances and quantiles must be ordered and in range."""
  with pytest.raises(ValidationError, match=message):
    MeshGenerationConfig.model_validate(_settings(fields=[field]))


def test_field_arguments():
  """A field's keyword arguments exclude its name, and its file path is placed under the data path."""
  field = ShoreProximityFieldConfig.model_validate({**SHORE, "field_max": 500_000.0, "size_min": 1.0, "size_max": 2.0})
  arguments = field_arguments(field, pathlib.Path("/data"))
  assert "name" not in arguments
  assert arguments["filepath"] == pathlib.Path("/data/coast.shp")
  assert field_arguments(field)["filepath"] == "coast.shp"


@pytest.fixture
def coastline(tmp_path) -> pathlib.Path:
  """A shapefile with one closed coastline, a ring at 70°S: the ocean is north of it, including the North Pole (the
  origin of the stereographic plane, where boundary coarsening starts)."""
  srs = osr.SpatialReference()
  srs.ImportFromEPSG(4326)
  srs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
  source = ogr.GetDriverByName("ESRI Shapefile").CreateDataSource(str(tmp_path / "coast.shp"))
  layer = source.CreateLayer("coast", srs, ogr.wkbLineString)
  layer.CreateField(ogr.FieldDefn("featurecla", ogr.OFTString))
  line = ogr.Geometry(ogr.wkbLineString)
  for lon in np.append(np.arange(-180.0, 180.0, 5.0), -180.0):
    line.AddPoint_2D(float(lon), COAST_LATITUDE)
  feature = ogr.Feature(layer.GetLayerDefn())
  feature.SetField("featurecla", "Coastline")
  feature.SetGeometry(line)
  layer.CreateFeature(feature)
  source = None  # Closes and writes the shapefile.
  return tmp_path


def _write_config(path: pathlib.Path, extra: str = "") -> pathlib.Path:
  path.write_text(
    'description = "A test mesh"\nlicense = "A test license"\n'
    "[gmsh]\nverbosity = 0\n"
    '[domain]\nfilepath = "coast.shp"\ncurve_type = "polyline"\n'
    "[coarsen_boundaries]\nmesh_size = 200_000.0\n"
    '[[fields]]\nname = "uniform"\nvalue = 1_000_000.0\n' + extra
  )
  return path


def test_make_ocean_mesh(coastline):
  """The CLI meshes the ocean north of the coastline and stores the description and the mesh size at every node."""
  config = _write_config(coastline / "mesh.toml")
  output = coastline / "out" / "mesh.msh"
  result = CliRunner().invoke(
    make_ocean_mesh, [str(config), str(output), "--data-path", str(coastline), "--log-level", "warning"]
  )
  assert result.exit_code == 0, result.output
  mesh_data = read_mesh_data(output, gmsh_verbosity=0)
  vertices = mesh_data.mesh_graph.vertices
  latitudes = np.degrees(np.arcsin(vertices[:, 2] / np.linalg.norm(vertices, axis=1)))
  assert len(vertices) > 100
  # Boundary nodes lie on the (piecewise linear) coastline, a little north of 70°S between its points.
  assert latitudes.min() > COAST_LATITUDE - 0.5
  assert latitudes.max() > 80.0
  assert mesh_data.mesh_size is not None
  np.testing.assert_allclose(mesh_data.mesh_size, 1_000_000.0)
  assert " ".join(mesh_data.description) == "A test mesh" and " ".join(mesh_data.license) == "A test license"
  # The coastline lines are the boundary (not multi-mesh edges): the edges are those of the triangles.
  graph = mesh_data.mesh_graph
  assert graph.boundary is not None and graph.boundary.any() and not graph.boundary.all()
  senders, receivers = faces_to_edges(graph.faces)
  np.testing.assert_array_equal(graph.edges[0], senders)
  np.testing.assert_array_equal(graph.edges[1], receivers)


def test_point_refinement_field():
  """The size is `size` at a point and grows by `growth` per metre of distance from the nearest point, whether the
  points are given in Earth-centred or stereographic coordinates (seamsh uses the latter, moby-dick the former)."""
  field = PointRefinementField(points=[(0.0, 0.0), (0.0, 90.0)], size=250.0, growth=0.02)
  # At the first point, 0.9° east of it, and 0.9° west of the second point (the nearest one).
  at = lat_lon_to_cartesian(np.array([0.0, 0.0, 0.0]), np.array([0.0, 0.9, 89.1]))
  sizes = field.mesh_size_3d(at, cartesian_crs)
  chord = np.linalg.norm(at[1] - at[0])  # ~100 km
  np.testing.assert_allclose(sizes, [250.0, 250.0 + 0.02 * chord, 250.0 + 0.02 * chord], rtol=1e-6)
  stereographic = get_transform(cartesian_crs, stereographic_crs)(at)
  np.testing.assert_allclose(field.mesh_size_3d(stereographic, stereographic_crs), sizes, rtol=1e-6)


@pytest.fixture
def channel_coastline(tmp_path) -> pathlib.Path:
  """The 70°S ring (the ocean's outer boundary in the North Pole stereographic plane) and a continent at the equator
  (lon 0-40, lat ±20) holding an inland sea (lon 15-25, lat ±5) joined to the ocean by a channel 0.2° (22 km) wide
  along the equator from lon 25 to 40, all as Natural Earth-like coastline lines."""
  srs = osr.SpatialReference()
  srs.ImportFromEPSG(4326)
  srs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
  source = ogr.GetDriverByName("ESRI Shapefile").CreateDataSource(str(tmp_path / "coast.shp"))
  layer = source.CreateLayer("coast", srs, ogr.wkbLineString)
  layer.CreateField(ogr.FieldDefn("featurecla", ogr.OFTString))
  ring = [(float(lon), COAST_LATITUDE) for lon in np.append(np.arange(-180.0, 180.0, 5.0), -180.0)]
  continent = [(40, 0.1), (40, 20), (0, 20), (0, -20), (40, -20), (40, -0.1), (25, -0.1), (25, -5), (15, -5), (15, 5)]
  continent += [(25, 5), (25, 0.1), (40, 0.1)]

  def densified(points, step=0.5):
    # Points every `step` degrees at most, as a real coastline has.
    out = []
    for (x0, y0), (x1, y1) in itertools.pairwise(points):
      n = max(1, int(np.ceil(max(abs(x1 - x0), abs(y1 - y0)) / step)))
      out += [(x0 + (x1 - x0) * i / n, y0 + (y1 - y0) * i / n) for i in range(n)]
    return out + [points[-1]]

  for points in (ring, densified(continent)):
    line = ogr.Geometry(ogr.wkbLineString)
    for lon, lat in points:
      line.AddPoint_2D(float(lon), float(lat))
    feature = ogr.Feature(layer.GetLayerDefn())
    feature.SetField("featurecla", "Coastline")
    feature.SetGeometry(line)
    layer.CreateFeature(feature)
  source = None  # Closes and writes the shapefile.
  return tmp_path


@pytest.mark.parametrize("refined", [False, True], ids=["channel_closed", "channel_refined"])
def test_a_narrow_channel_needs_refinement_to_keep_the_sea_behind_it(channel_coastline, refined):
  """A coarsening size larger than a channel's width closes it, and the sea behind it is not meshed; refining the
  coarsening size and the mesh size at the channel (point_refinement, as for the Bosporus) keeps the sea in the one
  connected mesh."""
  channel = '[{name = "point_refinement", points = [[0.0, 32.5]], size = 5_000.0, growth = 0.01}]'
  extra = f"fields = {channel}\n" if refined else ""
  config = channel_coastline / "mesh.toml"
  config.write_text(
    "[gmsh]\nverbosity = 0\n"
    '[domain]\nfilepath = "coast.shp"\ncurve_type = "polyline"\n'
    f"[coarsen_boundaries]\nmesh_size = 200_000.0\n{extra}"
    '[[fields]]\nname = "uniform"\nvalue = 300_000.0\n'
    + (
      '[[fields]]\nname = "point_refinement"\npoints = [[0.0, 32.5]]\nsize = 10_000.0\ngrowth = 0.3\n'
      if refined
      else ""
    )
  )
  output = channel_coastline / "mesh.msh"
  result = CliRunner().invoke(
    make_ocean_mesh, [str(config), str(output), "--data-path", str(channel_coastline), "--log-level", "warning"]
  )
  assert result.exit_code == 0, result.output
  graph = read_mesh_data(output, gmsh_verbosity=0).mesh_graph
  lat, lon = cartesian_to_lat_lon(graph.vertices)
  in_sea = (np.abs(lat) < 4.5) & (lon > 15.5) & (lon < 24.5)
  assert in_sea.any() == refined
  if refined:
    # One mesh: the sea is connected to the ocean through the channel.
    assert nx.number_connected_components(nx.Graph(zip(*graph.edges, strict=True))) == 1


def test_make_ocean_mesh_refuses_existing_output(coastline):
  """An existing output is replaced only with --overwrite."""
  config = _write_config(coastline / "mesh.toml")
  output = coastline / "mesh.msh"
  output.write_text("existing")
  result = CliRunner().invoke(make_ocean_mesh, [str(config), str(output), "--data-path", str(coastline)])
  assert result.exit_code != 0 and "--overwrite" in result.output
  assert output.read_text() == "existing"


def test_invalid_config_is_a_click_error(coastline):
  """Invalid settings are reported as a CLI error before anything is meshed or written."""
  config = _write_config(coastline / "mesh.toml", extra="unknown = 1\n")
  output = coastline / "mesh.msh"
  result = CliRunner().invoke(make_ocean_mesh, [str(config), str(output), "--data-path", str(coastline)])
  assert result.exit_code != 0 and "Invalid configuration" in result.output
  assert not output.exists()


SHORE_FIELD = (
  '[[fields]]\nname = "shore_proximity"\nfilepath = "coast.shp"\ncurve_type = "polyline"\nsampling = 50_000.0\n'
  "field_min = 0.0\nfield_max = 3_000_000.0\nsize_min = 500_000.0\nsize_max = 2_000_000.0\n"
)


def _mesh(coastline: pathlib.Path, name: str, extra: str = "") -> pathlib.Path:
  """Mesh the synthetic coastline with the test settings (plus extra TOML), into coastline / name.msh."""
  config = _write_config(coastline / f"{name}.toml", extra=extra)
  output = coastline / f"{name}.msh"
  result = CliRunner().invoke(make_ocean_mesh, [str(config), str(output), "--data-path", str(coastline)])
  assert result.exit_code == 0, result.output
  return output


def test_the_mesh_is_written_once(coastline):
  """With two fields (a view each, plus their minimum) the file holds the mesh once, and every view reads back."""
  output = _mesh(coastline, "two_fields", extra=SHORE_FIELD)
  assert sum(line.strip() == b"$Nodes" for line in output.open("rb")) == 1
  sizes = {}
  for name in ("MeshSize", "uniform", "shore_proximity"):
    mesh_size = read_mesh_data(output, gmsh_verbosity=0, mesh_size_tag_name=name).mesh_size
    assert mesh_size is not None
    sizes[name] = mesh_size
  np.testing.assert_allclose(sizes["uniform"], 1_000_000.0)
  np.testing.assert_allclose(sizes["MeshSize"], np.minimum(sizes["uniform"], sizes["shore_proximity"]))


def test_meshes_are_read_one_after_the_other(coastline):
  """Reading a mesh does not pick up the views of a mesh read before it, which have other node tags."""
  first = _mesh(coastline, "first")
  second = _mesh(coastline, "second", extra=SHORE_FIELD)
  for path in (first, second, first):
    mesh_data = read_mesh_data(path, gmsh_verbosity=0)
    assert mesh_data.mesh_size is not None
    assert len(mesh_data.mesh_size) == len(mesh_data.mesh_graph.vertices)
  # The first mesh, read last, still has its own (uniform) sizes, not those of the second.
  np.testing.assert_allclose(mesh_data.mesh_size, 1_000_000.0)
