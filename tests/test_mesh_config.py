"""Tests of the settings of the ocean mesh generator (mesh_config.py), of the shipped mesh configs, and of
graphcast make-ocean-mesh on a synthetic coastline (a ring at 70°S, meshed with a coarse uniform field)."""

import pathlib

import numpy as np
import pytest
from click.testing import CliRunner
from osgeo import ogr, osr
from pydantic import ValidationError

from graphcast import cli_utils
from graphcast.geospatial_mesh import make_ocean_mesh
from graphcast.geospatial_mesh_utils import read_mesh_data
from graphcast.mesh_config import MeshGenerationConfig, ShoreProximityFieldConfig, field_arguments

MESH_CONFIGS = pathlib.Path(__file__).parents[1] / "configs" / "mesh"
COAST_LATITUDE = -70.0
MINIMAL = {
  "domain": {"filepath": "coast.shp"},
  "coarsen_boundaries": {"mesh_size": 200_000.0},
  "fields": [{"name": "uniform", "value": 1_000_000.0}],
}


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


@pytest.mark.parametrize("path", sorted(MESH_CONFIGS.glob("*.toml")), ids=lambda path: path.name)
def test_shipped_configs_are_valid(path):
  """Every shipped mesh config validates, and a uniform config's name and description give its value in km."""
  configs = cli_utils.read_configs(path, schema=MeshGenerationConfig)
  if path.name.startswith("uniform_") and path.stem.endswith("km"):
    (field,) = configs.fields
    km = int(path.stem.removeprefix("uniform_").removesuffix("km"))
    assert field.name == "uniform" and field.value == km * 1000
    assert f"value of {km} km" in " ".join(configs.description.split())


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
