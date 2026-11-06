"""Tests of the coordinate transforms in gis_utils."""

import warnings

import numpy as np
import pytest

from graphcast import icosahedral_mesh
from graphcast.gis_utils import cartesian_crs, cartesian_unit_sphere_crs, equirectangular_crs, get_transform


@pytest.mark.parametrize("crs", [cartesian_unit_sphere_crs, cartesian_crs], ids=["unit_sphere", "wgs84"])
def test_points_on_the_surface_convert_to_latitude_and_longitude_without_warning(crs):
  """Dropping the elevation of points on the surface does not warn, also for float32 coordinates: the tolerance is
  relative to the size of the ellipsoid (it was 1e-8 in absolute terms, below float32 rounding on the unit sphere
  and below float64 rounding in metres)."""
  vertices = icosahedral_mesh.get_multi_mesh_graph(3, crs=crs).vertices.astype(np.float32)
  with warnings.catch_warnings():
    warnings.simplefilter("error")
    lon, lat = np.asarray(get_transform(crs, equirectangular_crs)(vertices)).T
  assert np.all(np.abs(lat) <= 90.0) and np.all(np.abs(lon) <= 180.0)


@pytest.mark.parametrize(
  ("crs", "scale"), [(cartesian_unit_sphere_crs, 1.001), (cartesian_crs, 1.0001)], ids=["unit_sphere", "wgs84"]
)
def test_points_off_the_surface_warn(crs, scale):
  """Points clearly off the surface (here 0.1 % of the radius above it on the unit sphere, 640 m on the Earth) still
  warn, e.g. coordinates in another reference system than declared."""
  vertices = icosahedral_mesh.get_multi_mesh_graph(1, crs=crs).vertices * scale
  with pytest.warns(UserWarning, match="elevation"):
    get_transform(crs, equirectangular_crs)(vertices)
