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
"""Tests of the coordinate transforms and grid helpers in gis_utils."""

import warnings

import numpy as np
import pytest
import xarray as xr
from osgeo import gdal

from mobydick import gis_utils, icosahedral_mesh
from mobydick.gis_utils import (
  cartesian_crs,
  cartesian_to_lat_lon,
  cartesian_unit_sphere_crs,
  equirectangular_crs,
  get_components_transform,
  get_transform,
  lat_lon_to_cartesian,
)


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


def test_transform_round_trip_of_arrays_and_components():
  """Longitude-latitude pairs go to Earth-centred coordinates and back: `get_transform` always takes and returns one
  array with the coordinates on the last axis (also with more leading axes), `get_components_transform` one array per
  coordinate, and both agree."""
  lon, lat = np.array([0.0, 90.0, -45.0]), np.array([0.0, 45.0, -30.0])
  to_cartesian = get_transform(equirectangular_crs, cartesian_crs)

  xyz = to_cartesian(np.stack([lon, lat], axis=-1))
  assert isinstance(xyz, np.ndarray) and xyz.shape == (3, 3)
  np.testing.assert_allclose(xyz[0], [6_378_137.0, 0.0, 0.0], atol=1e-6)
  back = get_transform(cartesian_crs, equirectangular_crs)(xyz.reshape(1, 3, 3))
  assert back.shape == (1, 3, 2)
  np.testing.assert_allclose(back[0], np.stack([lon, lat], axis=-1), atol=1e-9)
  components = get_components_transform(cartesian_crs, equirectangular_crs)(tuple(xyz.T))
  assert isinstance(components, tuple) and len(components) == 2
  np.testing.assert_allclose(components, (lon, lat), atol=1e-9)
  np.testing.assert_allclose(get_components_transform(equirectangular_crs, cartesian_crs)((lon, lat)), tuple(xyz.T))


@pytest.mark.parametrize("unit_sphere", [False, True], ids=["wgs84", "unit_sphere"])
def test_lat_lon_helpers_use_the_lat_lon_order(unit_sphere):
  """`cartesian_to_lat_lon` returns latitudes first and `lat_lon_to_cartesian` takes them first (the CRS axis order
  is longitude first, which callers had to swap and transpose); they invert each other, also with leading axes."""
  lat, lon = np.array([[10.0, -60.0], [0.0, 45.0]]), np.array([[20.0, 170.0], [-90.0, 0.0]])
  xyz = lat_lon_to_cartesian(lat, lon, unit_sphere=unit_sphere)
  assert xyz.shape == (2, 2, 3)
  # The point at latitude 0, longitude -90 is on the negative y axis.
  assert xyz[1, 0, 1] < 0 and abs(xyz[1, 0, 0]) < 1e-6 * abs(xyz[1, 0, 1])
  back_lat, back_lon = cartesian_to_lat_lon(xyz, unit_sphere=unit_sphere)
  np.testing.assert_allclose(back_lat, lat, atol=1e-9)
  np.testing.assert_allclose(back_lon, lon, atol=1e-9)


def test_transform_rejects_wrong_coordinates():
  to_cartesian = get_transform(equirectangular_crs, cartesian_crs)
  with pytest.raises(AssertionError, match="Expected 2 coordinates"):
    to_cartesian(np.zeros((4, 3)))
  with pytest.raises(TypeError, match="numpy array"):
    to_cartesian([[0.0, 0.0]])  # ty: ignore[invalid-argument-type]  # the wrong type under test


def test_map_on_grid():
  """A function of (longitude, latitude) points is evaluated on a grid and returned as a (lat, lon) DataArray."""
  lat = xr.DataArray(np.array([-30.0, 0.0, 30.0]), dims="lat", name="lat")
  lon = xr.DataArray(np.array([0.0, 120.0, 240.0, 300.0]), dims="lon", name="lon")

  def func(coordinates, crs):
    assert crs is equirectangular_crs
    return coordinates[..., 0] + 1000.0 * coordinates[..., 1]

  values = gis_utils.map_on_grid(func, lat, lon)
  assert values.dims == ("lat", "lon")
  np.testing.assert_array_equal(values.values, lon.values[None, :] + 1000.0 * lat.values[:, None])


@pytest.mark.parametrize(
  ("lon", "centres"),
  [
    (np.arange(0.0, 360.0, 90.0), [-90.0, 0.0, 90.0, 180.0]),
    (np.arange(-180.0, 180.0, 90.0), [-180.0, -90.0, 0.0, 90.0]),
  ],
  ids=["0_360", "wrapped"],
)
def test_xarray_to_gdal_raster(lon, centres):
  """A (lat, lon) field becomes a north-up raster in longitudes wrapped to [-180, 180] and latitudes, with pixel
  centres on the grid points and NaN as no-data; already wrapped longitudes work too (wrapping raised an
  IndexError)."""
  lat = np.array([-45.0, 0.0, 45.0])
  field = xr.DataArray(
    lat[:, None] * 1000.0 + np.where(lon > 180.0, lon - 360.0, lon)[None, :],
    dims=("lat", "lon"),
    coords={"lat": lat, "lon": lon},
  )

  raster = gis_utils.xarray_to_gdal_raster(field)

  x_min, x_res, _, y_max, _, y_res = raster.GetGeoTransform()
  assert (x_min, x_res, y_max, y_res) == (centres[0] - 45.0, 90.0, 67.5, -45.0)
  band = raster.GetRasterBand(1)
  assert np.isnan(band.GetNoDataValue())
  expected = np.array([45.0, 0.0, -45.0])[:, None] * 1000.0 + np.array(centres)[None, :]
  np.testing.assert_array_equal(band.ReadAsArray(), expected)
  assert raster.GetDriver().ShortName == "MEM" and band.DataType == gdal.GDT_Float32
