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
"""Tests of the mesh size fields (geospatial_mesh_utils.py) on synthetic rasters."""

import numpy as np
import pytest
import xarray as xr

from graphcast.geospatial_mesh_utils import (
  BathymetryField,
  BathymetryHessianField,
  RasterField,
  UniformField,
)
from graphcast.gis_utils import equirectangular_crs, stereographic_crs

LAT = np.arange(-60.0, 61.0, 30.0)
LON = np.arange(0.0, 360.0, 30.0)
SIZE_MIN, SIZE_MAX = 10_000.0, 50_000.0


def _grid(values: np.ndarray, name: str = "depth") -> xr.DataArray:
  return xr.DataArray(values, dims=("lat", "lon"), coords={"lat": LAT, "lon": LON}, name=name)


def _grid_points() -> np.ndarray:
  """(longitude, latitude) of the grid points away from the dateline, in degrees."""
  lon, lat = np.meshgrid(LON[1:-1], LAT, indexing="xy")
  return np.stack([np.where(lon > 180, lon - 360, lon).ravel(), lat.ravel()], axis=-1)


def _expected(values: np.ndarray, low: float, high: float) -> np.ndarray:
  return SIZE_MIN + (SIZE_MAX - SIZE_MIN) * (np.clip(values, low, high) - low) / (high - low)


def test_uniform_field_on_the_stereographic_plane():
  """The size in 3D is uniform; on the stereographic plane it grows away from the projection's pole, by
  (4R² + x² + y²) / 4R²."""
  field = UniformField(20_000.0)
  radius_squared = stereographic_crs.GetSemiMajor() * stereographic_crs.GetSemiMinor()
  points = np.array([[0.0, 0.0], [3e6, 4e6]])
  np.testing.assert_allclose(field.mesh_size_3d(points, stereographic_crs), 20_000.0)
  np.testing.assert_allclose(
    field(points, stereographic_crs), [20_000.0, 20_000.0 * (1 + 25e12 / (4 * radius_squared))]
  )


@pytest.mark.parametrize(("q_low", "q_high"), [(0.0, 1.0), (0.25, 0.75)])
def test_raster_field(q_low, q_high):
  """The size grows linearly with the raster value between its extrema (or quantiles, beyond which it is clipped),
  from size_min to size_max; missing values count as the maximum."""
  values = np.tile((LON - 180.0) ** 2, (LAT.size, 1))
  values[0, 3] = np.nan
  field = RasterField(_grid(values), SIZE_MIN, SIZE_MAX, q_low=q_low, q_high=q_high)

  low, high = np.nanquantile(values, q_low), np.nanquantile(values, q_high)
  points = _grid_points()
  expected = _expected(np.tile((LON[1:-1] - 180.0) ** 2, LAT.size), low, high)
  expected[3 - 1] = SIZE_MAX
  np.testing.assert_allclose(field.mesh_size_3d(points, equirectangular_crs), expected, rtol=1e-6)


@pytest.fixture
def bathymetry_path(tmp_path) -> str:
  depth = np.tile(np.linspace(0.0, 4000.0, LON.size), (LAT.size, 1))
  depth[:, 0] = np.nan
  path = tmp_path / "bathymetry.zarr"
  _grid(depth).to_dataset().to_zarr(path)
  return str(path)


def test_bathymetry_field(bathymetry_path):
  """The size grows with the square root of the depth (deeper water, larger cells), between the shallowest and the
  deepest point with data (the first column has none)."""
  field = BathymetryField(bathymetry_path, "depth", SIZE_MIN, SIZE_MAX)
  points = _grid_points()
  root_depth = np.sqrt(np.linspace(0.0, 1.0, LON.size))
  expected = _expected(np.tile(root_depth[1:-1], LAT.size), root_depth[1], 1.0)
  np.testing.assert_allclose(field.mesh_size_3d(points, equirectangular_crs), expected, rtol=1e-5)


def test_bathymetry_field_rejects_negative_depths(tmp_path):
  path = tmp_path / "bathymetry.zarr"
  _grid(np.tile(np.linspace(-10.0, 4000.0, LON.size), (LAT.size, 1))).to_dataset().to_zarr(path)
  with pytest.raises(AssertionError, match="negative depth"):
    BathymetryField(str(path), "depth", SIZE_MIN, SIZE_MAX)


def test_norm_of_hessian_of_a_quadratic():
  """For z = a lat², the Hessian norm is |2a| (no smoothing, a spline that reproduces quadratics), in degrees."""
  lat = np.arange(-60.0, 61.0, 5.0)
  lon = np.arange(0.0, 360.0, 5.0)
  depth = xr.DataArray(
    0.01 * np.tile(lat[:, None] ** 2, (1, lon.size)), dims=("lat", "lon"), coords={"lat": lat, "lon": lon}
  )
  norm = BathymetryHessianField.norm_of_hessian(depth, filter_kwargs={"sigma": 0.0})
  np.testing.assert_allclose(norm.isel(lat=slice(2, -2)).values, 0.02, rtol=1e-6)


def test_bathymetry_hessian_field(bathymetry_path):
  """Sizes from the inverse square root of the Hessian norm lie between size_min and size_max."""
  field = BathymetryHessianField(bathymetry_path, "depth", SIZE_MIN, SIZE_MAX)
  sizes = field.mesh_size_3d(_grid_points(), equirectangular_crs)
  assert np.all((sizes >= SIZE_MIN) & (sizes <= SIZE_MAX))
