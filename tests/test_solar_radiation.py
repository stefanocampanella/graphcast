# Copyright 2023 DeepMind Technologies Limited.
# Copyright 2026 Stefano Campanella.
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
#
# Ported from the absltest tests of the original GraphCast code to pytest.
import logging
import timeit
from collections.abc import Sequence

import numpy as np
import pandas as pd
import pytest
import xarray as xa

from graphcast import solar_radiation

logger = logging.getLogger(__name__)


def _get_grid_lat_lon_coords(num_lat: int, num_lon: int) -> tuple[np.ndarray, np.ndarray]:
  """Generates a linear latitude-longitude grid of the given size.

  Args:
    num_lat: Size of the latitude dimension of the grid.
    num_lon: Size of the longitude dimension of the grid.

  Returns:
    A tuple `(lat, lon)` containing 1D arrays with the latitude and longitude
    coordinates in degrees of the generated grid.
  """
  lat = np.linspace(-90.0, 90.0, num=num_lat, endpoint=True)
  lon = np.linspace(0.0, 360.0, num=num_lon, endpoint=False)
  return lat, lon


@pytest.fixture(autouse=True)
def _seed():
  np.random.seed(0)


def test_missing_dim_raises_value_error():
  """Inputs without the lat or lon dimensions are rejected."""
  data = xa.DataArray(
    np.random.randn(2, 2),
    coords=[np.array([0.1, 0.2]), np.array([0.0, 0.5])],
    dims=["lon", "x"],
  )
  with pytest.raises(ValueError, match=r".* dimensions are missing in `data_array_like`."):
    solar_radiation.get_toa_incident_solar_radiation_for_xarray(data, integration_period="1h", num_integration_bins=360)


def test_missing_coordinate_raises_value_error():
  """Inputs without the `datetime` coordinate are rejected."""
  data = xa.Dataset(
    data_vars={"var1": (["x", "lat", "lon"], np.random.randn(2, 3, 2))},
    coords={
      "lat": np.array([0.0, 0.1, 0.2]),
      "lon": np.array([0.0, 0.5]),
    },
  )
  with pytest.raises(ValueError, match=r".* coordinates are missing in `data_array_like`."):
    solar_radiation.get_toa_incident_solar_radiation_for_xarray(data, integration_period="1h", num_integration_bins=360)


def test_shape_multiple_timestamps():
  """With a time dimension, the radiation has the dimensions and shape of the input's (time, lat, lon)."""
  data = xa.Dataset(
    data_vars={"var1": (["time", "lat", "lon"], np.random.randn(2, 4, 2))},
    coords={
      "lat": np.array([0.0, 0.1, 0.2, 0.3]),
      "lon": np.array([0.0, 0.5]),
      "time": np.array([100, 200], dtype="timedelta64[s]"),
      "datetime": xa.Variable("time", np.array([10, 20], dtype="datetime64[D]")),
    },
  )

  actual = solar_radiation.get_toa_incident_solar_radiation_for_xarray(
    data, integration_period="1h", num_integration_bins=2
  )

  assert actual.dims == ("time", "lat", "lon")
  assert actual.shape == (2, 4, 2)


def test_shape_single_timestamp():
  """With a scalar `datetime`, the radiation has the input's (lat, lon) dimensions and shape."""
  data = xa.Dataset(
    data_vars={"var1": (["lat", "lon"], np.random.randn(4, 2))},
    coords={
      "lat": np.array([0.0, 0.1, 0.2, 0.3]),
      "lon": np.array([0.0, 0.5]),
      "datetime": np.datetime64(10, "D"),
    },
  )

  actual = solar_radiation.get_toa_incident_solar_radiation_for_xarray(
    data, integration_period="1h", num_integration_bins=2
  )

  assert actual.dims == ("lat", "lon")
  assert actual.shape == (4, 2)


@pytest.mark.parametrize(
  ("periods", "repeats", "use_jit"),
  [
    pytest.param(1, 3, True, id="one_timestamp_jitted"),
    pytest.param(1, 3, False, id="one_timestamp_non_jitted"),
    pytest.param(10, 1, False, id="ten_timestamps_non_jitted"),
  ],
)
def test_full_spatial_resolution(periods: int, repeats: int, use_jit: bool):
  """Benchmark at 0.25° resolution, jitted or not: the computation runs and its times are logged (not asserted)."""
  timestamps = pd.date_range(start="2023-09-25", periods=periods, freq="6h")
  # Generate a linear grid with 0.25 degrees resolution similar to ERA5.
  lat, lon = _get_grid_lat_lon_coords(num_lat=721, num_lon=1440)

  def benchmark() -> None:
    solar_radiation.get_toa_incident_solar_radiation(
      timestamps,
      lat,
      lon,
      integration_period="1h",
      num_integration_bins=360,
      use_jit=use_jit,
    ).block_until_ready()

  results = timeit.repeat(benchmark, repeat=repeats, number=1)

  logger.info(
    "Times to compute `tisr` for input of shape `%d, %d, %d` (seconds): %s",
    len(timestamps),
    len(lat),
    len(lon),
    np.array2string(np.array(results), precision=1),
  )


@pytest.mark.parametrize(
  ("loader", "expected_tsi"),
  [
    pytest.param(solar_radiation.reference_tsi_data, np.array([1361.0]), id="reference_tsi_data"),
    pytest.param(
      solar_radiation.era5_tsi_data,
      np.array([1360.9440]),  # 0.9965 * 1365.7240
      id="era5_tsi_data",
    ),
  ],
)
def test_mid_2020_lookup(loader: solar_radiation.TsiDataLoader, expected_tsi: np.ndarray):
  """The bundled total solar irradiance tables give their tabulated value in mid-2020 (ERA5's is scaled by 0.9965)."""
  tsi_data = loader()

  tsi = solar_radiation.get_tsi([np.datetime64("2020-07-02T00:00:00")], tsi_data)

  np.testing.assert_allclose(expected_tsi, tsi)


@pytest.mark.parametrize(
  ("timestamps", "expected_tsi"),
  [
    pytest.param([np.datetime64("2020-01-01T00:00:00")], np.array([1000.0]), id="beginning_2020_left_boundary"),
    pytest.param([np.datetime64("2020-07-02T00:00:00")], np.array([1000.0]), id="mid_2020_exact"),
    pytest.param([np.datetime64("2021-01-01T00:00:00")], np.array([1150.0]), id="beginning_2021_interpolated"),
    pytest.param([np.datetime64("2021-07-02T12:00:00")], np.array([1300.0]), id="mid_2021_lookup"),
    pytest.param([np.datetime64("2022-01-01T00:00:00")], np.array([1250.0]), id="beginning_2022_interpolated"),
    pytest.param([np.datetime64("2022-07-02T12:00:00")], np.array([1200.0]), id="mid_2022_lookup"),
    pytest.param([np.datetime64("2023-01-01T00:00:00")], np.array([1200.0]), id="beginning_2023_right_boundary"),
  ],
)
def test_interpolation(timestamps: Sequence[np.datetime64], expected_tsi: np.ndarray):
  """Total solar irradiance is interpolated linearly between yearly values centred mid-year, and held constant before
  the first and after the last."""
  tsi_data = xa.DataArray(
    np.array([1000.0, 1300.0, 1200.0]),
    dims=["time"],
    coords={"time": np.array([2020.5, 2021.5, 2022.5])},
  )

  tsi = solar_radiation.get_tsi(timestamps, tsi_data)

  np.testing.assert_allclose(expected_tsi, tsi)
