# Copyright 2026 Stefano Campanella.
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
"""Tests of the forecast scores (validation_utils.py) on hand-made fields with known scores."""

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from graphcast import validation_utils

# A regular global grid (latitudes from -90 + 45/2 to 90 - 45/2), as the latitude weights expect.
LAT = np.linspace(-67.5, 67.5, 4)
LON = np.arange(0.0, 360.0, 90.0)
LEVEL = np.array([0, 1])
TIME = pd.date_range("2020-01-30", periods=3, freq="D").values


def _field(values: np.ndarray) -> xr.Dataset:
  """A dataset with one variable over (batch, time, level, lat, lon)."""
  return xr.Dataset(
    {"t": (("batch", "time", "level", "lat", "lon"), values)},
    coords={"time": TIME, "level": LEVEL, "lat": LAT, "lon": LON},
  )


@pytest.fixture
def truth() -> xr.Dataset:
  return _field(np.random.default_rng(0).normal(size=(2, TIME.size, LEVEL.size, LAT.size, LON.size)))


def test_spatial_average_weights_latitudes_by_area():
  """The average over levels, latitudes and longitudes weighs latitudes by cos(latitude) (area), per batch and time."""
  values = np.broadcast_to(LAT[None, None, None, :, None], (2, TIME.size, LEVEL.size, LAT.size, LON.size))
  average = validation_utils.spatial_average(_field(np.abs(values)), skipna=False)

  weights = np.cos(np.deg2rad(LAT))
  expected = (np.abs(LAT) * weights).sum() / weights.sum()
  assert average["t"].dims == ("batch", "time")
  np.testing.assert_allclose(average["t"].values, expected)


def test_rmse_and_bias_of_a_shifted_forecast(truth):
  """A forecast off by +2 everywhere has RMSE 2 and bias +2 (forecast minus truth)."""
  forecast = truth + 2.0
  np.testing.assert_allclose(validation_utils.get_rmse(forecast, truth)["t"].values, 2.0)
  np.testing.assert_allclose(validation_utils.get_bias(forecast, truth)["t"].values, 2.0)


def test_normalized_rmse(truth):
  """The RMSE relative to that of the climatology: 2 / 4 when the climatology is off by 4."""
  nrmse = validation_utils.get_nrmse(truth + 2.0, truth, climatology_ref=truth + 4.0)
  np.testing.assert_allclose(nrmse["t"].values, 0.5)


@pytest.mark.parametrize(("sign", "expected"), [(1.0, 1.0), (-1.0, -1.0), (2.0, 1.0)])
def test_anomaly_correlation(truth, sign, expected):
  """The ACC is 1 for anomalies proportional to the true ones, -1 for opposite ones, whatever their amplitude."""
  climatology = xr.zeros_like(truth) + 0.5
  forecast = climatology + sign * (truth - climatology)
  np.testing.assert_allclose(validation_utils.get_acc(forecast, truth, climatology)["t"].values, expected)


def test_skipna_ignores_missing_points(truth):
  """With skipna, missing points (e.g. land) are left out of the averages."""
  forecast = (truth + 2.0).where(truth.lat > 0)
  np.testing.assert_allclose(validation_utils.get_rmse(forecast, truth, skipna=True)["t"].values, 2.0)
  assert np.isnan(validation_utils.get_rmse(forecast, truth, skipna=False)["t"].values).all()


def test_climatology_reference_of_the_target_days(truth):
  """The climatology of each target's day of year, on the target times and levels."""
  day_of_year = np.arange(1, 367)
  climatology = xr.Dataset(
    {"t": (("dayofyear", "level", "lat", "lon"), np.broadcast_to(day_of_year[:, None, None, None], (366, 3, 4, 4)))},
    coords={"dayofyear": day_of_year, "level": [0, 1, 2], "lat": LAT, "lon": LON},
  )
  reference = validation_utils.get_climatology_ref(climatology, truth)

  np.testing.assert_array_equal(reference["time"].values, TIME)
  np.testing.assert_array_equal(reference["level"].values, LEVEL)
  assert "dayofyear" not in reference.coords
  np.testing.assert_array_equal(reference["t"].isel(level=0, lat=0, lon=0).values, [30, 31, 32])
