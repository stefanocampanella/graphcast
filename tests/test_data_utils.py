# Copyright 2023 DeepMind Technologies Limited.
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
#
# Ported from the absltest tests of the original GraphCast code to pytest and updated for this fork.
"""Tests for `data_utils.py`."""

import datetime

import numpy as np
import pytest
import xarray as xa

from graphcast import data_utils


@pytest.fixture(autouse=True)
def _seed():
  # Fix the seed for reproducibility.
  np.random.seed(0)


def test_year_progress_is_zero_at_year_start_or_end():
  """Year progress is 0 at the epoch and at whole multiples of the average year length."""
  year_progress = data_utils.get_year_progress(
    np.array(
      [
        0,
        data_utils.AVG_SEC_PER_YEAR,
        data_utils.AVG_SEC_PER_YEAR * 42,  # 42 years.
      ]
    )
  )
  np.testing.assert_array_equal(year_progress, np.zeros(year_progress.shape))


def test_year_progress_is_almost_one_before_year_ends(subtests):
  """One second before a year ends, year progress is just below 1 (it wraps to 0, never reaching 1)."""
  year_progress = data_utils.get_year_progress(
    np.array(
      [
        data_utils.AVG_SEC_PER_YEAR - 1,
        (data_utils.AVG_SEC_PER_YEAR - 1) * 42,  # ~42 years
      ]
    )
  )
  with subtests.test("Year progress values are close to 1"):
    assert np.all(year_progress > 0.999)
  with subtests.test("Year progress values != 1"):
    assert np.all(year_progress < 1.0)


def test_day_progress_computes_for_all_times_and_longitudes(subtests):
  """Day progress has one value per time and longitude (shape `[time, lon]`)."""
  times = np.random.randint(low=0, high=10**10, size=10)
  longitudes = np.arange(0, 360.0, 1.0)
  day_progress = data_utils.get_day_progress(times, longitudes)
  with subtests.test("Day progress is computed for all times and longinutes"):
    assert tuple(day_progress.shape) == (len(times), len(longitudes))


@pytest.mark.parametrize(
  ("year", "month", "day", "hour", "minute", "second"),
  [
    pytest.param(1988, 11, 7, 2, 45, 34, id="random_date_1"),
    pytest.param(2022, 3, 12, 7, 1, 0, id="random_date_2"),
  ],
)
def test_day_progress_is_in_between_zero_and_one(subtests, year, month, day, hour, minute, second):
  """Day progress at any date and longitude lies in [0, 1)."""
  # Datetime from a timestamp.
  dt = datetime.datetime(year, month, day, hour, minute, second)
  # Epoch time.
  epoch_time = datetime.datetime(1970, 1, 1)
  # Seconds since epoch.
  seconds_since_epoch = np.array([(dt - epoch_time).total_seconds()])

  # Longitudes with 1 degree resolution.
  longitudes = np.arange(0, 360.0, 1.0)

  day_progress = data_utils.get_day_progress(seconds_since_epoch, longitudes)
  with subtests.test("Day progress >= 0"):
    assert np.all(day_progress >= 0.0)
  with subtests.test("Day progress < 1"):
    assert np.all(day_progress < 1.0)


def test_day_progress_is_zero_at_day_start_or_end():
  """At longitude 0, day progress is 0 at midnight UTC of any day."""
  day_progress = data_utils.get_day_progress(
    seconds_since_epoch=np.array(
      [
        0,
        data_utils.SEC_PER_DAY,
        data_utils.SEC_PER_DAY * 42,  # 42 days.
      ]
    ),
    longitude=np.array([0.0]),
  )
  np.testing.assert_array_equal(day_progress, np.zeros(day_progress.shape))


def test_day_progress_specific_value():
  """Day progress at longitude 0 is the elapsed fraction of the day (123 s / 86400 s)."""
  day_progress = data_utils.get_day_progress(
    seconds_since_epoch=np.array([123]),
    longitude=np.array([0.0]),
  )
  np.testing.assert_array_almost_equal(day_progress, np.array([[0.00142361]]), decimal=6)


def test_featurize_progress_valid_values_and_dimensions(subtests):
  """Progress features are the progress and its sine and cosine (of 2π × progress), with the requested dimensions."""
  day_progress = np.array([0.0, 0.45, 0.213])
  feature_dimensions = ("time",)
  progress_features = data_utils.featurize_progress(name="day_progress", dims=feature_dimensions, progress=day_progress)
  for feature in progress_features.values():
    with subtests.test(f"Valid dimensions for {feature}"):
      assert tuple(feature.dims) == feature_dimensions

  with subtests.test("Valid values for day_progress"):
    np.testing.assert_array_equal(day_progress, progress_features["day_progress"].values)

  with subtests.test("Valid values for day_progress_sin"):
    np.testing.assert_array_almost_equal(
      np.array([0.0, 0.30901699, 0.97309851]),
      progress_features["day_progress_sin"].values,
      decimal=6,
    )

  with subtests.test("Valid values for day_progress_cos"):
    np.testing.assert_array_almost_equal(
      np.array([1.0, -0.95105652, 0.23038943]),
      progress_features["day_progress_cos"].values,
      decimal=6,
    )


def test_featurize_progress_invalid_dimensions():
  """Dimensions that do not match the shape of the progress array are rejected."""
  year_progress = np.array([0.0, 0.45, 0.213])
  feature_dimensions = ("time", "longitude")
  with pytest.raises(ValueError):
    data_utils.featurize_progress(name="year_progress", dims=feature_dimensions, progress=year_progress)


def test_add_derived_vars_variables_added(subtests):
  """Year and day progress are added next to the existing variables."""
  data = xa.Dataset(
    data_vars={"var1": (["x", "lon", "datetime"], 8 * np.random.randn(2, 2, 3))},
    coords={
      "lon": np.array([0.0, 0.5]),
      "datetime": np.array(
        [
          datetime.datetime(2021, 1, 1),
          datetime.datetime(2023, 1, 1),
          datetime.datetime(2023, 1, 3),
        ]
      ),
    },
  )
  data_utils.add_derived_vars(data)
  all_variables = set(data.variables)

  with subtests.test("Original value was not removed"):
    assert "var1" in all_variables
  with subtests.test("Year progress feature was added"):
    assert data_utils.YEAR_PROGRESS in all_variables
  with subtests.test("Day progress feature was added"):
    assert data_utils.DAY_PROGRESS in all_variables


def test_add_derived_vars_existing_vars_not_overridden(subtests):
  """Progress variables already in the dataset are kept, not recomputed."""
  dims = ["x", "lon", "datetime"]
  data = xa.Dataset(
    data_vars={
      "var1": (dims, 8 * np.random.randn(2, 2, 3)),
      data_utils.YEAR_PROGRESS: (dims, np.full((2, 2, 3), 0.111)),
      data_utils.DAY_PROGRESS: (dims, np.full((2, 2, 3), 0.222)),
    },
    coords={
      "lon": np.array([0.0, 0.5]),
      "datetime": np.array(
        [
          datetime.datetime(2021, 1, 1),
          datetime.datetime(2023, 1, 1),
          datetime.datetime(2023, 1, 3),
        ]
      ),
    },
  )

  data_utils.add_derived_vars(data)

  with subtests.test("Year progress feature was not overridden"):
    np.testing.assert_allclose(data[data_utils.YEAR_PROGRESS], 0.111)
  with subtests.test("Day progress feature was not overridden"):
    np.testing.assert_allclose(data[data_utils.DAY_PROGRESS], 0.222)


@pytest.mark.parametrize(
  ("coord_name", "missing"),
  [
    pytest.param("lon", "datetime", id="missing_datetime"),
    pytest.param("datetime", "lon", id="missing_lon"),
  ],
)
def test_add_derived_vars_missing_coordinate_raises_value_error(coord_name, missing):
  """Without the `datetime` or `lon` coordinate the progress cannot be computed: the error names the missing one."""
  data = xa.Dataset(
    data_vars={"var1": (["x", coord_name], 8 * np.random.randn(2, 2))},
    coords={coord_name: np.array([0.0, 0.5])},
  )
  with pytest.raises(ValueError, match=missing):
    data_utils.add_derived_vars(data)


def test_add_tisr_var_variable_added():
  """Top-of-atmosphere solar radiation is added for a dataset with lat, lon and a datetime per time."""
  data = xa.Dataset(
    data_vars={"var1": (["time", "lat", "lon"], np.full((2, 2, 2), 8.0))},
    coords={
      "lat": np.array([2.0, 1.0]),
      "lon": np.array([0.0, 0.5]),
      "time": np.array([100, 200], dtype="timedelta64[s]"),
      "datetime": xa.Variable("time", np.array([10, 20], dtype="datetime64[D]")),
    },
  )

  data_utils.add_tisr_var(data)

  assert data_utils._DEFAULT_TISR_NAME in set(data.variables)


def test_add_tisr_var_existing_var_not_overridden():
  """Solar radiation already in the dataset is kept, not recomputed."""
  dims = ["time", "lat", "lon"]
  data = xa.Dataset(
    data_vars={
      "var1": (dims, np.full((2, 2, 2), 8.0)),
      data_utils._DEFAULT_TISR_NAME: (dims, np.full((2, 2, 2), 1200.0)),
    },
    coords={
      "lat": np.array([2.0, 1.0]),
      "lon": np.array([0.0, 0.5]),
      "time": np.array([100, 200], dtype="timedelta64[s]"),
      "datetime": xa.Variable("time", np.array([10, 20], dtype="datetime64[D]")),
    },
  )

  data_utils.add_derived_vars(data)

  np.testing.assert_allclose(data[data_utils._DEFAULT_TISR_NAME], 1200.0)


def test_add_tisr_var_works_with_batch_dim_size_one():
  """A batch dimension of size 1 is accepted."""
  data = xa.Dataset(
    data_vars={
      "var1": (
        ["batch", "time", "lat", "lon"],
        np.full((1, 2, 2, 2), 8.0),
      )
    },
    coords={
      "lat": np.array([2.0, 1.0]),
      "lon": np.array([0.0, 0.5]),
      "time": np.array([100, 200], dtype="timedelta64[s]"),
      "datetime": xa.Variable(("batch", "time"), np.array([[10, 20]], dtype="datetime64[D]")),
    },
  )

  data_utils.add_tisr_var(data)

  assert data_utils._DEFAULT_TISR_NAME in set(data.variables)


def test_add_tisr_var_fails_with_batch_dim_size_greater_than_one():
  """Batches larger than 1 are not supported (the datetimes may differ per batch element) and fail."""
  data = xa.Dataset(
    data_vars={
      "var1": (
        ["batch", "time", "lat", "lon"],
        np.full((2, 2, 2, 2), 8.0),
      )
    },
    coords={
      "lat": np.array([2.0, 1.0]),
      "lon": np.array([0.0, 0.5]),
      "time": np.array([100, 200], dtype="timedelta64[s]"),
      "datetime": xa.Variable(
        ("batch", "time"),
        np.array([[10, 20], [100, 200]], dtype="datetime64[D]"),
      ),
    },
  )

  with pytest.raises(ValueError, match=r"cannot select a dimension"):
    data_utils.add_tisr_var(data)


def test_fix_longitude_wraps_to_the_dateline():
  """Longitudes in [0, 360) become [-180, 180]: the data are rolled to start at the first longitude past 180°."""
  lon = np.array([0.0, 90.0, 180.0, 270.0])
  data = xa.DataArray(np.array([[1.0, 2.0, 3.0, 4.0]]), dims=("lat", "lon"), coords={"lat": [0.0], "lon": lon})

  fixed = data_utils.fix_longitude(data)

  np.testing.assert_array_equal(fixed["lon"].values, [-90.0, 0.0, 90.0, 180.0])
  np.testing.assert_array_equal(fixed.values, [[4.0, 1.0, 2.0, 3.0]])
  # The same values at the same places.
  np.testing.assert_array_equal(fixed.sel(lon=-90.0).values, data.sel(lon=270.0).values)


def test_fix_longitude_of_wrapped_data_is_a_no_op():
  """Data already in [-180, 180] are returned as they are (np.argwhere found no longitude past 180°: IndexError)."""
  lon = np.array([-90.0, 0.0, 90.0])
  data = xa.DataArray(np.array([[1.0, 2.0, 3.0]]), dims=("lat", "lon"), coords={"lat": [0.0], "lon": lon})
  xa.testing.assert_identical(data_utils.fix_longitude(data), data)
