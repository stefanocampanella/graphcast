import numpy as np
import pandas as pd
import pytest
import xarray as xr

from graphcast.dataloader import climatology_index


def _index(dates) -> np.ndarray:
  return climatology_index(xr.DataArray(pd.to_datetime(dates), dims="time"))


@pytest.mark.parametrize(
  ("date", "expected"),
  [
    ("2019-01-01", 0),
    ("2019-02-27", 57),
    ("2019-02-28", 58),
    ("2019-03-01", 59),
    ("2019-12-31", 364),
    ("2020-01-01", 0),
    ("2020-02-27", 57),
    ("2020-02-28", 58),
    ("2020-02-29", 58),
    ("2020-03-01", 59),
    ("2020-12-31", 364),
  ],
)
def test_climatology_index(date, expected):
  assert _index([date]).tolist() == [expected]


@pytest.mark.parametrize("year", [2019, 2020, 2000, 2100])
def test_climatology_index_covers_the_climatology(year):
  index = _index(pd.date_range(f"{year}-01-01", f"{year}-12-31", freq="D"))
  np.testing.assert_array_equal(np.unique(index), np.arange(365))
  assert np.all(np.diff(index) >= 0)
