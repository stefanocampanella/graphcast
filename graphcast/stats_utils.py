import warnings

import xarray as xr
import pandas as pd
from typing import Literal


def valid_datetime_index(idx: pd.DatetimeIndex) -> bool:
  # Check that the following implementation works for both DateTimeIndex and CFTimeIndex
  if idx.has_duplicates:
    dups = idx[idx.duplicated()]
    dup_values = pd.DatetimeIndex(dups.unique())
    preview = ", ".join(str(ts) for ts in dup_values[:5])
    more = "" if len(dup_values) <= 5 else f" and {len(dup_values) - 5} more"
    warnings.warn(f"Duplicate timestamps found in time coordinate: {preview}{more}")
    return False

  if len(idx) > 0:
    # noinspection PyTypeChecker
    expected = pd.date_range(start=idx[0], periods=len(idx), freq="D", tz=getattr(idx, "tz", None))
    if not idx.equals(expected):
      # Report missing or irregular timestamps for easier debugging
      # Compute missing by comparing against the sorted unique expected sequence
      sorted_idx = idx.sort_values()
      expected_full = pd.date_range(start=sorted_idx[0], end=sorted_idx[-1], freq="D", tz=getattr(sorted_idx, "tz", None))
      missing = expected_full.difference(sorted_idx)
      preview = ", ".join(str(ts) for ts in missing[:5])
      more = "" if len(missing) <= 5 else f" and {len(missing) - 5} more"
      warnings.warn(f"Missing or irregular dates detected: {preview}{more}")
      return False

  return True


def valid_cftime_index(idx: xr.CFTimeIndex) -> bool:
  # Equivalent checks for xarray.CFTimeIndex (cftime-based calendars)
  # Duplicates check
  if idx.has_duplicates:
    dups = idx[idx.duplicated()]
    dup_values = dups.unique()  # CFTimeIndex of unique duplicate timestamps
    preview = ", ".join(str(ts) for ts in dup_values[:5])
    more = "" if len(dup_values) <= 5 else f" and {len(dup_values) - 5} more"
    warnings.warn(f"Duplicate timestamps found in time coordinate: {preview}{more}")
    return False

  # Regularity check (daily frequency across CF calendars)
  if len(idx) > 0:
    calendar = getattr(idx, "calendar", None)
    # Build expected daily sequence with same calendar
    expected = xr.cftime_range(start=idx[0], periods=len(idx), freq="D", calendar=calendar)
    if not idx.equals(expected):
      # Compute missing days between min and max dates for helpful diagnostics
      sorted_idx = idx.sort_values()
      expected_full = xr.cftime_range(start=sorted_idx[0], end=sorted_idx[-1], freq="D", calendar=calendar)
      missing = expected_full.difference(sorted_idx)
      preview = ", ".join(str(ts) for ts in missing[:5])
      more = "" if len(missing) <= 5 else f" and {len(missing) - 5} more"
      warnings.warn(f"Missing or irregular dates detected: {preview}{more}")
      return False

  return True


def valid_time_coordinate(dataset: xr.Dataset, time_dim: str = "time") -> bool:
  idx = dataset[time_dim].to_index()
  if isinstance(idx, pd.DatetimeIndex):
    passed = valid_datetime_index(idx)
  elif isinstance(idx, xr.CFTimeIndex):
    passed = valid_cftime_index(idx)
  else:
    raise ValueError(f"Unexpected index type: {type(idx).__name__}")
  return passed


def compute_mean(dataset: xr.Dataset, time_dim: str = "time", **kwargs) -> xr.Dataset:
  return dataset.mean(dim=time_dim, **kwargs)


def compute_std(dataset: xr.Dataset, time_dim: str = "time", **kwargs) -> xr.Dataset:
  return dataset.std(dim=time_dim, **kwargs)


def compute_diff_std(dataset: xr.Dataset, time_dim: str = "time", **kwargs) -> xr.Dataset:
  # Given a sequence {x_i}_{i = 1, ..., N} the mean of the diffs {x_i - x_{i-1}} is proportional to the sum of a
  # telescopic series and equal to (x_N - x_1) / N, which becomes negligible for large N.
  # Also, Graphcast computes the increment between the present and next system state rescaled by diff_std.
  # This accounts to standardizing the targets. Whatever the rationale, one can reasonably approximate here the mean with zero.
  diff = dataset.diff(dim=time_dim)
  diff_var= (diff * diff).mean(dim=time_dim, **kwargs)
  diff_std = xr.ufuncs.sqrt(diff_var)
  return diff_std


def compute_climatology(dataset: xr.Dataset, time_dim: str = "time", climatology_dim: str = "dayofyear",
                        calendar: Literal["365_day", "366_day", "360_day"] = "365_day", skipna=False, **kwargs) -> xr.Dataset:
  # The current implementation assumes that, among other things, the time coordinate is daily, contiguous and without duplicates.
  if not valid_time_coordinate(dataset, time_dim):
    raise ValueError("Dataset has invalid time coordinate.")

  # Here we handle leap years.
  # See: https://github.com/pydata/xarray/issues/1844#issuecomment-417855365
  dataset = dataset.convert_calendar(calendar)
  if calendar == "360_day":
    n_days = 360
  elif calendar == "365_day":
    n_days = 365
  elif calendar == "366_day":
    n_days = 366
  else:
    raise ValueError(f"Unsupported calendar: {calendar}")

  size = dataset.sizes[time_dim]
  start = 0
  end = min(n_days, size)
  counter = 1
  dataset = dataset.assign_coords({climatology_dim: dataset[time_dim].dt.dayofyear})
  avg = dataset.isel({time_dim: slice(start, end)})
  avg = avg.drop(time_dim)
  avg = avg.swap_dims({time_dim: climatology_dim})
  while True:
    start = end
    end = min(start + n_days, size)
    if start < size:
      counter += 1
      value = dataset.isel({time_dim: slice(start, end)})
      value = value.drop(time_dim)
      value = value.swap_dims({time_dim: climatology_dim})
      if skipna:
        value = value.where(value.notnull(), avg)
      avg += (value - avg) / float(counter)
    else:
      break
  avg = avg.chunk({climatology_dim: 1})
  return avg

Stats = Literal["climatology", "mean", "std", "diff_std"]
StatsRegistry = {'climatology': compute_climatology, 'mean': compute_mean, 'std': compute_std, 'diff_std': compute_diff_std}
