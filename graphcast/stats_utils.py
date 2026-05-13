from typing import Literal

import numpy as np
import xarray as xr

from graphcast.dataset_utils import valid_time_coordinate


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
  diff_var = (diff * diff).mean(dim=time_dim, **kwargs)
  diff_std = xr.ufuncs.sqrt(diff_var)
  return diff_std


def compute_climatology(
  dataset: xr.Dataset,
  time_dim: str = "time",
  climatology_dim: str = "dayofyear",
  calendar: Literal["365_day", "366_day", "360_day"] = "365_day",
  skipna=False,
  **kwargs,
) -> xr.Dataset:
  # The current implementation assumes that, among other things, the time coordinate is daily, contiguous and without
  # duplicates. Finally, that it contains a whole number of years. Incomplete years data is considered as missing.
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
  avg = avg.reindex({climatology_dim: 1 + np.arange(n_days, dtype=int)}, copy=False)
  while True:
    start = end
    end = min(start + n_days, size)
    if start < size:
      counter += 1
      value = dataset.isel({time_dim: slice(start, end)})
      value = value.drop(time_dim)
      value = value.swap_dims({time_dim: climatology_dim})
      if skipna:
        value = value.reindex({climatology_dim: 1 + np.arange(n_days, dtype=int)}, copy=False)
        value = value.where(value.notnull(), avg)
      avg += (value - avg) / float(counter)
    else:
      break
  avg = avg.chunk({climatology_dim: 1})
  return avg


Stats = Literal["climatology", "mean", "std", "diff_std"]
StatsRegistry = {
  "climatology": compute_climatology,
  "mean": compute_mean,
  "std": compute_std,
  "diff_std": compute_diff_std,
}
