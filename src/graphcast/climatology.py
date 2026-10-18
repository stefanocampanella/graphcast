"""Calendar-aware alignment of daily climatologies."""

import numpy as np
import xarray as xr


def align_climatology(
    climatology: xr.Dataset | xr.DataArray,
    timeseries: xr.Dataset | xr.DataArray,
    time_dim: str = "time",
) -> xr.Dataset | xr.DataArray:
  """Align a 365-day climatology with the original timeseries timestamps.

  Args:
    climatology: Daily no-leap climatology with integer coordinates 0, ..., 364
      along ``time_dim``.
    timeseries: Original data with a one-dimensional datetime coordinate along
      ``time_dim``. Gregorian, Julian, no-leap and all-leap calendars are supported.
    time_dim: Name of the time dimension in both inputs.

  Returns:
    Climatology values selected by month and day at every original timestamp,
    including repeated or subdaily timestamps. February 29 is the mean of
    February 28 and March 1. Other missing values are left unchanged, and
    Dask-backed climatology data remain lazy. Neither input is modified.

  Raises:
    ValueError: The climatology is not indexed by 0, ..., 364, or the target
      timestamps are missing, not one-dimensional datetimes, or use an
      unsupported calendar (such as ``360_day``).
  """
  if (
      climatology.sizes.get(time_dim) != 365
      or time_dim not in climatology.coords
      or not np.issubdtype(climatology[time_dim].dtype, np.integer)
      or not np.array_equal(climatology[time_dim].values, np.arange(365))
  ):
    raise ValueError("Climatology must have integer time coordinates 0, ..., 364.")

  if time_dim not in timeseries.coords or timeseries[time_dim].dims != (time_dim,):
    raise ValueError("Timeseries must have a one-dimensional datetime coordinate.")
  time = timeseries[time_dim]
  try:
    calendar = time.dt.calendar
  except (AttributeError, TypeError, ValueError) as error:
    raise ValueError("Timeseries must have a datetime coordinate.") from error
  if calendar not in {
    "standard", "gregorian", "proleptic_gregorian", "julian",
    "noleap", "365_day", "all_leap", "366_day",
  }:
    raise ValueError(f"Unsupported timeseries calendar: {calendar}.")
  if bool(time.isnull().any()):
    raise ValueError("Timeseries timestamps must not be missing.")

  feb28 = climatology.isel({time_dim: slice(58, 59)})
  mar1 = climatology.isel({time_dim: slice(59, 60)}).assign_coords({time_dim: [58]})
  if isinstance(climatology, xr.Dataset):
    feb29 = feb28.copy(deep=False)
    for name, variable in feb28.data_vars.items():
      if time_dim in variable.dims:
        feb29[name] = 0.5 * (variable + mar1[name])
    concat_kwargs = {"data_vars": "minimal"}
  else:
    feb29 = 0.5 * (feb28 + mar1)
    concat_kwargs = {}
  leap_climatology = xr.concat(
    [
      climatology.isel({time_dim: slice(0, 59)}),
      feb29,
      climatology.isel({time_dim: slice(59, None)}),
    ],
    dim=time_dim,
    **concat_kwargs,
  ).assign_coords({time_dim: np.arange(366)})

  month_offsets = np.array([0, 31, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335])
  index = month_offsets[time.dt.month.values - 1] + time.dt.day.values - 1
  index = xr.DataArray(index, dims=(time_dim,))
  return leap_climatology.isel({time_dim: index}).assign_coords({time_dim: time})
