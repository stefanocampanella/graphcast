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

import datetime
from dataclasses import dataclass
from typing import Any, Dict, Literal, Sequence
from collections.abc import Iterable
import cdsapi

import numpy as np
import pandas as pd
import xarray as xr
import xarray_regrid

import logging


InterpolationMethods = Literal['linear', 'nearest', 'zero', 'slinear', 'quadratic', 'cubic', 'polynomial', 'barycentric', 'krogh', 'pchip', 'spline', 'akima', 'makima']
PreprocessingSteps = Literal['isel', 'chunks', 'regrid', 'interpolate', 'resample']

@dataclass
class DateInterval:
  """Low level (for internal use) representation of a date interval, which can be iterated over with arbitrary step size.

  When iterating over a DateInterval object, extrema (`start` and `stop`) are included.

  Attributes:
    start (datetime.datetime): The start date of the interval.
    stop (datetime.datetime): The stop date of the interval.
    step (datetime.timedelta): The step size used when iterating over the interval.
  """
  start: datetime.datetime
  stop: datetime.datetime
  step: datetime.timedelta = datetime.timedelta(days=1)

  def __iter__(self):

    def _date_iterator():
      current = self.start
      while current < self.stop:
        yield current
        current += self.step
      yield self.stop

    return _date_iterator()

  def __repr__(self):
    return f"from {self.start.isoformat()} to {self.stop.isoformat()}"

# DateInterval`s include the extrema (as open_dataset do). DateIntervalsRange uses Python convention (right open).
# The union of all intervals in DateIntervalsRange(start, stop, delta) == [start, stop) == [start, stop - delta]
class DateIntervalsRange(Iterable):
  """Iterable object that yields `DateInterval`s.

  Note:
    It uses Python convention (right open), such that
    the union of all intervals in DateIntervalsRange(start, stop, delta) == [start, stop) == [start, stop - delta]

  Args:
    start (datetime.datetime): The start date of the interval.
    stop (datetime.datetime): The stop date of the interval.
    step (str | datetime.timedelta, optional): Size of the sub-intervals. Defaults to 1 week.
    delta (datetime.timedelta, optional): The step size used when iterating over the interval. Defaults to 1 day.
  """
  def __init__(self,
               start: datetime.datetime | str,
               stop: datetime.datetime | str,
               step: datetime.timedelta | str | None = None,
               delta: datetime.timedelta | str | None = None):

    def may_parse_datetime(date: datetime.datetime | str) -> datetime.datetime:
      if isinstance(date, datetime.datetime):
        return date
      elif isinstance(date, str):
        return datetime.datetime.fromisoformat(date)

    def may_parse_timedelta(delta: datetime.timedelta | str) -> datetime.timedelta:
      if isinstance(delta, datetime.timedelta):
        return delta
      elif isinstance(delta, str):
        return pd.Timedelta(delta).to_pytimedelta()

    self.start = may_parse_datetime(start)
    self.stop = may_parse_datetime(stop)
    if step is not None:
      self.step = may_parse_timedelta(step)
    else:
      self.step = datetime.timedelta(weeks=1)
    if delta is not None:
      self.delta = may_parse_timedelta(delta)
    else:
      self.delta = datetime.timedelta(days=1)

  def __iter__(self):

    def _date_iterator() -> DateInterval:

      current = self.start
      next = min(current + self.step, self.stop)
      while current < self.stop:
        yield DateInterval(current, next - self.delta, self.delta)
        current = next
        next = min(current + self.step, self.stop)

    return _date_iterator()


class Postprocess:

  def __init__(self, configs: Dict[str, Any], steps: Sequence[PreprocessingSteps] | None = None):

    self.configs = configs
    if steps is None:
      self.steps = ['isel', 'chunks', 'regrid', 'interpolate', 'resample', 'round', 'fillna', 'astype']
    else:
      self.steps = steps

  def __call__(self, ds: xr.Dataset) -> xr.Dataset:

    for step in self.steps:
      logging.debug(f"Processing step {step}")
      if conf := self.configs.get(step):
        if step in dir(self):
          ds = getattr(self, step)(ds, **conf)
        else:
          ds = getattr(ds, step)(**conf)
    return ds

  def regrid(self, ds: xr.Dataset, grid=None, **kwargs) -> xr.Dataset:

    if grid is None:
      raise ValueError('Grid must be specified')
    new_grid = xarray_regrid.Grid(**grid)
    target_dataset = new_grid.create_regridding_dataset()
    method = kwargs.get('method', 'conservative')
    target_dataset = target_dataset.assign_coords(latitude=target_dataset.latitude.astype(np.float32),
                                                  longitude=target_dataset.longitude.astype(np.float32))
    regrid_conf = kwargs.get('kwargs', {})
    ds = getattr(ds.regrid, method)(target_dataset, **regrid_conf)
    return ds

  def resample(self, ds: xr.Dataset, **kwargs):

    ds = ds.resample(**kwargs).mean()
    return ds

  def interpolate(self,
                  ds: xr.Dataset,
                  minimum_latitude: float = -90.0,
                  maximum_latitude: float = 90.0,
                  minimum_longitude: float = -180.0,
                  maximum_longitude: float = 180.0,
                  resolution: float = 1.0,
                  method: InterpolationMethods = 'linear',
                  assume_sorted: bool = True,
                  kwargs: Dict[str, Any] | None = None) -> xr.Dataset:

    eps = np.finfo(ds.latitude.dtype).eps
    latitude = np.arange(start=minimum_latitude, stop=maximum_latitude + eps, step=resolution, dtype=np.float32)
    longitude = np.arange(start=minimum_longitude, stop=maximum_longitude, step=resolution, dtype=np.float32)
    ds = ds.interp(latitude=latitude, longitude=longitude, method=method, assume_sorted=assume_sorted, kwargs=kwargs).astype(np.float32)
    return ds


# Credits to the amazing Stefano Piani from OGS
def get_cdsapi_client(
    url: str | None = None, client_logger=None, **kwargs
):
  """Returns a cdsapi.Client instance.

  It also configures the returned client to use a specific logger (if
  submitted)

  Args:
    url (str): the url of the endpoint of the cdsapi. If it is None, it will be
      read from the ~/.cdsapi file (if exists)
    key (str): the key of the cdsapi user account. If it is None, it will be
      read from the ~/.cdsapi file (if exists)
    client_logger (logging.Logger): Logger that the returned client
      will use to print its messages

  Returns:
    cdsapi.Client instance
  """
  if client_logger is not None:

    def debug_callback(*args, **kwargs):
      return client_logger.debug(*args, **kwargs)

    def info_callback(*args, **kwargs):
      return client_logger.info(*args, **kwargs)

    def warning_callback(*args, **kwargs):
      return client_logger.warning(*args, **kwargs)

    def error_callback(*args, **kwargs):
      return client_logger.error(*args, **kwargs)
  else:
    debug_callback = None
    info_callback = None
    warning_callback = None
    error_callback = None

  client_kwargs = {
    "debug_callback": debug_callback,
    "info_callback": info_callback,
    "warning_callback": warning_callback,
    "error_callback": error_callback,
    **kwargs,
  }

  if url is None:
    client_kwargs["url"] = url

  cdsapi_client = cdsapi.Client(**client_kwargs)

  # This is a horrible hack that probably will become not necessary in the
  # next version of cdsapi. It removes the "logging decorator", which is a
  # context manager that changes the configuration of the logger
  if cdsapi_client.__class__.__name__.startswith("Legacy"):
    if hasattr(cdsapi_client, "logging_decorator"):
      cdsapi_client.logging_decorator = lambda x: x

  return cdsapi_client
