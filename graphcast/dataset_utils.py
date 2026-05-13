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
import functools
import logging
import pathlib
import tempfile
import warnings
from collections import OrderedDict
from collections.abc import Generator, Iterable, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any
from zipfile import ZipFile

import cdsapi
import copernicusmarine as cm
import gcsfs as gcs
import numpy as np
import pandas as pd
import requests
import xarray as xr
import xarray_regrid
from numcodecs import Blosc
from scipy.ndimage import gaussian_filter
from xarray.core.types import InterpOptions

from graphcast.cli_utils import Configs

logger = logging.getLogger(__name__)


@dataclass
class DateInterval:
  """Low level (for internal use) representation of a date interval, which can be iterated over with arbitrary step size.

  When iterating over a DateInterval object, extrema (`start` and `stop`) are included.

  Attributes:
    start (datetime.datetime): The start date of the interval.
    end (datetime.datetime): The stop date of the interval.
    step (datetime.timedelta): The step size used when iterating over the interval.
  """

  start: datetime
  end: datetime
  step: timedelta = timedelta(days=1)

  def __iter__(self):
    def _date_iterator():
      current = self.start
      while current < self.end:
        yield current
        current += self.step
      yield self.end

    return _date_iterator()

  def to_numpy(self) -> np.ndarray:
    """Convert the DateInterval to a numpy array of datetime64[ns].

    Returns:
      np.ndarray: Array of datetime64[ns] values from start to end (inclusive) with the given step.
    """
    # Create a Pandas date range from start to end (inclusive) with the given step
    date_range = pd.date_range(start=self.start, end=self.end, freq=self.step)

    # Convert to numpy array with datetime64[ns] dtype
    return date_range.to_numpy(dtype="datetime64[ns]")

  def __repr__(self):
    return f"[{self.start.isoformat()}, {self.end.isoformat()}]"


# DateInterval`s include the extrema (as open_dataset in copernicusmarine do).
# DateIntervalsRange uses Python convention (right open).
# The union of all intervals in DateIntervalsRange(start, stop, delta) == [start, stop) == [start, stop - delta]
class DateIntervalsRange(Iterable):
  """Iterable object that yields `DateInterval`s.

  Note:
    It uses Python convention (right open), such that
    the union of all intervals in DateIntervalsRange(start, stop, delta) == [start, stop) == [start, stop - delta]

  Args:
    start (datetime.datetime): The start date of the interval.
    end (datetime.datetime): The stop date of the interval.
    step (str | datetime.timedelta, optional): Size of the sub-intervals. Defaults to 1 week.
    delta (datetime.timedelta, optional): The step size used when iterating over the interval. Defaults to 1 day.
  """

  def __init__(
    self,
    start: datetime | str,
    end: datetime | str,
    step: timedelta | str | None = None,
    delta: timedelta | str | None = None,
  ):
    def may_parse_datetime(date: datetime | str) -> datetime:
      if isinstance(date, datetime):
        return date
      elif isinstance(date, str):
        return datetime.fromisoformat(date)
      else:
        raise ValueError(
          f"Invalid date format: {date}. Must be a datetime or a string in ISO format."
        )

    def may_parse_timedelta(delta: timedelta | str) -> timedelta:
      if isinstance(delta, timedelta):
        return delta
      elif isinstance(delta, str):
        return pd.Timedelta(delta).to_pytimedelta()
      else:
        raise ValueError(
          f"Invalid timedelta format: {delta}. Must be a timedelta or a string in ISO format."
        )

    self.start = may_parse_datetime(start)
    self.end = may_parse_datetime(end)
    if step is None:
      self.step = self.end - self.start
    else:
      self.step = may_parse_timedelta(step)

    if delta is not None:
      self.delta = may_parse_timedelta(delta)
    else:
      self.delta = timedelta(days=1)

  def __repr__(self):
    return f"from {self.start.isoformat()} to {(self.last_valid_date).isoformat()} (included)"

  def __iter__(self):
    def _date_iterator() -> Generator[DateInterval, Any]:
      current = self.start
      next = min(current + self.step, self.end)
      while current < self.end:
        yield DateInterval(current, next - self.delta, self.delta)
        current = next
        next = min(current + self.step, self.end)

    return _date_iterator()

  def __getitem__(self, item):
    return list(self)[item]

  @property
  def last_valid_date(self):
    return self.end - self.delta

  def __len__(self):
    return sum(1 for _ in self)


class Provider:
  def __init__(self, progress=True, log_level="info", client_logger=None):
    """
    Initializes the Provider object.

    Args:
      progress (bool): Whether to show progress bars.
      log_level (str): Logging level.
      client_logger (logging.Logger): Logger for the client.
    """
    self.progress = progress
    self.log_level = log_level
    self.client_logger = client_logger

  def open_dataset(
    date_interval: DateInterval | None, dir=pathlib.Path | None, **kwargs
  ) -> xr.Dataset:
    """Provides a common interface, whether one is downloading from Copernicus Marine, Climate Data Store, etc.
    Depending on the particular implementation, it might download temporary files to `dir`

    Args:
      date_interval (DateInterval, optional): Date interval to download. Defaults to None (whole dataset).
      dir (pathlib.Path, optional): Directory to download to. Defaults to None (download to tempfile default path).
      **kwargs: Additional keyword arguments passed to library code (e.g. copernicusmarine)
    """
    pass


class CopernicusMarine(Provider):
  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    # Avoid annoying copernicusmarine log handling
    cm_logger = logging.getLogger("copernicus_marine_root_logger")
    for handler in cm_logger.handlers:
      cm_logger.removeHandler(handler)
    cm_logger.setLevel(level=getattr(logging, self.log_level.upper()))

  def open_dataset(self, date_interval=None, dir=None, **kwargs):
    if date_interval is not None:
      # Note! Copernicus Marine Data Store uses Python convention (right open)
      kwargs = {
        "start_datetime": date_interval.start,
        "end_datetime": date_interval.end + date_interval.step,
        **kwargs,
      }
    ds = cm.open_dataset(**kwargs)
    return ds


class ClimateDataStore(Provider):
  """Support for cdsapi. ARCO-ERA5 (WeatherBench datasets) should be preferred."""

  def open_dataset(self, date_interval=None, dir=None, **kwargs):
    if date_interval is None:
      dataset_name, request = self._get_request()
      ds = self._process_request(dataset_name, request, dir, self.progress, self.client_logger)
    else:
      consecutive_dates = self._consecutive_dates_with_same_month_or_year(date_interval)
      datasets = []
      for dates in consecutive_dates:
        dataset_name, request = self._get_request(dates=dates, **kwargs)
        ds = self._process_request(dataset_name, request, dir, self.progress, self.client_logger)
        # _process_request drops the time dimension if of length one
        if len(dates) == 1:
          ds = ds.expand_dims(dim="time", axis=0)
        datasets.append(ds)
      ds = xr.merge(datasets)
      if date_interval is not None:
        if not np.array_equal(date_interval.to_numpy(), ds["time"]):
          warnings.warn(
            f"The requested date interval {date_interval!r} "
            f"is not matching the time coordinate {ds['time']!r}"
          )
    return ds

  @staticmethod
  def _consecutive_dates_with_same_month_or_year(date_interval):
    def partition(f, sequence):
      part = []
      subseq = []
      last = None
      for current in sequence:
        if (last is None) or (not subseq) or f(last, current):
          subseq.append(current)
        else:
          part.append(subseq)
          subseq = [current]
        last = current
      part.append(subseq)
      return part

    def same_month_or_year(x, y):
      return x.month == y.month and x.year == y.year

    return partition(same_month_or_year, date_interval)

  def _get_request(self, dates=None, **kwargs):
    "Returns a request to the Climate Data Store using the provided dates."
    request = kwargs.copy()
    dataset_name = request.pop("dataset")
    if dates is not None:
      request = {
        **request,
        "hyear": f"{dates[0].year}",
        "hmonth": [f"{dates[0].month:02}"],
        "hday": [f"{date.day:02}" for date in dates],
        "data_format": "grib2",
        "download_format": "zip",
      }
    return dataset_name, request

  def _process_request(self, dataset_name, request, dir, progress, client_logger):
    """Submit a request to the Climate Data Store, download some temporary NetCDFs, and returns a dataset.
    Temporary files are deleted on exit.
    """
    file = tempfile.NamedTemporaryFile("w+", dir=dir, suffix=".zip", delete=False)
    file.close()

    client = self.get_cdsapi_client(progress=progress, client_logger=client_logger)
    logger.debug(f"Submitting request {request} with destination {file.name}")
    client.retrieve(dataset_name, request, file.name)

    with tempfile.TemporaryDirectory(dir=dir) as tmpdir:
      path = pathlib.Path(tmpdir)
      with ZipFile(file.name) as zipfile:
        zipfile.extractall(path=path)
      # noinspection PyTypeChecker
      ds = xr.open_mfdataset(
        path.glob("*"), engine="cfgrib", decode_timedelta=True
      )  # cdsapi download one NetCDF per variable :(
      # ds = ds.rename(valid_time='time')
      if extra_coords := [
        name for name in ds.coords if name not in ["latitude", "longitude", "time"]
      ]:
        ds = ds.drop_vars(extra_coords)
      # The dataset must be loaded in memory, since the temporary directory will be deleted with all the NetCDFs within it.
      # However, ds should be rather small. Hence, there should be no need to lazily load the dataset.
      ds = ds.compute()
    return ds

  # Credits to the amazing Stefano Piani from OGS
  def get_cdsapi_client(url: str | None = None, client_logger=None, **kwargs):
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


class GoogleCloudStorage(Provider):
  def open_dataset(self, date_interval=None, dir=None, **kwargs):
    fs = gcs.GCSFileSystem(token="anon", access="read_only", consistency="md5")
    store = fs.get_mapper(kwargs["url"])
    ds = xr.open_zarr(store=store)
    if (variables := kwargs.get("variables")) is not None:
      if variables_not_found := [name for name in variables if name not in ds.data_vars]:
        logger.warning(f"{', '.join(variables_not_found)} variables not found")
      ds = ds.drop_vars(names=[name for name in ds.data_vars if name not in variables])
    if date_interval is not None:
      ds = ds.sel(time=slice(date_interval.start, date_interval.end))
    return ds


class URLProvider(Provider):
  """Provider that downloads a NetCDF file from a URL.

  This provider downloads a NetCDF file from a specified URL and opens it as an xarray Dataset.
  It supports filtering by date interval if the dataset has a time dimension.
  """

  def open_dataset(self, date_interval=None, dir=None, **kwargs):
    """Downloads a NetCDF file from a URL and opens it as an xarray Dataset.

    Args:
      date_interval (DateInterval, optional): Date interval to filter the dataset. Defaults to None.
      dir (pathlib.Path, optional): Directory to download temporary files to. Defaults to None.
      **kwargs: Additional keyword arguments, must include 'url'.
        url (str): URL of the NetCDF file to download.
        variables (list, optional): List of variables to keep in the dataset.

    Returns:
      xr.Dataset: The downloaded dataset.

    Raises:
      ValueError: If 'url' is not provided in kwargs.
    """
    if "url" not in kwargs:
      raise ValueError("URL must be provided for URLProvider")

    url = kwargs["url"]
    logger.info(f"Downloading NetCDF from URL: {url}")

    # Create a temporary file to download the NetCDF
    with tempfile.NamedTemporaryFile(dir=dir, suffix=".nc", delete=False) as temp_file:
      temp_path = temp_file.name

    try:
      # Use requests to download the file
      response = requests.get(url, stream=True)
      response.raise_for_status()  # Raise an exception for HTTP errors

      with open(temp_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
          f.write(chunk)

      # Open the downloaded file as an xarray Dataset
      ds = xr.open_dataset(temp_path)

      # Filter by variables if specified
      if (variables := kwargs.get("variables")) is not None:
        if variables_not_found := [name for name in variables if name not in ds.data_vars]:
          logger.warning(f"{', '.join(variables_not_found)} variables not found")
        ds = ds.drop_vars(names=[name for name in ds.data_vars if name not in variables])

      # Filter by date interval if specified
      if date_interval is not None and "time" in ds.dims:
        ds = ds.sel(time=slice(date_interval.start, date_interval.end))

      return ds

    finally:
      # Clean up the temporary file
      if pathlib.Path(temp_path).exists():
        pathlib.Path(temp_path).unlink()


ProvidersRegistry = {
  "cds": ClimateDataStore,
  "cm": CopernicusMarine,
  "gcs": GoogleCloudStorage,
  "url": URLProvider,
}


class Process:
  """
  Applies a sequence of pre|post-processing steps to a xarray.Dataset.

  The steps are provided as a list of dictionaries containing their configurations.
  All methods defined on a xarray.Dataset can be used as processing steps, in addition to the
  methods defined in this class (which have precedence).

  Attributes:
    configs (OrderedDict[str, Any]): Configuration for each processing step.
    mask (xr.DataArray): Values on which to apply the processing step.
  """

  def __init__(self, steps: Sequence[Configs] | None = None, mask: xr.DataArray | None = None):
    """
    Initializes the Postprocess object.

    Args:
      steps (Dict[str, Any]): Configuration for each processing step.
      mask (xr.DataArray, optional): Mask to apply to the dataset.
    """

    self.configs = OrderedDict()
    if steps is not None:
      for step in steps:
        # Copy to avoid mutating the arguments
        step = step.copy()
        name = step.pop("name")
        self.configs[name] = step
    self.mask = mask

  def __call__(self, ds: xr.Dataset) -> xr.Dataset:
    """
    Applies the configured processing steps to the provided dataset.
    Each step takes a Dataset as input and returns a Dataset.
    Additional arguments have to be provided as keyword arguments.
    Missing required arguments must issue a warning and step degrades into a no-op.

    Args:
      ds (xr.Dataset): The input dataset to process.

    Returns:
      xr.Dataset: The processed dataset.
    """

    if ("regrid" in self.configs) and ("interpolate" in self.configs):
      logger.debug(
        "Both 'regrid' and 'interpolate' are specified in the configuration. "
        "'regrid' will be applied first, followed by 'interpolate'."
      )
    logger.info(
      "Processing dataset with the following steps: " + ", ".join(self.configs.keys()) + ". "
    )
    for step in self.configs:
      conf = self.configs.get(step, {})
      logger.info(f"Applying {step} with configuration {conf}")
      if step in dir(self):
        ds = getattr(self, step)(ds, **conf)
      elif step in dir(ds):
        ds = getattr(ds, step)(**conf)
      else:
        warnings.warn(f"Unrecognized processing step {step} with configuration {conf}")
    return ds

  def transpose(self, ds: xr.Dataset, dims: Sequence[str] | None = None, **kwargs):
    if dims is None:
      warnings.warn("dims must be specified")
    else:
      ds = ds.transpose(*dims, **kwargs)
    return ds

  def regrid(
    self,
    ds: xr.Dataset,
    grid=None,
    latitude_dim="latitude",
    longitude_dim="longitude",
    **kwargs,
  ) -> xr.Dataset:
    """
    Regrids the dataset to a new grid using xarray-regrid (default using nearest neighbor algorithm).

    Args:
      ds (xr.Dataset): The input dataset.
      grid (dict, optional): Grid specification for the target grid.
      **kwargs: Additional arguments for the regridding method.

    Returns:
      xr.Dataset: The regridded dataset.

    Raises:
      ValueError: If grid is not specified.
    """

    if grid is None:
      warnings.warn("Grid must be specified")
    else:
      new_grid = xarray_regrid.Grid(**grid)
      target_dataset = new_grid.create_regridding_dataset(
        lat_name=latitude_dim, lon_name=longitude_dim
      )
      method = kwargs.get("method", "nearest")
      target_dataset = target_dataset.assign_coords(
        {
          latitude_dim: target_dataset[latitude_dim].astype(np.float32),
          longitude_dim: target_dataset[longitude_dim].astype(np.float32),
        }
      )
      regrid_conf = kwargs.get("kwargs", {})
      if method == "conservative":
        regrid_conf |= dict(latitude_coord=latitude_dim)
      ds = getattr(ds.regrid, method)(target_dataset, **regrid_conf)
    return ds

  def resample(self, ds: xr.Dataset, reduce=None, **kwargs):
    """
    Resamples the dataset along a specified dimension and reduces it using a given method.
    The implementation checks that the 'reduce' method is specified.

    Args:
      ds (xr.Dataset): The input dataset.
      **kwargs: Arguments for resampling, it must include 'reduce' specifying the reduction method.

    Returns:
      xr.Dataset: The resampled dataset.

    Raises:
      ValueError: If 'reduce' method is not specified.
    """

    if reduce is None:
      warnings.warn("Reduce method must be specified")
    else:
      ds_resample = ds.resample(**kwargs)
      ds = getattr(ds_resample, reduce)()
    return ds

  def rescale(self, ds: xr.Dataset, values=None) -> xr.Dataset:
    if values is None:
      warnings.warn("Values must be specified")
    else:
      for var, da in ds.data_vars.items():
        if var in values:
          ds[var] = values[var] * da
    return ds

  def rename_coordinates(self, ds: xr.Dataset, name_dict=None, set_new_coordinate=None):
    if name_dict is None:
      warnings.warn("Name dictionary must be specified")
    else:
      name_dict = {
        old_name: new_name for old_name, new_name in name_dict.items() if old_name in ds.coords
      }
      ds = ds.rename(name_dict=name_dict)
      for old_name, new_name in name_dict.items():
        if (set_new_coordinate is not None) and (new_name in set_new_coordinate):
          # Keep old coordinate values as a non-index coordinate (with new name)
          old_coordinate = ds.coords[new_name]
          new_coordinate = set_new_coordinate[new_name]
          ds = ds.drop_indexes(new_name)
          ds = ds.drop_vars(new_name)
          ds = ds.assign_coords({new_name: (new_name, new_coordinate)})
          ds = ds.assign_coords({old_name: (new_name, old_coordinate.data)})
    return ds

  def clip_negative(self, ds: xr.Dataset, variables: bool | Sequence[str] = True):
    for var, da in ds.data_vars.items():
      if variables is True or var in variables:
        ds[var] = da.clip(min=0.0)
    return ds

  def select_vars(self, ds: xr.Dataset, variables: Sequence[str] = None) -> xr.Dataset:
    if variables is None:
      warnings.warn("Variables must be specified")
    else:
      ds = ds.drop_vars(names=[var for var in ds.data_vars if var not in variables])
    return ds

  def apply_mask(self, ds: xr.Dataset, fill_value: float = np.nan, **kwargs) -> xr.Dataset:
    for var, da in ds.data_vars.items():
      mask = self.mask.isel({dim: 0 for dim in self.mask.dims if dim not in da.dims}, drop=True)
      ds[var] = da.where(mask, fill_value, **kwargs)
    return ds

  def fill(self, ds: xr.Dataset, fill_value: float = 0.0, variables=None) -> xr.Dataset:
    if variables is None:
      warnings.warn("Variables must be specified")
    else:
      for var, da in ds.data_vars.items():
        mask = self.mask.isel({dim: 0 for dim in self.mask.dims if dim not in da.dims}, drop=True)
        if var in variables:
          ds[var] = xr.where(da.isnull() & mask, fill_value, da)
    return ds

  def gauss_fill(self, ds: xr.Dataset, variables=None, **kwargs) -> xr.Dataset:
    if variables is None:
      warnings.warn("Variables must be specified")
    else:
      for var, da in ds.data_vars.items():
        if var in variables:
          mask = self.mask.isel({dim: 0 for dim in self.mask.dims if dim not in da.dims}, drop=True)
          # TODO: check that does as intended (why was I seeing less and less nans with increasing radius?)
          ds[var] = da.map_blocks(gauss_filter_nan, args=(mask,), kwargs=kwargs, template=da)
    return ds

  # FIXME: poor choice of the name, misleading. Rename it to `not_null_mask`, and revise toml configuration files accordingly
  def get_land_mask(self, ds: xr.Dataset, variable=None, mask_name=None) -> xr.Dataset:
    if variable is None:
      warnings.warn("Variable must be specified")
    elif mask_name is None:
      warnings.warn("Mask name must be specified")
    else:
      ds[mask_name] = xr.where(ds[variable].notnull(), True, False)
    return ds

  def time_shift(self, ds: xr.Dataset, quantity=None) -> xr.Dataset:
    if quantity is None:
      warnings.warn("Shift amount must be specified")
    else:
      ds = ds.assign_coords(time=ds.time - pd.Timedelta(quantity))
    return ds

  def flip(self, ds: xr.Dataset, dim=None) -> xr.Dataset:
    if dim is None:
      warnings.warn("Dim must be specified")
    else:
      ds = ds.isel({dim: slice(None, None, -1)})
    return ds

  def interpolate_na(self, ds, **kwargs) -> xr.Dataset:
    dim = kwargs.get("dim", "time")
    output_chunks = kwargs.pop("output_chunks", {})
    ds = ds.chunk({dim: -1})
    ds = ds.interpolate_na(**kwargs)
    ds = ds.chunk(**output_chunks)
    return ds

  def interpolate(
    self,
    ds: xr.Dataset,
    minimum_latitude: float = -90.0,
    maximum_latitude: float = 90.0,
    minimum_longitude: float = -180.0,
    maximum_longitude: float = 179.0,
    resolution: float = 1.0,
    method: InterpOptions = "linear",
    assume_sorted: bool = True,
    kwargs: dict[str, Any] | None = None,
  ) -> xr.Dataset:
    """
    Interpolates the dataset to a regular latitude/longitude grid.

    Args:
      ds (xr.Dataset): The input dataset.
      minimum_latitude (float): Minimum latitude of the target grid.
      maximum_latitude (float): Maximum latitude of the target grid.
      minimum_longitude (float): Minimum longitude of the target grid.
      maximum_longitude (float): Maximum longitude of the target grid.
      resolution (float): Grid resolution in degrees.
      method (InterpOptions): Interpolation method.
      assume_sorted (bool): Whether to assume the input coordinates are sorted.
      kwargs (Dict[str, Any], optional): Additional arguments for xarray's interp.

    Returns:
      xr.Dataset: The interpolated dataset.
    """

    eps = np.finfo(ds.latitude.dtype).eps
    latitude = np.arange(
      start=minimum_latitude, stop=maximum_latitude + eps, step=resolution, dtype=np.float32
    )
    longitude = np.arange(
      start=minimum_longitude, stop=maximum_longitude, step=resolution, dtype=np.float32
    )
    ds = ds.interp(
      latitude=latitude,
      longitude=longitude,
      method=method,
      assume_sorted=assume_sorted,
      kwargs=kwargs,
    ).astype(np.float32)
    return ds

  def get_sea_mask(
    self,
    ds: xr.Dataset,
    bathymetry="deptho",
    depth_coordinate="depth",
    depth_dim="depth",
    mask_name="sea_land_mask",
  ) -> xr.Dataset:
    bathymetry_values = ds[bathymetry].values
    depth = ds[depth_coordinate].values

    def get_mask(depth_map: np.ndarray, column_depths: np.ndarray, dtype="bool") -> np.ndarray:
      cell_center_depths = np.insert(column_depths[:-1], 0, 0.0) + 0.5 * np.diff(
        column_depths, prepend=0.0
      )
      mask = np.stack([depth_map >= d for d in cell_center_depths], axis=0)
      return mask.astype(dtype)

    mask = get_mask(bathymetry_values, depth)
    dims = (depth_dim,) + ds[bathymetry].dims
    ds[mask_name] = (dims, mask)
    return ds

  # TODO: document astype behaviour
  def astype(self, ds: xr.Dataset, dtype=None, casting=None, **kwargs) -> xr.Dataset:
    if dtype is not None and casting is not None:
      ds = ds.astype(dtype=dtype, casting=casting)
    elif kwargs is not None:
      for variable, astype_kwargs in kwargs.items():
        if variable in ds.data_vars:
          ds[variable] = ds[variable].astype(**astype_kwargs)
    return ds


def gauss_filter_nan(data, mask, **kwargs):
  data_u = xr.where(data.isnull() | np.logical_not(mask), 0.0, data)
  data_u.values = gaussian_filter(data_u.values, **kwargs)
  data_v = xr.where(data.isnull() | np.logical_not(mask), 0.0, 1.0)
  data_v.values = gaussian_filter(data_v.values, **kwargs)
  data_u = data_u / data_v
  data = xr.where(data.isnull() & mask, data_u, data)
  return data


def check_values(variables: None | Sequence[str] = None, mask: None | xr.DataArray = None):
  def checker(block, dataset_info=None, block_info=None):
    try:
      if mask is not None:
        mask_ = mask.isel({dim: 0 for dim in mask.dims if dim not in block.dims}, drop=True)
        masked_block = block.where(mask_, 0.0)
        nonvalid_values = masked_block.isnull()
      else:
        nonvalid_values = block.isnull()
      if nonvalid_values.any():
        warnings.warn(
          f"{nonvalid_values.sum().values} NaN values found "
          f"in dataset {dataset_info['dataset']} for variable {dataset_info['variable']}"
        )
    except UserWarning as w:
      # FIXME: block_info here is None, find a way to forward information about the location of the nans
      w.add_note(f"{block_info=}")

    return block

  def decorator(reader):
    @functools.wraps(reader)
    def decorated(pathlike: str | pathlib.Path, *args, **kwargs) -> xr.Dataset:
      ds = reader(pathlike, *args, **kwargs)
      logger.info("Checking for Nans")
      allowed_variables = variables or ds.data_vars
      for var, da in ds.data_vars.items():
        if var in allowed_variables:
          checker_kwargs = {"dataset_info": {"dataset": pathlike, "variable": var}}
          ds[var] = da.map_blocks(checker, kwargs=checker_kwargs, template=da)
      return ds

    return decorated

  return decorator


# FIXME: this is probaby broken, and should be integrated by
def check_date_range(start_date: datetime, end_date: datetime):
  """
  Decorator to check that all dates in a dataset are within the specified range
  and that each day is included in the interval.

  Args:
    start_date: Start date in YYYY-MM-DD format (required)
    end_date: End date in YYYY-MM-DD format (required)
  """

  def decorator(reader):
    @functools.wraps(reader)
    def decorated(pathlike: str | pathlib.Path, *args, **kwargs) -> xr.Dataset:
      ds = reader(pathlike, *args, **kwargs)

      logger.info("Checking for mismatching dates")
      if "time" in ds.dims:
        # Check that each day is included in the interval
        expected_dates = pd.date_range(start=start_date, end=end_date, freq="D")
        dataset_dates = pd.to_datetime(ds.time.values)

        # Find missing dates within the expected range
        missing_dates = []
        for expected_date in expected_dates:
          if expected_date not in dataset_dates:
            missing_dates.append(expected_date)

        if missing_dates:
          missing_str = [date.strftime("%Y-%m-%d") for date in missing_dates[:10]]  # Show first 10
          if len(missing_dates) > 10:
            missing_str.append(f"... and {len(missing_dates) - 10} more")
          warnings.warn(
            f"Dataset at {pathlike} is missing {len(missing_dates)} dates "
            f"from the expected interval [{start_date}, {end_date}]: {missing_str}"
          )

      return ds

    return decorated

  return decorator


def check_coordinates(reader):
  @functools.wraps(reader)
  def decorated(pathlike: str | pathlib.Path, *args, **kwargs) -> xr.Dataset:
    ds = reader(pathlike, *args, **kwargs)

    logger.info("Checking for mismatching coordinates")
    # 1. Check that each dataset contains all the days between beginning and end
    if "time" in ds.dims:
      time_range = pd.date_range(start=ds.time.min().item(), end=ds.time.max().item(), freq="D")
      if not all(date in ds.time.values for date in time_range):
        missing_dates = [date for date in time_range if date not in ds.time.values]
        warnings.warn(f"Dataset at {pathlike} is missing dates: {missing_dates}")

    # 2. Check that each dataset uses the [0, 360) convention for longitude
    if "longitude" in ds.dims or "lon" in ds.dims:
      lon_dim = "longitude" if "longitude" in ds.dims else "lon"
      if ds[lon_dim].min().item() < 0 or ds[lon_dim].max().item() >= 360:
        warnings.warn(f"Dataset at {pathlike} does not use the [0, 360) convention for longitude")

    # 3. Check that each dataset contains all latitudes in [-90, 90], and use the [-90, 90] convention
    if "latitude" in ds.dims or "lat" in ds.dims:
      lat_dim = "latitude" if "latitude" in ds.dims else "lat"
      if ds[lat_dim].min().item() < -90 or ds[lat_dim].max().item() > 90:
        warnings.warn("Dataset has latitudes outside the [-90, 90] range")
      if ds[lat_dim][0].item() > ds[lat_dim][-1].item():
        warnings.warn(f"Dataset at {pathlike} does not use the [90, -90] convention for latitude")

    return ds

  return decorated


# FIXME: the code should handle both Zarr (using a DirectoryStore or a ZipStore) and NetCDF files.
def open_dataset_wo_static(path: pathlib.Path, time_dim: str = "time", chunks=None) -> xr.Dataset:
  """
  Open a dataset from a single Zarr file/store or a directory containing multiple Zarr zip files.

  - If `path` is a directory with one or more .zip files, open all of them via xarray.open_mfdataset(engine='zarr').
  - In all other cases, open it via xarray.open_dataset(engine='zarr').

  Returns a xarray.Dataset filtered to only data variables that include the provided time dimension.
  """
  path = pathlib.Path(path)
  if not path.exists():
    raise ValueError(f"Input path {path} does not exist")

  def _drop_static_vars(ds: xr.Dataset, time_dim: str) -> xr.Dataset:
    ds = ds.drop_vars([name for (name, var) in ds.data_vars.items() if time_dim not in var.dims])
    return ds

  # As the Dask graph tends to be huge it's important to avoid inline_array=True,
  # see: https://docs.dask.org/en/latest/generated/dask.array.from_array.html#dask.array.from_array
  if path.is_dir():
    zip_files = sorted(p for p in path.glob("*.zip"))
    if zip_files:
      # noinspection PyTypeChecker
      ds = xr.open_mfdataset(
        [str(p) for p in zip_files],
        preprocess=lambda ds: _drop_static_vars(ds, time_dim),
        engine="zarr",
        combine="by_coords",
        inline_array=False,
        chunks=chunks,
      )
      return ds

  ds = xr.open_dataset(str(path), engine="zarr", inline_array=False, chunks=chunks)
  ds = _drop_static_vars(ds, time_dim)

  return ds


def open_mfdataset(
  paths: Sequence[str | pathlib.Path], time_dim: str = "time", chunks=None
) -> xr.Dataset:
  """
  Open multiple zipped Zarr datasets and combine them as xarray.open_mfdataset would, with a
  specific behavior for static variables (those without the provided time dimension):

  - Time-varying variables (containing `time_dim` among their dimensions) are merged along
    coordinates (typically along the time dimension) using xarray.open_mfdataset(combine='by_coords').
  - Static variables (that do not contain `time_dim`) are expected to be identical across the
    input datasets if duplicated; they are validated and included once, as-is, in the output.

  Parameters
  ---------
  paths: Sequence[str | pathlib.Path]
      List of paths to zipped Zarr stores (.zip). They must exist. The function does not support
      directories; pass individual .zip paths instead.
  time_dim: str
      Name of the time dimension. Variables that do not include this dimension are considered static.
  chunks: Any
      Chunking specification forwarded to xarray open calls. Use None to keep existing chunking.

  Returns
  -------
  xr.Dataset
      Dataset obtained by combining the time-varying variables by coordinates and adding the static
      variables (validated to be equal across inputs) unchanged.
  """
  if not isinstance(paths, (list, tuple)):
    raise TypeError("paths must be a sequence of path-like strings pointing to zipped Zarr stores")
  if len(paths) == 0:
    raise ValueError("paths cannot be empty")

  str_paths = [str(pathlib.Path(p)) for p in paths]
  for p in str_paths:
    if not pathlib.Path(p).exists():
      raise ValueError(f"Input path {p} does not exist")

  # Phase 1: scan inputs to collect and validate static variables (no `time_dim`).
  static_vars: dict[str, xr.DataArray] = {}

  def _collect_and_validate_static(ds: xr.Dataset):
    nonlocal static_vars
    for name, var in ds.data_vars.items():
      if time_dim not in var.dims:
        if name in static_vars:
          # Ensure equality (values and coordinates). Attributes are ignored.
          if not var.equals(static_vars[name]):
            raise ValueError(
              f"Static variable '{name}' differs across inputs. All static variables must be identical."
            )
        else:
          static_vars[name] = var

  # Open each dataset quickly to inspect static variables. Keep inline_array=False to avoid huge graphs.
  for p in str_paths:
    ds = xr.open_dataset(p, engine="zarr", inline_array=False, chunks=chunks)
    try:
      _collect_and_validate_static(ds)
    finally:
      ds.close()

  # Phase 2: combine time-varying variables by coordinates using open_mfdataset
  def _drop_static(ds: xr.Dataset) -> xr.Dataset:
    to_drop = [name for name, var in ds.data_vars.items() if time_dim not in var.dims]
    if to_drop:
      # Drop only those present to avoid errors if some files lack certain static vars
      ds = ds.drop_vars(to_drop, errors="ignore")
    return ds

  ds_dynamic = xr.open_mfdataset(
    str_paths,
    engine="zarr",
    combine="by_coords",
    preprocess=_drop_static,
    inline_array=False,
    chunks=chunks,
  )

  # Merge back the validated static variables (if any)
  if static_vars:
    static_ds = xr.Dataset({k: v for k, v in static_vars.items()})
    # xr.merge will align coordinates as needed; prefer dynamic attrs
    ds_dynamic = xr.merge([ds_dynamic, static_ds], combine_attrs="override")

  return ds_dynamic


def save_to_zarr(
  dataset: xr.Dataset,
  output_path: pathlib.Path,
  overwrite=False,
  precompute=False,
  compressor_kwargs=None,
):
  compressor_kwargs = compressor_kwargs or {}

  if precompute:
    dataset = dataset.compute()

  for var in dataset.data_vars:
    if "chunks" in dataset[var].encoding:
      del dataset[var].encoding["chunks"]

  for var in dataset.data_vars:
    dataset[var].encoding["compressor"] = Blosc(**compressor_kwargs)

  if output_path.exists() and not overwrite:
    raise ValueError(f"Output path {output_path} already exists")

  # Notice that parallel writes to Zarr using zip store are (apparently) not supported.
  logger.info(f"Saving dataset to {output_path} as Zarr")
  dataset.to_zarr(output_path, compute=True, consolidated=True, mode="w")


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
      expected_full = pd.date_range(
        start=sorted_idx[0],
        end=sorted_idx[-1],
        freq="D",
        tz=getattr(sorted_idx, "tz", None),
      )
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
      expected_full = xr.cftime_range(
        start=sorted_idx[0], end=sorted_idx[-1], freq="D", calendar=calendar
      )
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
