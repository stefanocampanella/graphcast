from zipfile import ZipFile

import argparse
import copernicusmarine as cm
import gcsfs as gcs
import logging
import pathlib
import tempfile
import tomllib
import xarray
import xarray as xr
from datetime import datetime
from zarr.storage import TempStore

from graphcast.download_utils import DateInterval, DateIntervalsRange, Postprocess, get_cdsapi_client

parser = argparse.ArgumentParser(prog=__file__)
parser.add_argument("config", help="Path to configuration file", type=pathlib.Path)
parser.add_argument("output", help="Path to output file", type=pathlib.Path)
parser.add_argument("--provider", choices=["cds", "cm", "gcs"], default="cm")
parser.add_argument("--static", help="Download static dataset", action='store_true')
parser.add_argument("--start", help="Start date", default=None, type=str)
parser.add_argument("--stop", help="Stop date", default=None, type=str)
parser.add_argument("--year", help="Year", default=None, type=int)
parser.add_argument("--step", default='2 W')
parser.add_argument("--delta", default='1 day')
parser.add_argument("--overwrite", help="Whether to overwrite existing outputs", action='store_true')
parser.add_argument("--no-progress", help="Whether to display a progress bar", action='store_true')
parser.add_argument('--log', default='info')
args = parser.parse_args()

logger = logging.getLogger(__name__)
logging.basicConfig(format='%(levelname)s - %(asctime)s: %(message)s',
                    datefmt='%Y-%m-%dT%H:%M:%S',
                    level=getattr(logging, args.log.upper()))

# Avoid annoying copernicusmarine log handling
if args.provider == 'cm':
  cm_logger = logging.getLogger('copernicus_marine_root_logger')
  for handler in cm_logger.handlers:
    cm_logger.removeHandler(handler)
  cm_logger.setLevel(level=getattr(logging, args.log.upper()))

# If destination exists and should not overwrite, raise and exit.
if args.output.exists() and not args.overwrite:
  raise ValueError('Output destination already exists')

# The same script should download both static (no time dimension, e.g. masks) and timeseries.
# Also, arguments allows to download a single year, or from a specific date range.
# Finally, the specified date range should be divided and processed in sub-intervals to allow downloading and
# postprocessing with memory constraints of CINECA Leonardo serial nodes (the only ones with internet connection).
if args.static:
  date_intervals = [None]
elif (args.start is not None) and (args.stop is not None):
  date_intervals = DateIntervalsRange(args.start, args.stop, step=args.step, delta=args.delta)
elif args.year is not None:
  date_intervals = DateIntervalsRange(datetime(year=args.year, month=1, day=1),
                                      datetime(year=args.year + 1, month=1, day=1),
                                      step=args.step, delta=args.delta)
else:
  raise ValueError("Options '--static', '--start' and '--stop', or '--year'  must be provided.")

if args.no_progress:
  from contextlib import nullcontext as ProgressBar
else:
  from dask.diagnostics import ProgressBar

with args.config.open('rb') as file:
  configs = tomllib.load(file)

postprocess = Postprocess(configs=configs['postprocess'])

def _open_dataset(date_interval: DateInterval | None, dir = pathlib.Path | None, **kwargs) -> xarray.Dataset:
  """Provides a common interface, whether one is downloading from Copernicus Marine, Climate Data Store, etc.
  Depending on the particular implementation, it might download temporary files to `dir`

  Args:
    date_interval (DateInterval, optional): Date interval to download. Defaults to None (whole dataset).
    dir (pathlib.Path, optional): Directory to download to. Defaults to None (download to tempfile default path).
    **kwargs: Additional keyword arguments passed to library code (e.g. copernicusmarine)
  """
  pass

if args.provider == 'cm':

  def _open_dataset(date_interval=None, dir=None, **kwargs):
    if date_interval is not None:
      kwargs = {'start_datetime': date_interval.start, 'end_datetime': date_interval.stop, **kwargs}
    ds = cm.open_dataset(**kwargs)
    return ds

elif args.provider == 'cds':

  def _open_dataset(date_interval=None, dir=None, **kwargs):

    def _get_request(dates=None):
      "Returns a request to the Climate Data Store using the provided dates."
      request = kwargs.copy()
      dataset_name = request.pop('dataset')
      if dates is not None:
        request = {
          **request,
          "year": f"{dates[0].year}",
          "month": [f"{dates[0].month:02}"],
          "day": [f"{date.day:02}" for date in dates],
          "format": "netcdf"}
      return dataset_name, request

    def _process_request(dataset_name, request):
      """Submit a request to the Climate Data Store, download some temporary NetCDFs, and returns a dataset.
      Temporary files are deleted on exit.
      """
      file = tempfile.NamedTemporaryFile(dir=dir, suffix='.zip', delete=False)
      file.close()

      client = get_cdsapi_client(progress=not args.no_progress, client_logger=logger)
      logging.debug(f"Submitting request {request} with destination {file.name}")
      client.retrieve(dataset_name, request, file.name)

      with tempfile.TemporaryDirectory(dir=dir) as tmpdir:
        path = pathlib.Path(tmpdir)
        with ZipFile(file.name) as zipfile:
          zipfile.extractall(path=path)
        # noinspection PyTypeChecker
        ds = xarray.open_mfdataset(path.glob('*.nc'), engine='h5netcdf') # cdsapi download one NetCDF per variable :(
        ds = ds.rename(valid_time='time')
        if extra_coords := [name for name in ds.coords if name not in ['latitude', 'longitude', 'time']]:
          ds = ds.drop_vars(extra_coords)
        # The dataset must be loaded in memory, since the temporary directory will be deleted with all the NetCDFs within it.
        # However, ds should be rather small. Hence, there should be no need to lazily load the dataset.
        ds = ds.compute()
      return ds

    def _dataset_slices():
      """For each sequence of consecutive dates within the same month and year, yields the corresponding slice of the requested dataset (along the time dimension).
      Indeed, cdsapi can't process a request with arbitrary date intervals."""
      current_date = date_interval.start
      dates = []
      for date in date_interval:
        if date.month == current_date.month and date.year == current_date.year:
          dates.append(date)
        else:
          dataset_name, request = _get_request(dates=dates)
          ds = _process_request(dataset_name, request)
          current_date = date
          dates = [current_date]
          yield ds

      dataset_name, request = _get_request(dates=dates)
      ds = _process_request(dataset_name, request)
      yield ds

    if date_interval is None:
      dataset_name, request = _get_request()
      ds = _process_request(dataset_name, request)
    else:
      ds = xarray.merge(_dataset_slices())
    return ds

elif args.provider == 'gcs':

  def _open_dataset(date_interval=None, dir=None, **kwargs):

    fs = gcs.GCSFileSystem(token='anon', access='read_only', consistency='md5')
    store = fs.get_mapper(kwargs['url'])
    ds = xr.open_zarr(store=store)
    if (variables := kwargs.get('variables')) is not None:
      if variables_not_found := [name for name in variables if name not in ds.data_vars]:
        logging.warning(f"{', '.join(variables_not_found)} variables not found")
      ds = ds.drop_vars(names=[name for name in ds.data_vars if name not in variables])
    if date_interval is not None:
      ds = ds.sel(time=slice(date_interval.start, date_interval.stop))
    return ds

else:

  raise ValueError("The option '--provider' must be one of 'cds', 'cm', or 'gcs'")


def _download(store, date_interval, tempdir, **kwargs):
  if date_interval is None:
    info_msg = "Processing static variable"
  else:
    info_msg = f"Processing timeseries {date_interval}"
  logging.info(info_msg)
  # When downloading from Copernicus Marine Data Store or Climate Data Store, the typical case is a large dataset,
  # spanning a long time period, with several sets of variables in different datasets (bio, phys, etc.),
  # which needs to be downloaded one piece at a time. Hence, `datasets` in the TOML configuration file
  # represents different pieces of the same dataset.
  datasets = []
  for ds_conf in configs['datasets']:
    ds = _open_dataset(date_interval, tempdir, **ds_conf)
    ds = postprocess(ds)
    datasets.append(ds)
  dataset = xr.merge(datasets, join='exact')
  dataset = dataset.chunk(**{dim: -1 for dim in dataset.dims})
  delayed = dataset.to_zarr(store=store, compute=False, **kwargs)
  with ProgressBar():
    delayed.compute()

if __name__ == '__main__':

  with TempStore() as store:
    is_first_iteration = True
    # For each date_interval: download, postprocess, and append the dataset to a temporary Zarr
    for date_interval in date_intervals:
      with tempfile.TemporaryDirectory() as tempdir:
        if is_first_iteration:
          _download(store, date_interval, tempdir=tempdir, mode='w')
          is_first_iteration = False
        else:
          _download(store, date_interval, tempdir=tempdir, mode='a-', append_dim='time')

    logging.info(f"Saving dataset to {args.output}")
    # Load the temporary Zarr, eventually rechunk and save to final destination.
    dataset = xarray.open_zarr(store)
    rechunk_conf = configs.get('rechunk', {})
    # If rechunk_conf is empty, the following _should_ be a no-op
    dataset = dataset.chunk(**rechunk_conf)
    output_conf = configs.get('output_settings', {})
    delayed = dataset.to_zarr(store=args.output, mode='w', compute=False, **output_conf)

    with ProgressBar():
      delayed.compute()
