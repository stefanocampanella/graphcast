# TODO:
#   1. Documentation is missing, fix it.
#   2. Download and merge phases could, and should, be fused together.
#   3. Postprocessing step should be renamed preprocess, and passed as an argument to
#      xr.open_dataset or xr.open_mfdataset. However, how logging and performance would be affected by the latter?
#   4. Download and merge phases should use Dask MPI.
#   5. Download and merge should use the function save_to_zarr from utils, which in turn should be extended to support
#      them.
#   6. Progress bars should be dropped.
import logging
import pathlib
import pprint
import tempfile
import tomllib
from contextlib import nullcontext
from datetime import datetime

import click
import dask
import numpy as np
import xarray as xr
from dask.diagnostics import ProgressBar
from numcodecs.blosc import Blosc
from zarr.storage import TempStore, ZipStore

from graphcast.cli_utils import DictParamType
from graphcast.dataset_utils import (Configs,
                                     DateIntervalsRange,
                                     Process,
                                     ProvidersRegistry,
                                     check_coordinates,
                                     check_date_range,
                                     check_values,
                                     open_mfdataset,
                                     save_to_zarr,
                                     valid_time_coordinate)
from graphcast.dask_distributed_utils import get_client


def bar(progress):
  if progress:
    return ProgressBar()
  else:
    return nullcontext()


def _parse_timeseries_arguments(configs, output_path, start = None, end = None, array_id = None):
  # For a time series, if the dataset is downloaded using SLURM arrays,
  # the date interval is determined by the array ID, and so the output path.
  start = start or configs.get('start', None)
  end = end or configs.get('end', None)
  if start is None or end is None:
    raise ValueError("Start and end dates must be specified when downloading timeseries, either in the config file "
                     "or as command line arguments. See --help for more information.")
  if array_id is not None:
    array_config = configs.get('slurm_array', {})
    step = array_config.get('step', None)
    if array_config and step is not None:
      intervals_range = DateIntervalsRange(start=start, end=end, step=step)
      if array_id >= len(intervals_range):
        raise IndexError(f"Invalid array ID {array_id}. Must be between 0 and {len(intervals_range) - 1}.")
      start = intervals_range[array_id].start
      end = intervals_range[array_id + 1].start if array_id + 1 < len(intervals_range) else intervals_range.end
      output_path = output_path / f"{start.strftime('%Y%m%d')}-{end.strftime('%Y%m%d')}"
    else:
      raise ValueError("Configs top table should contain the 'slurm_array' table specifying the 'step' key when "
                       "downloading using SLURM arrays. See --help for more information.")
  temporary_config = configs.get('temporary', {})
  date_intervals = DateIntervalsRange(start=start, end=end, step=temporary_config.get('step', None))

  return date_intervals, output_path


@click.group()
def cli():
  pass


@cli.command()
@click.argument("config_path",
                required=True,
                type=click.Path(path_type=pathlib.Path, file_okay=True, readable=True))
@click.argument("output_path",
                required=True,
                type=click.Path(path_type=pathlib.Path, dir_okay=True, writable=True))
@click.option("--start",
              help="Start of the date interval to download",
              default=None,
              type=click.DateTime())
@click.option("--end",
              help="End of the date interval to download",
              default=None,
              type=click.DateTime())
@click.option("--array-id",
              help="ID of the SLURM array to download",
              default=None,
              type=int)
@click.option("--overwrite/--no-overwrite",
              help="Whether to overwrite existing outputs",
              default=False,
              is_flag=True)
@click.option("--progress/--no-progress",
              "progress",
              help="Whether to display a progress bar",
              default=False,
              is_flag=True)
@click.option("--dry-run",
              default=False,
              is_flag=True)
@click.option('--log-level',
              default='info',
              type=click.Choice(['debug', 'info', 'warning', 'error', 'critical'], case_sensitive=False))
def download(
    config_path: pathlib.Path,
    output_path: pathlib.Path,
    array_id: int | None = None,
    start: datetime | None = None,
    end: datetime | None = None,
    dry_run: bool = False,
    progress: bool = False,
    logger: logging.Logger | None= None,
    log_level: str = 'info',
    overwrite: bool = False):

  if logger is None:
    logger = logging.getLogger(__name__)
    logging.basicConfig(format='%(levelname)s - %(asctime)s: %(message)s',
                        datefmt='%Y-%m-%dT%H:%M:%S',
                        level=getattr(logging, log_level.upper()))

  # Open the configuration file and load the TOML configs.
  configs = Configs.read(config_path)

  dataset_type = configs.get('type', None)
  if dataset_type is None:
    raise ValueError("Configs TOML top table should contain the 'type' key.")

  # Check if time interval options are valid, and get the date intervals to process.
  if dataset_type == 'static':
    date_intervals = (None,)
    logging.info("Processing static dataset")
  elif dataset_type == 'timeseries':
    date_intervals, output_path = _parse_timeseries_arguments(configs,
                                                              output_path,
                                                              start=start,
                                                              end=end,
                                                              array_id=array_id)
    logging.info(f"Processing timeseries {date_intervals}")
  else:
    raise ValueError(f"The 'type' key value must be one of 'static' or 'timeseries'.")

  # If destination exists and should not overwrite, raise and exit.
  if output_path.exists() and not overwrite:
    raise ValueError(f"Output destination {output_path} already exists")

  if provider_name := configs.get('provider', {}):
    if provider_name in ProvidersRegistry:
      provider = ProvidersRegistry[provider_name](progress=progress, log_level=log_level, client_logger=logger)
    else:
      raise ValueError(f"The 'provider' key value must be one of {ProvidersRegistry.keys()}.")
  else:
    raise ValueError("Configs TOML top table should contain the 'provider' key.")

  postprocess_configs = configs.get('postprocess')
  postprocess = Process(steps=postprocess_configs)

  with TempStore() as temporary_store:
    is_first_fragment = True
    # For each date_interval: download, postprocess, and append the dataset to a temporary Zarr
    for date_interval in date_intervals:

      def _download_step(**kwargs):
        if date_interval is not None:
          logging.info(f"Processing step {date_interval}")
        # When downloading from Copernicus Marine Data Store or Climate Data Store, the typical case is a large dataset,
        # spanning a long time period, with several sets of variables in different datasets (bio, phys, etc.),
        # which needs to be downloaded one piece at a time. Hence, `datasets` array values in the TOML configuration file
        # represent different pieces of the same dataset.
        # When downloading from CDS, a temporary directory is needed to store partial netCDF files.
        with tempfile.TemporaryDirectory() as tempdir:
          fragment_datasets = []
          for ds_conf in configs['datasets']:
            ds = provider.open_dataset(date_interval, tempdir, **ds_conf)
            ds = postprocess(ds)
            fragment_datasets.append(ds)
          fragment = xr.merge(fragment_datasets, join='exact')
          # fragment = fragment.chunk(**{dim: -1 for dim in fragment.dims})
          with bar(progress):
            # Requires that fragment fits into memory
            fragment = fragment.compute() if not dry_run else fragment
        # Here we save the fragment to a temporary Zarr store using default parameters.
        # Being fragment underlying data numpy arrays, the chunk size will be determined by zarr,
        # which tend to produce small chunks (1MB without compression).
        # This might be optimized, but as we are saving it fast local storage (SSD), it is probably fine.
        for var in fragment.data_vars:
          fragment[var].encoding['compressor'] = None
        logging.info(f"Saving temporary dataset to {temporary_store.path}")
        if not dry_run:
          fragment.to_zarr(store=temporary_store, **kwargs)

      if is_first_fragment:
        _download_step(mode='w')
        is_first_fragment = False
      else:
        _download_step(mode='a-', append_dim='time')

    save_configs = configs.get('save', {})
    output_suffix = ''.join(output_path.suffixes + ['.zip'])
    output_path = output_path.absolute().with_suffix(output_suffix)
    # Get the parent directory and create it if it doesn't exist
    output_parent_dir = output_path.parent
    if not dry_run:
        output_parent_dir.mkdir(parents=True, exist_ok=True)
    logging.info(f"Saving dataset to {output_path} with {save_configs}")
    if not dry_run:
      # Load the temporary Zarr, eventually rechunk and save to final destination.
      dataset = xr.open_zarr(temporary_store, overwrite_encoded_chunks=True)
      if rechunk_conf := save_configs.pop('chunk', {}):
        dataset = dataset.chunk(**rechunk_conf)
        # see: https://github.com/pydata/xarray/issues/4380
        for var in dataset.data_vars:
            del dataset[var].encoding['chunks']
      if compressor_conf := save_configs.pop('compressor', {}):
        for var in dataset.data_vars:
          dataset[var].encoding['compressor'] = Blosc(**compressor_conf)
      store = ZipStore(path=str(output_path), mode='w', compression=0, allowZip64=True)
      with bar(progress):
        dataset.to_zarr(store=store, compute=True, **save_configs)


@cli.command()
@click.argument("config_path",
                required=True,
                type=click.Path(path_type=pathlib.Path, file_okay=True, readable=True))
@click.option("--sbatch-flag/--no-sbatch-flag",
              "sbatch_flag",
              default=False,
              is_flag=True)
def slurm_array_range(config_path: pathlib.Path,
                      start: datetime | None = None,
                      end: datetime | None = None,
                      sbatch_flag: bool = False):

  with config_path.open('rb') as file:
    configs = tomllib.load(file)

  array_id = 0
  slurm_array = []
  while True:
    try:
      date_interval, _ = _parse_timeseries_arguments(configs, pathlib.Path(), start=start, end=end, array_id=array_id)
      slurm_array.append(date_interval)
      array_id += 1
    except IndexError:
      break

  if sbatch_flag:
    print(f"--array=0-{len(slurm_array) - 1}")
  else:
    pprint.pprint(slurm_array, compact=True)


@cli.command()
@click.argument("config_path",
                required=True,
                type=click.Path(path_type=pathlib.Path, file_okay=True, readable=True))
@click.argument("data_dir",
                required=True,
                type=click.Path(path_type=pathlib.Path, dir_okay=True, readable=True))
@click.argument("output_path",
                required=True,
                type=click.Path(path_type=pathlib.Path, dir_okay=True, writable=True))
@click.option("--array-id",
              help="ID of the SLURM array to download",
              default=None,
              type=int)
@click.option("--start",
              "start_date",
              help="Start of the date interval to download",
              default=None,
              type=click.DateTime())
@click.option("--end",
              "end_date",
              help="End of the date interval to download",
              default=None,
              type = click.DateTime())
@click.option("--overwrite/--no-overwrite",
              help="Whether to overwrite existing outputs",
              default=False,
              is_flag=True)
@click.option("--progress/--no-progress",
              "progress",
              help="Whether to display a progress bar",
              default=False,
              is_flag=True)
@click.option('--log-level',
              default='info',
              type=click.Choice(['debug', 'info', 'warning', 'error', 'critical'], case_sensitive=False))
@click.option("--debug/--no-debug",
              "debug",
              help="Use synchronous Dask scheduler",
              default=False,
              is_flag=True)
def merge(
    config_path: pathlib.Path,
    data_dir: pathlib.Path,
    output_path: pathlib.Path,
    array_id: int | None = None,
    start_date: datetime | None = None,
    end_date: datetime | None = None,
    progress: bool = False,
    log_level: str = 'info',
    overwrite: bool = False,
    debug: bool = False):
  """
  Merges multiple datasets into a single dataset.

  The function reads dataset configurations, applies necessary preprocessing steps,
  and saves the merged dataset to a Zarr store.
  """

  if debug:
    dask.config.set(scheduler='synchronous')

  logging.basicConfig(format='%(levelname)s - %(asctime)s: %(message)s',
                      datefmt='%Y-%m-%dT%H:%M:%S',
                      level=getattr(logging, log_level.upper()))

  # Open the configuration file and load the TOML configs.
  configs = Configs.read(config_path)

  # If destination exists and should not overwrite, raise and exit.
  if output_path.exists() and not overwrite:
    raise ValueError(f"Output destination {output_path} already exists")

  date_intervals, output_path = _parse_timeseries_arguments(configs, output_path,
                                                            start=start_date, end=end_date, array_id=array_id)

  # The `date_intervals` return value of `_parse_timeseries_arguments` is used within the download command to iterate
  # through dates segments, which when joined contains all dates between date_intervals.start and
  # date_intervals.last_valid_date, i.e. date_intervals.end - date_intervals.delta (included).
  # Here we process all valid dates at ones, hence end_date is date_intervals.last_valid_date, and **not** date_intervals.end.
  start_date = date_intervals.start
  end_date = date_intervals.last_valid_date

  def reader(path, **kwargs):
    logging.info(f"Reading {path}")
    if path.is_dir():
      path = list(path.glob('*.zip'))
    else:
      path = path.with_suffix('.zip')
    # noinspection PyTypeChecker
    ds = xr.open_mfdataset(path, engine='zarr', **kwargs)
    return ds

  datasets = []
  for dataset_conf in configs.get('datasets', []):
    if mask_conf := dataset_conf.get('mask'):
      postprocess_mask_conf = mask_conf.get('postprocess')
      mask_var = mask_conf['variable']

      @check_coordinates
      @check_values(variables=[mask_var])
      def mask_reader(path, **kwargs):
        ds = reader(path, **kwargs)
        postprocess = Process(steps=postprocess_mask_conf)
        ds = postprocess(ds)
        return ds
      mask_ds = mask_reader(data_dir / mask_conf['file'])
      mask_da = mask_ds[mask_var]
    else:
      mask_ds = None
      mask_da = None

    postprocess_confs = dataset_conf.get('postprocess', {})

    @check_coordinates
    @check_date_range(start_date=start_date, end_date=end_date)
    @check_values(mask=mask_da)
    def dataset_reader(path, **kwargs):
      # FIXME: we should have really done most of processing at download time...
      ds = reader(path, **kwargs)
      postprocess = Process(steps=postprocess_confs, mask=mask_da)
      ds = postprocess(ds)
      if mask_ds is not None:
        ds = xr.merge([ds, mask_ds])
      return ds

    datasets.append(dataset_reader(data_dir / dataset_conf['file'], **dataset_conf.get('kwargs', {}))),

  dataset = xr.merge(datasets, join='inner')
  dataset = dataset.sel(time=slice(start_date, end_date))

  # Save the dataset in a Zarr using sensible chunking and compression
  save_configs = configs.get('save', {})
  output_suffix = ''.join(output_path.suffixes + ['.zip'])
  output_path = output_path.absolute().with_suffix(output_suffix)

  # Get the parent directory and create it if it doesn't exist
  output_parent_dir = output_path.parent
  output_parent_dir.mkdir(parents=True, exist_ok=True)

  logging.info(f"Saving merged dataset to {output_path} with {save_configs}")
  if rechunk_conf := save_configs.pop('chunk', {}):
    dataset = dataset.chunk(**rechunk_conf)
    # see: https://github.com/pydata/xarray/issues/4380
    for var in dataset.data_vars:
      if dataset[var].encoding and dataset[var].encoding.get('chunks'):
        del dataset[var].encoding['chunks']
  if compressor_conf := save_configs.pop('compressor', {}):
    for var in dataset.data_vars:
      dataset[var].encoding['compressor'] = Blosc(**compressor_conf)
  else:
    for var in dataset.data_vars:
      dataset[var].encoding['compressor'] = None
  store = ZipStore(path=str(output_path), mode='w', compression=0, allowZip64=True)
  with bar(progress):
    dataset.to_zarr(store=store, compute=True, **save_configs)


@cli.command()
@click.argument("config_path",
                required=True,
                type=click.Path(path_type=pathlib.Path, file_okay=True, readable=True))
@click.argument("output_path",
                required=True,
                type=click.Path(path_type=pathlib.Path, dir_okay=True, writable=True))
@click.option("--data-prefix",
              "data_path_prefix",
              help="Prefix to prepend to data paths.",
              default=None,
              type=click.Path(path_type=pathlib.Path, dir_okay=True, readable=True))
@click.option("--overwrite/--no-overwrite",
              help="Whether to overwrite existing outputs",
              default=False,
              is_flag=True)
@click.option("--progress/--no-progress",
              "progress",
              help="Whether to display a progress bar",
              default=False,
              is_flag=True)
@click.option('--log-level',
              default='info',
              type=click.Choice(['debug', 'info', 'warning', 'error', 'critical'], case_sensitive=False))
@click.option("--debug/--no-debug",
              "debug",
              help="Use synchronous Dask scheduler",
              default=False,
              is_flag=True)
def normalization(
    config_path: pathlib.Path,
    output_path: pathlib.Path,
    data_path_prefix: pathlib.Path | None = None,
    progress: bool = False,
    log_level: str = 'info',
    overwrite: bool = False,
    debug: bool = False):
  """
  Computes normalization artifacts for the target dataset (location/scale per variable, and residuals' scale) using precomputed statistics listed in the configuration, and saves them as a DataTree Zarr.
  """

  if debug:
    dask.config.set(scheduler='synchronous')

  logging.basicConfig(format='%(levelname)s - %(asctime)s: %(message)s',
                      datefmt='%Y-%m-%dT%H:%M:%S',
                      level=getattr(logging, log_level.upper()))

  # Open the configuration file and load the TOML configs.
  configs = Configs.read(config_path)

  # If destination exists and should not overwrite, raise and exit.
  if output_path.exists() and not overwrite:
    raise ValueError(f"Output destination {output_path} already exists")

  def open_dataset(name: str) -> xr.Dataset:
    path = pathlib.Path(configs[name])
    path = path if data_path_prefix is None else data_path_prefix / path
    logging.info(f"Loading {name} from {path}")
    return xr.open_dataset(path, engine='zarr')

  # Load dataset and statistics
  dataset = open_dataset('dataset')
  mean = open_dataset('mean')
  std = open_dataset('std')
  diff_std = open_dataset('diff_std')
  assert set(mean.data_vars) == set(std.data_vars) == set(diff_std.data_vars), \
    "The precomputed stats do not contain the same variables."
  assert np.array_equal(mean.coords, std.coords) and np.array_equal(mean.coords, diff_std.coords), \
    "The precomputed stats do not contain the same dimensions."

  # Identify variables
  boolean_variables = set(var for (var, data) in dataset.data_vars.items() if np.isdtype(data.dtype, np.bool_))
  missing_normalization_variables = set(dataset.data_vars) - set(mean.data_vars)

  # Build working datasets
  normal_vars = list(missing_normalization_variables - boolean_variables)
  normalization_dataset = dataset[normal_vars] if normal_vars else xr.Dataset()
  template_dataset = dataset[list(missing_normalization_variables)] if missing_normalization_variables else xr.Dataset()

  # Compute per-variable scale/location for non-boolean vars and add defaults for booleans
  if normalization_dataset.data_vars:
    inputs_scale = normalization_dataset.max(skipna=True) - normalization_dataset.min(skipna=True)
    inputs_location = normalization_dataset.min(skipna=True) / inputs_scale
  else:
    inputs_scale = xr.Dataset()
    inputs_location = xr.Dataset()

  def get_typed_value(value):
    return np.float64(value).astype(configs['dtype'])

  inputs_location = xr.merge([
    inputs_location,
    xr.Dataset(data_vars={var: ((), get_typed_value(0.0)) for var in boolean_variables})
  ])
  inputs_scale = xr.merge([
      inputs_scale,
      xr.Dataset(data_vars={var: ((), get_typed_value(1.0)) for var in boolean_variables})
  ])

  # Extend the level-wise stats with per-variable location/scale
  inputs_location = xr.merge([mean, xr.zeros_like(template_dataset) + inputs_location])
  inputs_scale = xr.merge([std, xr.ones_like(template_dataset) * inputs_scale])

  common_coordinates = set(dataset.coords) & set(inputs_location.coords)
  common_coordinates = list(common_coordinates)
  # Build datatree with coordinates from the merged dataset
  dt = xr.DataTree.from_dict({
      '/': xr.Dataset(coords={name: dataset.coords[name] for name in common_coordinates}),
      '/inputs/location': inputs_location.drop_vars(common_coordinates),
      '/inputs/scale': inputs_scale.drop_vars(common_coordinates),
      '/residuals/scale': diff_std.drop_vars(common_coordinates),
  })

  # Save the DataTree to a Zarr store
  logging.info(f"Saving normalization datatree to {output_path}")
  with bar(progress):
    dt.to_zarr(store=output_path, mode="w" if overwrite else "w-", compute=True)


@cli.command()
@click.argument("input_path",
                required=True,
                type=click.Path(path_type=pathlib.Path, dir_okay=True, readable=True))
@click.argument("output_path",
                required=True,
                type=click.Path(path_type=pathlib.Path, dir_okay=True, writable=True))
@click.option("--time-dim",
              help="Time dimension name used in the input dataset",
              default="time")
@click.option("--start",
              "start_date",
              help="Override start of the date interval",
              default=None,
              type=click.DateTime())
@click.option("--end",
              "end_date",
              help="Override end of the date interval",
              default=None,
              type=click.DateTime())
@click.option("--chunks",
              default=None,
              show_default=True,
              type=DictParamType(),
              help="String containing chunking specs used when reading.")
@click.option("--compressor-name",
              "cname",
              default="lz4",
              show_default=True,
              help="Name of the compressor to use.")
@click.option("--compressor-level",
              "clevel",
              default=1,
              show_default=True,
              help="Compressor level to use.")
@click.option("--overwrite/--no-overwrite",
              help="Whether to overwrite existing outputs",
              default=False,
              is_flag=True)
@click.option("--local/--no-local",
              default=False,
              help="Whether to use Dask LocalCluster.",
              show_default=True)
@click.option("--debug/--no-debug",
              default=False,
              help="Use synchronous Dask scheduler",
              is_flag=True)
@click.option('--log-level',
              default='info',
              type=click.Choice(['debug', 'info', 'warning', 'error', 'critical'], case_sensitive=False))
def unpack(
    input_path: pathlib.Path,
    output_path: pathlib.Path,
    time_dim: str = 'time',
    start_date: datetime | None = None,
    end_date: datetime | None = None,
    chunks: dict | None = None,
    cname: str = "lz4",
    clevel: int = 1,
    overwrite: bool = False,
    local: bool = False,
    debug: bool = False,
    log_level: str = 'info'):
  """
  Unpack a collection of zipped Zarr datasets into a single directory Zarr store.

  The command reads zipped Zarr fragments with xarray.open_mfdataset(engine='zarr'),
  optionally slices the time range, rechunks, and saves to a DirectoryStore at the
  given output path.
  """

  logger = logging.getLogger(__name__)
  logging.basicConfig(format='%(levelname)s - %(asctime)s: %(message)s',
                      datefmt='%Y-%m-%dT%H:%M:%S',
                      level=getattr(logging, log_level.upper()))
  client = get_client(logger=logger, debug=debug, local=local)

  # If destination exists and should not overwrite, raise and exit.
  if output_path.exists() and not overwrite:
    raise ValueError(f"Output destination {output_path} already exists")

  def build_paths(path: pathlib.Path):
    if path.is_dir():
      return sorted(path.glob('*.zip'))
    else:
      return [path.with_suffix('.zip')]

  paths = build_paths(input_path)
  logging.info(f"Reading {len(paths)} zipped Zarrs from {input_path}")

  # noinspection PyTypeChecker
  dataset = open_mfdataset(paths, chunks=chunks)
  if not valid_time_coordinate(dataset, time_dim=time_dim):
    raise ValueError(f"Time coordinate {time_dim} is not valid (contains duplicates or missing dates)")
  dataset = dataset.sel(time=slice(start_date, end_date))
  dataset = dataset.chunk({dim: (1 if dim == 'time' else -1) for dim in dataset.dims})

  output_path = output_path.absolute()
  # Ensure parent directory exists
  output_path.parent.mkdir(parents=True, exist_ok=True)
  logging.info(f"Saving unpacked dataset to directory Zarr at {output_path}")

  save_to_zarr(dataset, output_path, overwrite=overwrite, compressor_kwargs=dict(cname=cname, clevel=clevel))

  client.close()


if __name__ == '__main__':
  cli()