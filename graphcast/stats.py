import logging
import pathlib
from contextlib import nullcontext
from typing import Union

import click
import dask
import xarray as xr
from dask.diagnostics import ProgressBar


def _bar(progress: bool):
  return ProgressBar() if progress else nullcontext()


def _drop_static_vars(ds: xr.Dataset, time_dim: str) -> xr.Dataset:
  ds = ds.drop_vars([name for (name, var) in ds.data_vars.items() if time_dim not in var.dims])
  return ds


def _open_dataset(path: pathlib.Path, time_dim: str = "time") -> xr.Dataset:
  """
  Open a dataset from a single Zarr file/store or a directory containing multiple Zarr zip files.

  - If `path` is a directory with one or more .zip files, open all of them via xarray.open_mfdataset(engine='zarr').
  - In all other cases, open it via xarray.open_dataset(engine='zarr').

  Returns a xarray.Dataset filtered to only data variables that include the provided time dimension.
  """
  path = pathlib.Path(path)
  if not path.exists():
    raise ValueError(f"Input path {path} does not exist")

  if path.is_dir():
    zip_files = sorted(p for p in path.glob("*.zip"))
    if zip_files:
      # noinspection PyTypeChecker
      ds = xr.open_mfdataset([str(p) for p in zip_files],
                             preprocess=lambda ds: _drop_static_vars(ds, time_dim),
                             engine="zarr",
                             combine="by_coords")
      return ds

  ds = xr.open_dataset(str(path), engine="zarr")
  ds = _drop_static_vars(ds, time_dim)
  return ds


@click.group()
def cli():
  pass


@cli.command()
@click.argument("input_path",
                required=True,
                type=click.Path(path_type=pathlib.Path, file_okay=True, dir_okay=True, exists=True, readable=True))
@click.argument("output_dir",
                required=True,
                type=click.Path(path_type=pathlib.Path, file_okay=False, dir_okay=True, writable=True))
@click.argument("basename_prefix",
                required=True,
                type=str)
@click.option("--climatology/--no-climatology",
              default=True,
              help="Whether to compute the climatology.",
              show_default=True)
@click.option("--exact/--no-exact",
              "exact",
              default=False,
              help="Whether to compute the exact average.",
              show_default=True)
@click.option("--time-dim",
              default="time",
              help="Name of the time dimension to average over.",
              show_default=True)
@click.option("--climatology-dim",
              default="dayofyear",
              show_default=True,
              help="Name of the climatology dimension (default 'dayofyear').")
@click.option("--skipna/--no-skipna",
              default=False,
              help="Whether to skip NaNs when averaging.",
              show_default=True)
@click.option("--overwrite/--no-overwrite",
              default=False,
              is_flag=True,
              help="Whether to overwrite existing outputs.")
@click.option("--progress/--no-progress",
              "progress",
              default=False,
              is_flag=True,
              help="Whether to display a progress bar.")
@click.option("--log-level",
              default='info',
              type=click.Choice(['debug', 'info', 'warning', 'error', 'critical'], case_sensitive=False),
              show_default=True)
def compute(input_path: pathlib.Path,
            output_dir: pathlib.Path,
            basename_prefix: str,
            climatology: bool = True,
            exact: bool = False,
            time_dim: str = "time",
            climatology_dim: str = "dayofyear",
            skipna: bool = False,
            progress: bool = False,
            log_level: str = "info",
            overwrite: bool = False):
  """
  Compute the mean, std, and diff std over time.

  Parameters
  ----------
  input_path : pathlib.Path
      Path to the input Zarr, or directory of zipped Zarrs.
  output_dir : pathlib.Path
      Path to the directory of output stats.
  basename_prefix : str
      Prefix to prepend to the output stats filenames.
  climatology : bool, default True
      Whether to compute the daily climatology by grouping data by day of year.
  time_dim : str, default "time"
      Name of the time dimension to reduce or group by.
  climatology_dim : str, default "dayofyear"
      Name to use for the climatology dimension (replaces the default 'dayofyear').
  skipna : bool, default False
      Whether to skip NaNs when computing the mean.
  progress : bool, default False
      Whether to display a dask progress bar during writing.
  log_level : {"debug","info","warning","error","critical"}, default "info"
      Logging verbosity.
  overwrite : bool, default False
      Whether to overwrite the output_path if it already exists.
  """

  # Configure logging
  logger = logging.getLogger(__name__)
  logging.basicConfig(
    format='%(levelname)s - %(asctime)s: %(message)s',
    datefmt='%Y-%m-%dT%H:%M:%S',
    level=getattr(logging, log_level.upper()),
  )

  if not input_path.exists():
    raise ValueError(f"Input path {input_path} does not exist")

  logger.info(f"Opening input dataset from {input_path}")
  dataset = _open_dataset(input_path, time_dim=time_dim)

  logger.info(f"Computing averages over dimension '{time_dim}' (skipna={skipna})")
  stats = {}

  # Computing the climatology beforehand could be a source of optimization for large datasets, but it introduces a small numerical error.
  # Say you have a collection of values {x_i} and labels {l_i} so that each label corresponds to multiple values.
  # Then you can compute the mean of the whole collection x_mean, or the mean of the averages for each label x_clim_mean.
  # If the number of values for each label is the same, then the two quantities are strictly equal. But it's not true in general.
  # Indeed, in the case of a daily climatology, the two would differ because of leap years, which would introduce a relative error of
  # the order of 1/365. Finally, the dataset might not start on the 1st of January or end before the 31 of December,
  # which would further distort the results. However, the quantities computed here are used to standardize the input features, and
  # therefore such approximations are reasonably acceptable.
  if climatology:
    time_coord = dataset[time_dim]
    if not (hasattr(time_coord, 'dt') and hasattr(time_coord.dt, 'dayofyear')):
      raise TypeError(
        f"Time coordinate '{time_dim}' must be datetime-like to compute climatology. "
        f"Found dtype={time_coord.dtype}."
      )
    logger.info(f"Computing daily climatology grouped by '{time_dim}.dayofyear' (skipna={skipna})")
    # Here we handle leap years.
    # See: https://github.com/pydata/xarray/issues/1844#issuecomment-417855365
    clim_ds = dataset.convert_calendar('365_day')
    clim_ds = clim_ds.groupby(f"{time_dim}.dayofyear")
    clim_ds = clim_ds.mean(keep_attrs=True, skipna=skipna)
    # Optionally rename the climatology dimension.
    if climatology_dim != 'dayofyear' and 'dayofyear' in clim_ds.dims:
      clim_ds = clim_ds.rename({'dayofyear': climatology_dim})
    stats["climatology"] = clim_ds

  if climatology and not exact:
    stats["mean"] = stats["climatology"].mean(dim=climatology_dim, keep_attrs=True, skipna=skipna)
  else:
    stats["mean"] = dataset.mean(dim=time_dim, keep_attrs=True, skipna=skipna)

  # There is no easy way to pass the mean to std, one should open an issue to XArray
  def std_from_mean(ds: xr.Dataset, mean: Union[xr.Dataset, float], **kwargs) -> xr.Dataset:
    ds = ds - mean
    var = (ds * ds).mean(**kwargs)
    return xr.ufuncs.sqrt(var)

  stats["std"] = std_from_mean(dataset, stats["mean"], dim=time_dim, keep_attrs=True, skipna=skipna)
  # Given a sequence {x_i}_{i = 1, ..., N} the mean of the diffs {x_i - x_{i-1}} is proportional to the sum of a
  # telescopic series and equal to (x_N - x_1) / N, which becomes negligible for large N.
  # Also, Graphcast computes the increment between the present and next system state rescaled by diff_std.
  # This accounts to standardizing the targets. Whatever the rationale, one can reasonably approximate here the mean with zero.
  if exact:
    stats["diff_std"] = dataset.diff(dim=time_dim).std(dim=time_dim, keep_attrs=True, skipna=skipna)
  else:
    stats["diff_std"] = std_from_mean(dataset.diff(dim=time_dim), 0.0, dim=time_dim, keep_attrs=True, skipna=skipna)

  # Ensure parent dir exists
  delayed_saves = []
  output_dir.mkdir(parents=True, exist_ok=True)
  for (stat_name, stat_ds) in stats.items():
    stat_ds = stat_ds.chunk({dim: -1 for dim in stat_ds.dims})
    for var in stat_ds.data_vars:
      # Some pipelines set encodings that can conflict with saving
      if 'chunks' in stat_ds[var].encoding:
        del stat_ds[var].encoding['chunks']
      # Do not force compressors by default; leave as None for speed/compatibility
      stat_ds[var].encoding['compressor'] = None
    output_path = output_dir / f"{basename_prefix}_{stat_name}.zip"
    output_path = output_path.absolute()
    if output_path.exists() and not overwrite:
      raise ValueError(f"Output destination {output_path} already exists")
    logger.info(f"Saving {stat_name} dataset to {output_path}")
    save_stat = stat_ds.to_zarr(store=str(output_path), compute=False, mode='w')
    delayed_saves.append(save_stat)

  with _bar(progress):
    dask.compute(*delayed_saves)


if __name__ == "__main__":
  cli()
