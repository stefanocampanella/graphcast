# SPDX-FileCopyrightText: 2026 Stefano Campanella
# SPDX-License-Identifier: MIT
"""Write the standardized anomalies of a dataset, its mask and the tendency statistics to local Zarr stores.

Time-dependent variables become anomaly z-scores with respect to a 365-day climatology and anomaly standard deviation,
static variables are min-max normalized, missing values are filled with 0 and all variables are float32. Training
reads these stores (dataloader.StandardizedAnomalySource), adding the batch dimension and relative time.
"""

import logging
import os
import pathlib
import time
from collections.abc import Hashable

import botocore.session
import click
import dask
import dask.array
import humanize
import numcodecs
import numpy as np
import tensorstore
import xarray as xr
import xarray_tensorstore
from dask.diagnostics import ProgressBar
from zarr.codecs import BloscCodec, BytesCodec, ShardingCodec, TransposeCodec

from graphcast import cli_utils
from graphcast.dataloader import get_tensorstore_context
from graphcast.download_config import CompressorConfig, DownloadConfig, MaskDownloadConfig, TendencyStdConfig

logger = logging.getLogger(__name__)

# Dimensions with one chunk per index; arrays are whole along the others.
_CHUNKED_DIMS = ("time", "depth")

# Day of year of February 29 (1-based).
_FEB_29 = 60


def climatology_index(time: xr.DataArray) -> np.ndarray:
  """Return the index of each date in a 365-day climatology with time coordinates 0, ..., 364.

  In leap years, February 29 takes the values of February 28 and later days are shifted back by one.
  """
  dayofyear = time.dt.dayofyear.values
  after_feb_28 = time.dt.is_leap_year.values & (dayofyear >= _FEB_29)
  return dayofyear - 1 - after_feb_28


def get_aws_credentials() -> dict:
  """Return the TensorStore AWS credentials matching those of fsspec (botocore), resolved from the environment.

  TensorStore cannot run a profile's credential_process (e.g. AWS_PROFILE with a signing helper): the keys resolved by
  botocore are exported to the environment of this process, without refreshing them. FSSPEC_S3_ANON=true, or no
  credentials, selects anonymous access.
  """
  credentials = None
  if os.environ.get("FSSPEC_S3_ANON", "").lower() not in ("1", "true"):
    credentials = botocore.session.Session().get_credentials()
  if credentials is None:
    logger.info("Reading S3 anonymously")
    return {"type": "anonymous"}
  credentials = credentials.get_frozen_credentials()
  os.environ["AWS_ACCESS_KEY_ID"] = credentials.access_key
  os.environ["AWS_SECRET_ACCESS_KEY"] = credentials.secret_key
  if credentials.token:
    os.environ["AWS_SESSION_TOKEN"] = credentials.token
  else:
    os.environ.pop("AWS_SESSION_TOKEN", None)
  return {"type": "environment"}


def open_input(path: str, context: tensorstore.Context) -> xr.Dataset:
  """Open a Zarr store with consolidated metadata, given as a local path or a URL (e.g. s3://bucket/path).

  Data variables are read with TensorStore, undecoded (e.g. masks are integers); coordinates are decoded by xarray.
  Metadata of remote stores is read with fsspec, data with TensorStore: S3 credentials come from the environment
  (e.g. AWS_PROFILE, or FSSPEC_S3_ANON=true for anonymous access) and from the context (get_aws_credentials).
  """
  logger.info(f"Opening {path}")
  return xarray_tensorstore.open_zarr(path, context=context, mask_and_scale=False, consolidated=True)


def _write_chunks(variable: xr.Variable, configs: DownloadConfig) -> dict[str, int]:
  """Timesteps per Dask chunk: a v3 shard, also used for v2 to keep the graph small. Other dimensions are whole."""
  if "time" not in variable.dims:
    return {}
  return {"time": configs.depth_time_shard if "depth" in variable.dims else configs.time_shard}


def _variable_masks(dataset: xr.Dataset, configs: DownloadConfig) -> dict[Hashable, xr.DataArray]:
  """Boolean mask of each masked variable (read into memory); a mask with depth masks variables without it by level 0."""
  masks = {}
  for mask_name, names in configs.masks.items():
    if mask_name not in dataset:
      raise ValueError(f"Mask {mask_name!r} not in the dataset")
    mask = xarray_tensorstore.read(dataset[mask_name]).compute().astype(bool).drop_encoding()
    for name in names:
      if name not in dataset:
        raise ValueError(f"Variable {name!r} of mask {mask_name!r} not in the dataset")
      if "depth" in mask.dims and "depth" not in dataset[name].dims:
        masks[name] = mask.isel(depth=0, drop=True)
      else:
        masks[name] = mask
  return masks


def _check_finite(values: xr.DataArray, name: Hashable) -> xr.DataArray:
  """Fail, when computed, on values that are not finite: missing data must be outside the variable's mask."""

  def check(block: np.ndarray) -> np.ndarray:
    if not np.isfinite(block).all():
      raise ValueError(f"{name}: non-finite values inside its mask (or without a mask); check the inputs and masks")
    return block

  return xr.apply_ufunc(check, values, dask="parallelized", output_dtypes=[values.dtype])


def _anomaly_std_floor(anomaly_std: xr.DataArray, inside: xr.DataArray | None) -> xr.DataArray:
  """Smallest positive anomaly_std inside the mask, per depth level (one level at a time, to bound memory).

  As a floor it leaves the in-sample z-scores unchanged and keeps out-of-sample values with a zero anomaly_std finite.
  Levels without positive values take the smallest value of the other levels (1 if there is none).
  """

  def smallest_positive(std: xr.DataArray, mask: xr.DataArray | None) -> float:
    values = xarray_tensorstore.read(std).values
    positive = values > 0
    if mask is not None:
      positive &= np.broadcast_to(mask.transpose(*std.dims[1:]).values, values.shape)
    return float(values[positive].min()) if positive.any() else np.nan

  if "depth" not in anomaly_std.dims:
    floors = np.array(smallest_positive(anomaly_std, inside))
    dims, coords = (), {}
  else:
    floors = np.array(
      [
        smallest_positive(
          anomaly_std.isel(depth=level),
          None if inside is None else inside.isel(depth=level) if "depth" in inside.dims else inside,
        )
        for level in range(anomaly_std.sizes["depth"])
      ]
    )
    dims, coords = ("depth",), {"depth": anomaly_std["depth"]}
  if np.isnan(floors).all():
    floors = np.ones_like(floors)
  floors = np.where(np.isnan(floors), np.nanmin(floors), floors)
  return xr.DataArray(floors.astype(np.float32), dims=dims, coords=coords)


def _min_max(values: xr.DataArray, inside: xr.DataArray | None) -> xr.DataArray:
  """Min-max normalize a static field over the points inside its mask; integer fields (masks) are only cast."""
  if not np.issubdtype(values.dtype, np.floating):
    return values.astype(np.float32)
  values = xarray_tensorstore.read(values).compute()
  inside_values = values if inside is None else values.where(inside)
  minimum = inside_values.min()
  span = inside_values.max() - minimum
  return ((values - minimum) / span.where(span != 0, 1)).astype(np.float32)


def standardize(
  dataset: xr.Dataset,
  climatology: xr.Dataset,
  anomaly_std: xr.Dataset,
  configs: DownloadConfig,
) -> xr.Dataset:
  """Return the anomaly z-scores of the time-dependent variables and the normalized static variables (lazily).

  - z = (x - climatology) / max(anomaly_std, floor), with the floor of `_anomaly_std_floor`; z = 0 where x equals the
    climatology within `equal_tolerance` float32 epsilons, whatever the anomaly_std (constant series).
  - Outside `reference_period`, if set, |z| is capped at its largest value inside it, per variable and level.
  - Static variables, including `static_variables` (from their first timestep), are min-max normalized.
  - Values outside the variable's mask (`masks`) are 0; non-finite values elsewhere are an error.
  """
  static = set(configs.static_variables)
  if unknown := sorted(static - set(dataset.data_vars)):
    raise ValueError(f"Static variables not in the dataset: {unknown}")
  missing = [
    name
    for name, values in dataset.data_vars.items()
    if "time" in values.dims and name not in static and (name not in climatology or name not in anomaly_std)
  ]
  if missing:
    raise ValueError(f"Time-dependent variables without climatology or anomaly_std: {missing}")
  masks = _variable_masks(dataset, configs)

  # Align climatology and anomaly_std with the timeseries; the lazy indexing reads only the selected days.
  index = climatology_index(dataset["time"])
  aligned_climatology = climatology.isel(time=index).assign_coords(time=dataset["time"])
  aligned_anomaly_std = anomaly_std.isel(time=index).assign_coords(time=dataset["time"])
  if configs.reference_period is None:
    in_period = None
  else:
    start, end = (np.datetime64(day, "ns") for day in configs.reference_period)
    in_period = (dataset["time"] >= start) & (dataset["time"] <= end + np.timedelta64(1, "D") - np.timedelta64(1, "ns"))
    if bool(in_period.all()):
      in_period = None

  standardized = {}
  for name, values in dataset.data_vars.items():
    inside = masks.get(name)
    if "time" not in values.dims or name in static:
      if "time" in values.dims:
        values = values.isel(time=0, drop=True)
      # One chunk, as written (in memory after the normalization).
      result = _min_max(values, inside).chunk({})
    else:
      chunks = _write_chunks(values.variable, configs)
      x = values.chunk(chunks)
      mean = aligned_climatology[name].chunk(chunks)
      std = aligned_anomaly_std[name].chunk(chunks)
      floor = _anomaly_std_floor(anomaly_std[name], inside)
      # NaN anomaly_std stays NaN (an error inside the mask): xr.where, not a maximum.
      result = (x - mean) / xr.where(std < floor, floor, std)
      tolerance = configs.equal_tolerance * np.finfo(np.float32).eps
      equal = abs(x - mean) <= tolerance * xr.ufuncs.maximum(abs(x), abs(mean))
      result = xr.where(equal, 0.0, result)
      if in_period is not None:
        reduced = [dim for dim in result.dims if dim != "depth"]
        cap = abs(result.where(in_period)).max(reduced)
        result = xr.where(in_period, result, result.clip(-cap, cap))
    if inside is not None:
      result = result.where(inside, 0.0)
    standardized[name] = _check_finite(result.astype(np.float32), name)

  result = xr.Dataset(standardized, attrs=dataset.attrs).drop_encoding()
  result.attrs["history"] = (
    f"{dataset.attrs.get('history', '')}\n"
    "Standardized anomalies: z = (x - climatology) / max(anomaly_std, smallest positive anomaly_std), 0 where x equals "
    f"the climatology within {configs.equal_tolerance:g} float32 epsilons; static variables min-max normalized; 0 "
    "outside the masks of the variables (graphcast download)."
  ).strip()
  return result


def _subchunk_transpose(variable: xr.Variable, configs: DownloadConfig) -> tuple[int, ...] | None:
  """Order of a transpose codec applied before sharding, or None.

  Writers fill shards in lexicographic order of their chunk grid: reversing the chunked dimensions yields the
  colexicographic order. Chunks are single indices along those dimensions, so their contents are unchanged.
  """
  chunked = [axis for axis, dim in enumerate(variable.dims) if dim in _CHUNKED_DIMS]
  if configs.subchunk_order == "lexicographic" or len(chunked) < 2:
    return None
  order = list(range(variable.ndim))
  for axis, transposed in zip(chunked, reversed(chunked), strict=True):
    order[axis] = transposed
  return tuple(order)


def _compressor(blosc: CompressorConfig, zarr_format: int) -> numcodecs.Blosc | BloscCodec:
  if zarr_format == 2:
    shuffle = {
      "noshuffle": numcodecs.Blosc.NOSHUFFLE,
      "shuffle": numcodecs.Blosc.SHUFFLE,
      "bitshuffle": numcodecs.Blosc.BITSHUFFLE,
    }[blosc.shuffle]
    return numcodecs.Blosc(cname=blosc.cname, clevel=blosc.clevel, shuffle=shuffle)
  return BloscCodec(cname=blosc.cname, clevel=blosc.clevel, shuffle=blosc.shuffle)


def get_encoding(dataset: xr.Dataset, configs: DownloadConfig) -> dict[Hashable, dict]:
  """One chunk per timestep and depth level, sharded along time in Zarr v3; static and coordinate arrays whole."""
  compressor = _compressor(configs.compressor, configs.zarr_format)
  encoding = {}
  for name, variable in dataset.variables.items():
    if name in dataset.coords or "time" not in variable.dims:
      encoding[name] = {"chunks": variable.shape, "compressors": compressor}
      continue
    chunks = tuple(1 if dim in _CHUNKED_DIMS else size for dim, size in variable.sizes.items())
    if configs.zarr_format == 2:
      encoding[name] = {"chunks": chunks, "compressors": compressor}
      continue
    # Shards are the Dask chunks (_write_chunks). The sharding codec is explicit to put the transpose codec, if any,
    # before it.
    order = _subchunk_transpose(variable, configs)
    encoding[name] = {
      "chunks": variable.data.chunksize,
      "filters": [] if order is None else [TransposeCodec(order=order)],
      "serializer": ShardingCodec(
        chunk_shape=chunks if order is None else tuple(chunks[axis] for axis in order),
        codecs=[BytesCodec(), compressor],
      ),
      "compressors": None,
    }
  return encoding


def write(
  standardized: xr.Dataset,
  output_path: pathlib.Path,
  configs: DownloadConfig,
  context: tensorstore.Context,
  overwrite: bool = False,
):
  """Write to a local Zarr store: metadata and coordinates with xarray, data variables with TensorStore."""
  # Without computing, to_zarr writes metadata and in-memory (coordinate) arrays only.
  standardized.to_zarr(
    output_path,
    mode="w" if overwrite else "w-",
    compute=False,
    encoding=get_encoding(standardized, configs),
    zarr_format=configs.zarr_format,
    consolidated=True,
  )
  driver = "zarr3" if configs.zarr_format == 3 else "zarr"
  names = list(standardized.data_vars)
  targets = [
    tensorstore.open(
      {
        "driver": driver,
        "kvstore": {"driver": "file", "path": str(output_path / str(name))},
      },
      context=context,
      open=True,
      write=True,
    )
    for name in names
  ]
  dask.array.store(
    [standardized[name].data for name in names],
    [target.result() for target in targets],
    lock=False,
  )


def get_mask(dataset: xr.Dataset, configs: MaskDownloadConfig) -> xr.DataArray:
  """Return the mask variable of the dataset at one depth level, as a boolean array without depth (read)."""
  if configs.name not in dataset:
    raise ValueError(f"Mask {configs.name!r} not in the dataset")
  mask = dataset[configs.name]
  if "time" in mask.dims:
    raise ValueError(f"Mask {configs.name!r} depends on time: expected a static variable")
  if "depth" in mask.dims:
    mask = mask.isel(depth=configs.level, drop=True)
  elif configs.level != 0:
    raise ValueError(f"Mask {configs.name!r} has no depth dimension: level must be 0")
  # The source encoding (e.g. an integer dtype and fill value) does not apply to booleans.
  mask = xarray_tensorstore.read(mask).compute().astype(bool).drop_encoding()
  mask.attrs["history"] = (
    f"{mask.attrs.get('history', '')}\n{configs.name} at depth position {configs.level}, as booleans (graphcast "
    "download-mask)."
  ).strip()
  return mask


def tendency_std(standardized: xr.Dataset, configs: TendencyStdConfig) -> xr.Dataset:
  """Return the standard deviation of the one-step differences of the time-dependent variables (lazily).

  The mean of the differences is taken as zero: it is (x_N - x_1) / (N - 1), a telescopic sum, negligible for long
  series. For `zero_inflated` variables (e.g. sea ice, whose differences are mostly zero) only the non-zero differences
  count. Constant series have 0 (training excludes them from the loss). Squares are accumulated in float64; the result
  is float32, without the time dimension.
  """
  time_dependent = [name for name, values in standardized.data_vars.items() if "time" in values.dims]
  names = time_dependent if configs.variables is None else configs.variables
  unknown = [name for name in [*names, *configs.zero_inflated] if name not in time_dependent]
  if unknown:
    raise ValueError(f"Not time-dependent variables of the dataset: {unknown}")
  result = {}
  for name in names:
    # One depth level per chunk, as in the store: chunks along time only would hold all levels.
    values = standardized[name]
    values = values.chunk({"time": configs.time_chunk, **({"depth": 1} if "depth" in values.dims else {})})
    diffs = values.astype(np.float64).diff("time")
    squares = diffs * diffs
    if name in configs.zero_inflated:
      count = (diffs != 0).sum("time")
      variance = squares.sum("time") / count.where(count > 0, 1)
    else:
      variance = squares.mean("time")
    result[name] = xr.ufuncs.sqrt(variance).astype(np.float32).assign_attrs(standardized[name].attrs)
  dataset = xr.Dataset(result, attrs=standardized.attrs).drop_encoding()
  zero_inflated = [name for name in names if name in configs.zero_inflated]
  dataset.attrs["history"] = (
    f"{standardized.attrs.get('history', '')}\nStandard deviation of the one-step differences, assuming zero mean"
    f"{f', of the non-zero ones for {zero_inflated}' if zero_inflated else ''} (graphcast compute-tendency-std)."
  ).strip()
  dataset.attrs.update(configs.attrs)
  return dataset


_CONFIG_OPTION = click.option(
  "--config",
  "config_path",
  type=click.Path(path_type=pathlib.Path, exists=True, dir_okay=False, resolve_path=True),
  help="TOML settings; defaults are used without it.",
)
_OVERWRITE_OPTION = click.option("--overwrite", is_flag=True, default=False, help="Overwrite OUTPUT if it exists.")
_OUTPUT_ARGUMENT = click.argument(
  "output_path",
  metavar="OUTPUT",
  type=click.Path(path_type=pathlib.Path, file_okay=False, resolve_path=True),
)


def _set_up[Config: (DownloadConfig, MaskDownloadConfig, TendencyStdConfig)](
  schema: type[Config],
  config_path: pathlib.Path | None,
  output_path: pathlib.Path,
  overwrite: bool,
  log_level: str,
) -> Config:
  """Configure logging, read the settings (defaults without a config file) and check the output path."""
  cli_utils.configure_logging(log_level)
  configs = schema() if config_path is None else cli_utils.read_configs(config_path, schema=schema)
  if output_path.exists() and not overwrite:
    raise click.ClickException(f"{output_path} exists: pass --overwrite to replace it")
  logger.info(f"Using settings: {configs!r}")
  return configs


def _input_context(io_threads: int, *paths: str) -> tensorstore.Context:
  """TensorStore context of the inputs, with S3 credentials if any of them is on S3."""
  is_s3 = any(path.startswith("s3://") for path in paths)
  return get_tensorstore_context(io_threads, get_aws_credentials() if is_s3 else None)


@click.command()
@click.argument("dataset_path", metavar="DATASET", type=str)
@_OUTPUT_ARGUMENT
@_CONFIG_OPTION
@_OVERWRITE_OPTION
@cli_utils.LOG_LEVEL_OPTION
def download_mask(
  dataset_path: str,
  output_path: pathlib.Path,
  config_path: pathlib.Path | None = None,
  overwrite: bool = False,
  log_level: str = "info",
):
  """Write a mask of DATASET at one depth level to OUTPUT, a local Zarr store holding one boolean DataArray.

  DATASET is a Zarr store with consolidated metadata, given as a local path or a URL (e.g. s3://bucket/path). The
  mask variable and level come from the settings (graphcast.download_config.MaskDownloadConfig).
  """
  configs = _set_up(MaskDownloadConfig, config_path, output_path, overwrite, log_level)
  dataset = open_input(dataset_path, _input_context(1, dataset_path))
  mask = get_mask(dataset, configs)
  logger.info(f"Writing {configs.name} ({dict(mask.sizes)}, {int(mask.sum())} unmasked points) to {output_path}")
  mask.to_zarr(output_path, mode="w" if overwrite else "w-", zarr_format=configs.zarr_format, consolidated=True)


@click.command()
@click.argument("standardized_path", metavar="STANDARDIZED", type=str)
@_OUTPUT_ARGUMENT
@_CONFIG_OPTION
@_OVERWRITE_OPTION
@cli_utils.LOG_LEVEL_OPTION
def compute_tendency_std(
  standardized_path: str,
  output_path: pathlib.Path,
  config_path: pathlib.Path | None = None,
  overwrite: bool = False,
  log_level: str = "info",
):
  """Write the standard deviation of the one-step differences of STANDARDIZED to OUTPUT, a local Zarr store.

  STANDARDIZED is a store written by graphcast download, given as a local path or a URL (e.g. s3://bucket/path). The
  output (tendency_std of the training configs) normalizes the residuals predicted by the model. Settings:
  graphcast.download_config.TendencyStdConfig.
  """
  configs = _set_up(TendencyStdConfig, config_path, output_path, overwrite, log_level)
  standardized = open_input(standardized_path, _input_context(configs.io_threads, standardized_path))
  result = tendency_std(standardized, configs)
  compressor = _compressor(configs.compressor, configs.zarr_format)
  encoding = {
    name: {
      "chunks": tuple(1 if dim == "depth" else size for dim, size in values.sizes.items()),
      "compressors": compressor,
    }
    for name, values in result.data_vars.items()
  }
  logger.info(f"Reading {humanize.naturalsize(standardized[list(result.data_vars)].nbytes)} of {standardized_path}")
  start_time = time.perf_counter()
  with (
    dask.config.set(scheduler="threads", num_workers=configs.num_workers),
    ProgressBar(dt=5),
  ):
    result.to_zarr(
      output_path,
      mode="w" if overwrite else "w-",
      encoding=encoding,
      zarr_format=configs.zarr_format,
      consolidated=True,
    )
  logger.info(f"Done in {humanize.precisedelta(time.perf_counter() - start_time)}")


@click.command()
@click.argument("dataset_path", metavar="DATASET", type=str)
@click.argument("climatology_path", metavar="CLIMATOLOGY", type=str)
@click.argument("anomaly_std_path", metavar="ANOMALY_STD", type=str)
@_OUTPUT_ARGUMENT
@_CONFIG_OPTION
@_OVERWRITE_OPTION
@cli_utils.LOG_LEVEL_OPTION
def download(
  dataset_path: str,
  climatology_path: str,
  anomaly_std_path: str,
  output_path: pathlib.Path,
  config_path: pathlib.Path | None = None,
  overwrite: bool = False,
  log_level: str = "info",
):
  """Write the standardized anomalies of DATASET to OUTPUT, a local Zarr store with consolidated metadata.

  DATASET, CLIMATOLOGY and ANOMALY_STD are Zarr stores with consolidated metadata, given as local paths or URLs
  (e.g. s3://bucket/path). Settings: graphcast.download_config.DownloadConfig (Zarr v2 without a config file).
  """
  configs = _set_up(DownloadConfig, config_path, output_path, overwrite, log_level)
  context = _input_context(configs.io_threads, dataset_path, climatology_path, anomaly_std_path)
  dataset = open_input(dataset_path, context)
  climatology = open_input(climatology_path, context)
  anomaly_std = open_input(anomaly_std_path, context)

  standardized = standardize(dataset, climatology, anomaly_std, configs)
  logger.info(f"Writing {humanize.naturalsize(standardized.nbytes)} (uncompressed) to {output_path}")
  start_time = time.perf_counter()
  with (
    dask.config.set(scheduler="threads", num_workers=configs.num_workers),
    ProgressBar(dt=5),
  ):
    write(standardized, output_path, configs, context, overwrite=overwrite)
  logger.info(f"Done in {humanize.precisedelta(time.perf_counter() - start_time)}")
