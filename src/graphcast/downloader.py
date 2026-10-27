# SPDX-FileCopyrightText: 2026 Stefano Campanella
# SPDX-License-Identifier: MIT
"""Write the standardized anomalies of a dataset, read from local storage or object storage, to a local Zarr store.

The preprocessing is that of `dataloader.ARCOTimeseriesSource`, except adding the batch dimension and making time
relative to the first timestep, which stay in the dataloader.
"""

import logging
import os
import pathlib
import time

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
from graphcast.dataloader import climatology_index
from graphcast.download_config import DownloadConfig

logger = logging.getLogger(__name__)

# Dimensions with one chunk per index; arrays are whole along the others.
_CHUNKED_DIMS = ("time", "depth")


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


def get_tensorstore_context(configs: DownloadConfig, aws_credentials: dict | None = None) -> tensorstore.Context:
  resources = {
    "data_copy_concurrency": {"limit": configs.io_threads},
    "file_io_concurrency": {"limit": configs.io_threads},
    "s3_request_concurrency": {"limit": configs.io_threads},
  }
  if aws_credentials is not None:
    resources["aws_credentials"] = aws_credentials
  return tensorstore.Context(resources)


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


def standardize(
    dataset: xr.Dataset,
    climatology: xr.Dataset,
    anomaly_std: xr.Dataset,
    configs: DownloadConfig,
) -> xr.Dataset:
  """Return the anomaly z-scores of the time-dependent variables and the normalized static variables (lazily)."""
  missing = [
    name for name, values in dataset.data_vars.items()
    if "time" in values.dims and (name not in climatology or name not in anomaly_std)
  ]
  if missing:
    raise ValueError(f"Time-dependent variables without climatology or anomaly_std: {missing}")

  # Align climatology and anomaly_std with the timeseries; the lazy indexing reads only the selected days.
  index = climatology_index(dataset["time"])
  climatology = climatology.isel(time=index).assign_coords(time=dataset["time"])
  anomaly_std = anomaly_std.isel(time=index).assign_coords(time=dataset["time"])

  standardized = {}
  for name, values in dataset.data_vars.items():
    chunks = _write_chunks(values.variable, configs)
    values = values.chunk(chunks)
    if "time" in values.dims:
      mean = climatology[name].chunk(chunks).fillna(value=0.0)
      std = anomaly_std[name].chunk(chunks).fillna(value=1.0).clip(min=1e-16)
      values = (values - mean) / std
    elif np.issubdtype(values.dtype, np.floating):
      # Min-max normalization of static fields; masks (integers) are only cast to float32.
      minimum = values.min()
      span = values.max() - minimum
      values = (values - minimum) / span.where(span != 0, 1)
    standardized[name] = values.fillna(value=0.0).astype(np.float32)

  result = xr.Dataset(standardized, attrs=dataset.attrs).drop_encoding()
  result.attrs["history"] = (
    f"{dataset.attrs.get('history', '')}\n"
    "Standardized anomalies (climatology and anomaly_std: missing values filled with 0 and 1, std clipped below at "
    "1e-16), static variables min-max normalized, missing values filled with 0 (graphcast download)."
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


def get_encoding(dataset: xr.Dataset, configs: DownloadConfig) -> dict[str, dict]:
  """One chunk per timestep and depth level, sharded along time in Zarr v3; static and coordinate arrays whole."""
  blosc = configs.compressor
  if configs.zarr_format == 2:
    shuffle = {"noshuffle": numcodecs.Blosc.NOSHUFFLE, "shuffle": numcodecs.Blosc.SHUFFLE,
               "bitshuffle": numcodecs.Blosc.BITSHUFFLE}[blosc.shuffle]
    compressor = numcodecs.Blosc(cname=blosc.cname, clevel=blosc.clevel, shuffle=shuffle)
  else:
    compressor = BloscCodec(cname=blosc.cname, clevel=blosc.clevel, shuffle=blosc.shuffle)
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
      {"driver": driver, "kvstore": {"driver": "file", "path": str(output_path / name)}},
      context=context,
      open=True,
      write=True,
    )
    for name in names
  ]
  dask.array.store(
    [standardized[name].data for name in names], [target.result() for target in targets], lock=False
  )


@click.command()
@click.argument("dataset_path", metavar="DATASET", type=str)
@click.argument("climatology_path", metavar="CLIMATOLOGY", type=str)
@click.argument("anomaly_std_path", metavar="ANOMALY_STD", type=str)
@click.argument(
  "output_path",
  metavar="OUTPUT",
  type=click.Path(path_type=pathlib.Path, file_okay=False, resolve_path=True),
)
@click.option(
  "--config",
  "config_path",
  type=click.Path(path_type=pathlib.Path, exists=True, dir_okay=False, resolve_path=True),
  help="TOML settings (graphcast.download_config.DownloadConfig); defaults to Zarr v2.",
)
@click.option("--overwrite", is_flag=True, default=False, help="Overwrite OUTPUT if it exists.")
@click.option(
  "--log-level",
  default="info",
  type=click.Choice(["debug", "info", "warning", "error", "critical"], case_sensitive=False),
)
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
  (e.g. s3://bucket/path).
  """
  logging.basicConfig(
    format="%(levelname)s - %(asctime)s: %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
    level=log_level.upper(),
    force=True,
  )
  configs = DownloadConfig() if config_path is None else cli_utils.read_configs(config_path, schema=DownloadConfig)
  if output_path.exists() and not overwrite:
    raise click.ClickException(f"{output_path} exists: pass --overwrite to replace it")
  logger.info(f"Using settings: {configs!r}")

  is_s3 = any(path.startswith("s3://") for path in (dataset_path, climatology_path, anomaly_std_path))
  context = get_tensorstore_context(configs, get_aws_credentials() if is_s3 else None)
  dataset = open_input(dataset_path, context)
  climatology = open_input(climatology_path, context)
  anomaly_std = open_input(anomaly_std_path, context)

  standardized = standardize(dataset, climatology, anomaly_std, configs)
  logger.info(f"Writing {humanize.naturalsize(standardized.nbytes)} (uncompressed) to {output_path}")
  start_time = time.perf_counter()
  with dask.config.set(scheduler="threads", num_workers=configs.num_workers), ProgressBar(dt=5):
    write(standardized, output_path, configs, context, overwrite=overwrite)
  logger.info(f"Done in {humanize.precisedelta(time.perf_counter() - start_time)}")
