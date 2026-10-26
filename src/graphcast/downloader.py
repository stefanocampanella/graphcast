# SPDX-FileCopyrightText: 2026 Stefano Campanella
# SPDX-License-Identifier: MIT
"""Write the standardized anomalies of a dataset, read from local storage or object storage, to a local Zarr store.

The preprocessing is that of `dataloader.ARCOTimeseriesSource`, except adding the batch dimension and making time
relative to the first timestep, which stay in the dataloader.
"""

import logging
import pathlib
import time

import click
import dask
import fsspec
import humanize
import numcodecs
import numpy as np
import xarray as xr
from dask.diagnostics import ProgressBar
from zarr.codecs import BloscCodec, BytesCodec, ShardingCodec

from graphcast import cli_utils
from graphcast.dataloader import climatology_index
from graphcast.download_config import DownloadConfig

logger = logging.getLogger(__name__)

# Coordinates are small: compress them more, as in the original datasets.
_COORDS_CLEVEL = 5
_TIME_COORD_CHUNK = 128


def open_input(path: str, storage_options: dict) -> xr.Dataset:
  """Open a Zarr store with consolidated metadata, given as a local path or an fsspec URL (e.g. s3://...)."""
  logger.info(f"Opening {path}")
  is_local = fsspec.utils.get_protocol(path) in ("file", "local")
  # chunks=None keeps lazily indexed arrays: Dask chunks are chosen later, one task per output chunk.
  return xr.open_zarr(
    path, consolidated=True, chunks=None, storage_options=None if is_local else storage_options
  )


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
    elif np.issubdtype(values.dtype, np.bool_):
      values = values.astype(np.float32)
    else:
      # Min-max normalization of static numeric variables
      minimum = values.min()
      span = values.max() - minimum
      values = (values - minimum) / span.where(span != 0, 1)
    standardized[name] = values.fillna(value=0.0).astype(np.float32)

  # Encodings come from get_encoding, except time's (units and calendar), which the source chunks would contradict.
  result = xr.Dataset(standardized, attrs=dataset.attrs).drop_encoding()
  result["time"].encoding = dataset["time"].encoding
  result.attrs["history"] = (
    f"{dataset.attrs.get('history', '')}\n"
    "Standardized anomalies (climatology and anomaly_std: missing values filled with 0 and 1, std clipped below at "
    "1e-16), static variables min-max normalized, missing values filled with 0 (graphcast download)."
  ).strip()
  return result


def get_encoding(dataset: xr.Dataset, configs: DownloadConfig) -> dict[str, dict]:
  """Chunks and compressors of the datasets used in experiments: one chunk per timestep and depth level."""
  blosc = configs.compressor
  encoding = {}
  for name, variable in dataset.variables.items():
    inner_chunks = tuple(1 if dim in ("time", "depth") else size for dim, size in variable.sizes.items())
    clevel = blosc.clevel if name in dataset.data_vars else _COORDS_CLEVEL
    if name == "time" and configs.zarr_format == 2:
      inner_chunks = (min(_TIME_COORD_CHUNK, variable.size),)
    elif name in dataset.coords:
      inner_chunks = variable.shape
    if configs.zarr_format == 2:
      shuffle = {"noshuffle": numcodecs.Blosc.NOSHUFFLE, "shuffle": numcodecs.Blosc.SHUFFLE,
                 "bitshuffle": numcodecs.Blosc.BITSHUFFLE}[blosc.shuffle]
      compressor = numcodecs.Blosc(cname=blosc.cname, clevel=clevel, shuffle=shuffle)
    else:
      compressor = BloscCodec(cname=blosc.cname, clevel=clevel, shuffle=blosc.shuffle)
    if configs.zarr_format == 3 and name in dataset.data_vars and {"time", "depth"} & set(variable.dims):
      # Shards are whole along all dimensions but time. The sharding codec is passed explicitly since the order of
      # chunks within shards is a codec option (not stored in metadata, used for writing only).
      shard_shape = tuple(_write_chunks(variable, configs).get(dim, size) for dim, size in variable.sizes.items())
      encoding[name] = {
        "chunks": tuple(min(shard, size) for shard, size in zip(shard_shape, variable.shape, strict=True)),
        "serializer": ShardingCodec(
          chunk_shape=inner_chunks,
          codecs=[BytesCodec(), compressor],
          subchunk_write_order=configs.subchunk_write_order,
        ),
        "compressors": None,
      }
    else:
      encoding[name] = {"chunks": inner_chunks, "compressors": compressor}
  # Keep the time units and calendar of the source dataset.
  time_encoding = dataset["time"].encoding
  encoding["time"].update({key: time_encoding[key] for key in ("units", "calendar", "dtype") if key in time_encoding})
  return encoding


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

  DATASET, CLIMATOLOGY and ANOMALY_STD are Zarr stores with consolidated metadata, given as local paths or fsspec
  URLs (e.g. s3://bucket/datasets/res=0p25).
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

  dataset = open_input(dataset_path, configs.storage_options)
  climatology = open_input(climatology_path, configs.storage_options)
  anomaly_std = open_input(anomaly_std_path, configs.storage_options)

  standardized = standardize(dataset, climatology, anomaly_std, configs)
  encoding = get_encoding(standardized, configs)
  logger.info(f"Writing {humanize.naturalsize(standardized.nbytes)} (uncompressed) to {output_path}")
  start_time = time.perf_counter()
  with dask.config.set(scheduler="threads", num_workers=configs.num_workers), ProgressBar(dt=5):
    standardized.to_zarr(
      output_path,
      mode="w" if overwrite else "w-",
      encoding=encoding,
      zarr_format=configs.zarr_format,
      consolidated=True,
    )
  logger.info(f"Done in {humanize.precisedelta(time.perf_counter() - start_time)}")
