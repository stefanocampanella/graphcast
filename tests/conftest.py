"""Shared fixtures: small synthetic inputs of the downloader, shaped like the ocean dataset and its statistics."""

import pathlib

import numpy as np
import pandas as pd
import pytest
import xarray as xr

DEPTH, LAT, LON = 3, 4, 5


@pytest.fixture(scope="session")
def inputs() -> tuple[xr.Dataset, xr.Dataset, xr.Dataset]:
  """Dataset, 365-day climatology and anomaly_std with missing values, around February 29 of a leap year."""
  rng = np.random.default_rng(0)
  time = pd.date_range("2020-02-20", "2020-03-05", freq="D")
  coords = {"depth": np.array([0.5, 10.0, 100.0]), "lat": np.linspace(-60, 60, LAT), "lon": np.linspace(0, 288, LON)}
  # Land below the first level at one point; it is missing in all inputs.
  land = np.zeros((DEPTH, LAT, LON), dtype=bool)
  land[1:, 0, 0] = True

  def fields(num_days: int, offset: float, scale: float) -> dict[str, tuple]:
    thetao = offset + scale * rng.random((num_days, DEPTH, LAT, LON))
    thetao[:, land] = np.nan
    zos = offset + scale * rng.random((num_days, LAT, LON))
    return {"thetao": (("time", "depth", "lat", "lon"), thetao), "zos": (("time", "lat", "lon"), zos)}

  dataset = xr.Dataset(fields(len(time), 10.0, 5.0), coords={"time": time, **coords})
  dataset["deptho"] = (("lat", "lon"), rng.uniform(100.0, 5000.0, (LAT, LON)))
  dataset["deptho"][0, 1] = np.nan
  dataset["glorys_mask"] = (("depth", "lat", "lon"), (~land).astype(np.int8))
  climatology = xr.Dataset(fields(365, 10.0, 5.0), coords={"time": np.arange(365), **coords})
  anomaly_std = xr.Dataset(fields(365, 0.5, 1.0), coords={"time": np.arange(365), **coords})
  # Zero deviations are clipped, missing ones filled.
  anomaly_std["zos"][59, 0, 1] = 0.0
  anomaly_std["zos"][58, 1, 1] = np.nan
  return dataset, climatology, anomaly_std


@pytest.fixture(scope="session")
def input_paths(inputs, tmp_path_factory) -> tuple[str, ...]:
  """The inputs as local Zarr v2 stores with consolidated metadata."""
  root = tmp_path_factory.mktemp("inputs")
  paths = []
  for name, data in zip(("dataset", "climatology", "anomaly_std"), inputs, strict=True):
    path = root / name
    data.to_zarr(path, zarr_format=2, consolidated=True)
    paths.append(str(path))
  return tuple(paths)


@pytest.fixture
def output_path(tmp_path) -> pathlib.Path:
  return tmp_path / "standardized.zarr"


def _climatology_day(date: pd.Timestamp) -> int:
  """Zero-based day of a 365-day climatology: February 29 reuses February 28, later leap-year days shift back."""
  if date.is_leap_year and (date.month, date.day) >= (2, 29):
    return date.dayofyear - 1 - 1
  return date.dayofyear - 1


@pytest.fixture(scope="session")
def reference(inputs) -> xr.Dataset:
  """Standardized anomalies of the inputs, computed directly (absolute time, no batch dimension)."""
  dataset, climatology, anomaly_std = inputs
  days = [_climatology_day(date) for date in dataset.indexes["time"]]
  result = {}
  for name, values in dataset.data_vars.items():
    if "time" in values.dims:
      mean = climatology[name].isel(time=days).fillna(0.0).values
      std = anomaly_std[name].isel(time=days).fillna(1.0).clip(min=1e-16).values
      values = (values - mean) / std
    elif np.issubdtype(values.dtype, np.floating):
      values = (values - values.min()) / (values.max() - values.min())
    result[name] = values.fillna(0.0).astype(np.float32)
  return xr.Dataset(result)


@pytest.fixture(scope="session")
def standardized_path(input_paths, tmp_path_factory) -> str:
  """The inputs standardized by graphcast download into a Zarr v2 store."""
  from click.testing import CliRunner

  from graphcast.downloader import download

  path = tmp_path_factory.mktemp("standardized") / "zarr_v2"
  result = CliRunner().invoke(download, [*input_paths, str(path), "--log-level", "warning"])
  assert result.exit_code == 0, result.output
  return str(path)


@pytest.fixture(scope="session")
def mask_path(input_paths, tmp_path_factory) -> str:
  """The surface mask of the inputs written by graphcast download-mask."""
  from click.testing import CliRunner

  from graphcast.downloader import download_mask

  path = tmp_path_factory.mktemp("mask") / "mask"
  result = CliRunner().invoke(download_mask, [input_paths[0], str(path), "--log-level", "warning"])
  assert result.exit_code == 0, result.output
  return str(path)
