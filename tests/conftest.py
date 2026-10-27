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
  dataset["mask"] = (("depth", "lat", "lon"), (~land).astype(np.int8))
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
