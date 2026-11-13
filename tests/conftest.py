# SPDX-FileCopyrightText: 2026 Stefano Campanella
# SPDX-License-Identifier: Apache-2.0
#
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
"""Shared fixtures: small synthetic inputs of the downloader, shaped like the ocean dataset and its statistics."""

import dataclasses
import os
import pathlib

import numpy as np
import pandas as pd
import pytest
import xarray as xr

# The tests run on CPU: without this, in a shell of the image (e.g. apptainer run) on a node without GPUs, the CUDA
# plugin fails to initialize ("No visible GPU devices") instead of falling back to the CPU.
os.environ.setdefault("JAX_PLATFORMS", "cpu")
# Plots are drawn without a display.
os.environ.setdefault("MPLBACKEND", "Agg")
# Four CPU devices, for the tests of data parallelism (pmap, shard_map). Set before jax is imported (none of the
# imports above does); the image sets XLA_FLAGS itself, so the flag is appended.
if "--xla_force_host_platform_device_count" not in os.environ.get("XLA_FLAGS", ""):
  os.environ["XLA_FLAGS"] = (os.environ.get("XLA_FLAGS", "") + " --xla_force_host_platform_device_count=4").strip()

DEPTH, LAT, LON = 3, 4, 5
# Download settings matching the synthetic inputs (as configs/download/zarr_*.toml do for the real ones).
MASKS = {"glorys_mask": ["thetao", "zos", "deptho"]}
STATIC_VARIABLES = ["lsm"]
DOWNLOAD_SETTINGS = {"masks": MASKS, "static_variables": STATIC_VARIABLES}
# Points of zos with a zero anomaly_std on 2020-03-01 (climatology day 59): equal to the climatology, and not.
ZOS_EQUAL, ZOS_UNEQUAL = (0, 1), (1, 1)
LAND_COLUMN = (3, 4)


def to_toml(settings: dict) -> str:
  """TOML of flat settings: scalars, lists (of scalars or of tables, written inline) and tables of them (enough for
  the download and graph-statistics configs)."""

  def value(item) -> str:
    if isinstance(item, bool):
      return str(item).lower()
    if isinstance(item, str):
      return f'"{item}"'
    if isinstance(item, dict):
      return "{" + ", ".join(f'"{name}" = {value(element)}' for name, element in item.items()) + "}"
    if isinstance(item, (list, tuple)):
      return "[" + ", ".join(value(element) for element in item) + "]"
    return str(item)

  lines = [f"{key} = {value(item)}" for key, item in settings.items() if not isinstance(item, dict)]
  for key, table in settings.items():
    if isinstance(table, dict):
      lines += [f"[{key}]", *(f'"{name}" = {value(item)}' for name, item in table.items())]
  return "\n".join(lines) + "\n"


@pytest.fixture(scope="session")
def inputs() -> tuple[xr.Dataset, xr.Dataset, xr.Dataset]:
  """Dataset, 365-day climatology and anomaly_std, around February 29 of a leap year.

  Missing values are outside the mask only: a land column, and land below the first level at another point. A
  time-dependent field (lsm) is constant, and zos has a zero anomaly_std on one day at two points.
  """
  rng = np.random.default_rng(0)
  time = pd.date_range("2020-02-20", "2020-03-05", freq="D")
  # A regular global grid (latitudes from -90 + 45/2 to 90 - 45/2), as the loss latitude weights expect.
  coords = {
    "depth": np.array([0.5, 10.0, 100.0]),
    "lat": np.linspace(-67.5, 67.5, LAT),
    "lon": np.linspace(0, 288, LON),
  }
  land = np.zeros((DEPTH, LAT, LON), dtype=bool)
  land[1:, 0, 0] = True
  land[:, LAND_COLUMN[0], LAND_COLUMN[1]] = True

  def fields(num_days: int, offset: float, scale: float) -> dict[str, tuple]:
    thetao = offset + scale * rng.random((num_days, DEPTH, LAT, LON))
    thetao[:, land] = np.nan
    zos = offset + scale * rng.random((num_days, LAT, LON))
    zos[:, land[0]] = np.nan
    return {"thetao": (("time", "depth", "lat", "lon"), thetao), "zos": (("time", "lat", "lon"), zos)}

  dataset = xr.Dataset(fields(len(time), 10.0, 5.0), coords={"time": time, **coords})
  dataset["deptho"] = (("lat", "lon"), np.where(land[0], np.nan, rng.uniform(100.0, 5000.0, (LAT, LON))))
  dataset["glorys_mask"] = (("depth", "lat", "lon"), (~land).astype(np.int8))
  dataset["lsm"] = (("time", "lat", "lon"), np.broadcast_to(rng.random((LAT, LON)), (len(time), LAT, LON)).copy())
  climatology = xr.Dataset(fields(365, 10.0, 5.0), coords={"time": np.arange(365), **coords})
  anomaly_std = xr.Dataset(fields(365, 0.5, 1.0), coords={"time": np.arange(365), **coords})
  # 2020-03-01 uses climatology day 59: a zero anomaly_std where the value equals the climatology, and where not.
  day = time.get_loc(pd.Timestamp("2020-03-01"))
  for point in (ZOS_EQUAL, ZOS_UNEQUAL):
    anomaly_std["zos"][(59, *point)] = 0.0
  dataset["zos"][(day, *ZOS_EQUAL)] = climatology["zos"][(59, *ZOS_EQUAL)]
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


@pytest.fixture
def download_settings() -> dict:
  """Download settings matching the synthetic inputs (masks and static variables)."""
  return {"masks": {k: list(v) for k, v in MASKS.items()}, "static_variables": list(STATIC_VARIABLES)}


@pytest.fixture
def special_points() -> dict:
  """Points of the synthetic inputs exercising the rules (zos on 2020-03-01, the land column)."""
  return {"zos_equal": ZOS_EQUAL, "zos_unequal": ZOS_UNEQUAL, "land_column": LAND_COLUMN}


@pytest.fixture
def write_config(tmp_path):
  """Write settings to a TOML file and return its path."""

  def write(settings: dict, name: str = "configs.toml") -> pathlib.Path:
    path = tmp_path / name
    path.write_text(to_toml(settings))
    return path

  return write


def _climatology_day(date: pd.Timestamp) -> int:
  """Zero-based day of a 365-day climatology: February 29 reuses February 28, later leap-year days shift back."""
  if date.is_leap_year and (date.month, date.day) >= (2, 29):
    return date.dayofyear - 1 - 1
  return date.dayofyear - 1


def _inside(dataset: xr.Dataset, name: str) -> xr.DataArray | None:
  for mask_name, names in MASKS.items():
    if name in names:
      mask = dataset[mask_name].astype(bool)
      return mask if "depth" in dataset[name].dims else mask.isel(depth=0, drop=True)
  return None


@pytest.fixture(scope="session")
def reference(inputs) -> xr.Dataset:
  """Standardized anomalies of the inputs, computed directly by the rules (absolute time, no batch dimension)."""
  dataset, climatology, anomaly_std = inputs
  days = [_climatology_day(date) for date in dataset.indexes["time"]]
  result = {}
  for name, values in dataset.data_vars.items():
    inside = _inside(dataset, name)
    if name in STATIC_VARIABLES or "time" not in values.dims:
      values = values.isel(time=0, drop=True) if "time" in values.dims else values
      if np.issubdtype(values.dtype, np.floating):
        valid = values if inside is None else values.where(inside)
        values = (values - valid.min()) / (valid.max() - valid.min())
    else:
      # Floor: smallest positive anomaly_std inside the mask, per level, over the 365 days.
      positive = anomaly_std[name].where(anomaly_std[name] > 0)
      positive = positive if inside is None else positive.where(inside)
      floor = positive.min([dim for dim in positive.dims if dim != "depth"])
      mean = climatology[name].isel(time=days).values
      std = anomaly_std[name].isel(time=days)
      x = values.values
      z = (x - mean) / np.maximum(std, floor).values
      equal = np.abs(x - mean) <= 4 * np.finfo(np.float32).eps * np.maximum(np.abs(x), np.abs(mean))
      values = values.copy(data=np.where(equal, 0.0, z))
    if inside is not None:
      values = values.where(inside, 0.0)
    result[name] = values.astype(np.float32)
  return xr.Dataset(result)


@pytest.fixture(scope="session")
def standardized_path(input_paths, tmp_path_factory) -> str:
  """The inputs standardized by moby-dick download into a Zarr v2 store."""
  from click.testing import CliRunner

  from mobydick.downloader import download

  root = tmp_path_factory.mktemp("standardized")
  config = root / "download.toml"
  config.write_text(to_toml(DOWNLOAD_SETTINGS))
  path = root / "zarr_v2"
  result = CliRunner().invoke(download, [*input_paths, str(path), "--config", str(config), "--log-level", "warning"])
  assert result.exit_code == 0, result.output
  return str(path)


@pytest.fixture(scope="session")
def mask_path(input_paths, tmp_path_factory) -> str:
  """The surface mask of the inputs written by moby-dick download-mask."""
  from click.testing import CliRunner

  from mobydick.downloader import download_mask

  path = tmp_path_factory.mktemp("mask") / "mask"
  result = CliRunner().invoke(download_mask, [input_paths[0], str(path), "--log-level", "warning"])
  assert result.exit_code == 0, result.output
  return str(path)


@dataclasses.dataclass(frozen=True)
class TinyModel:
  """A tiny Leviathan task on a coarse grid and the 42-vertex icosahedral multi-mesh, with a batch of random data."""

  lat: np.ndarray
  lon: np.ndarray
  task: object
  model_config: object
  mesh_data: object
  batch_size: int
  batch: tuple[xr.Dataset, xr.Dataset, xr.Dataset]
  mask: xr.DataArray

  def leviathan(self, scan: bool = False):
    from mobydick.leviathan.model import Leviathan

    return Leviathan(
      _model_config=self.model_config,
      _task_config=self.task,
      _grid_lat=self.lat,
      _grid_lon=self.lon,
      _grid_mask=None,
      _mesh_data=self.mesh_data,
      _scan=scan,
    )


@pytest.fixture(scope="session")
def tiny_model() -> TinyModel:
  """Inputs `zos`, targets `zos`, forcings `2t` over a batch of 8, a loss mask with land at the northernmost row."""
  from mobydick import icosahedral_mesh
  from mobydick.leviathan.model import ModelConfig, TaskConfig

  lat, lon, batch_size = np.arange(-75.0, 90.0, 30.0), np.arange(0.0, 360.0, 45.0), 8
  rng = np.random.default_rng(0)

  def field(day: int) -> xr.DataArray:
    return xr.DataArray(
      rng.normal(size=(batch_size, 1, lat.size, lon.size)).astype(np.float32),
      dims=("batch", "time", "lat", "lon"),
      coords={"time": np.array([day], dtype="timedelta64[D]"), "lat": lat, "lon": lon},
    )

  ocean = np.broadcast_to(lat[:, None] < 70.0, (lat.size, lon.size))
  return TinyModel(
    lat=lat,
    lon=lon,
    task=TaskConfig(
      input_variables=("zos",), target_variables=("zos",), forcing_variables=("2t",), levels=(0,), input_duration="1D"
    ),
    model_config=ModelConfig(latent_size=4, gnn_msg_steps=1, hidden_layers=1, radius_query_fraction_edge_length=0.6),
    mesh_data=icosahedral_mesh.get_multi_mesh_data(1),
    batch_size=batch_size,
    batch=(xr.Dataset({"zos": field(0)}), xr.Dataset({"zos": field(1)}), xr.Dataset({"2t": field(1)})),
    mask=xr.DataArray(ocean, dims=("lat", "lon"), coords={"lat": lat, "lon": lon}),
  )
