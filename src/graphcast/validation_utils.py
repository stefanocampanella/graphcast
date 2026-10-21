# Copyright 2025 Stefano Campanella.
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

import numpy as np
import xarray as xr

from graphcast.losses import normalized_latitude_weights


def get_climatology_ref(
  climatology: xr.Dataset,
  targets_sample: xr.Dataset,
  day_of_year_dim: str = "dayofyear",
  time_dim: str = "time",
) -> xr.Dataset:
  variables = list(targets_sample.keys())
  climatology_chunk = climatology[variables]
  if "level" in targets_sample.dims:
    climatology_chunk = climatology_chunk.sel(level=targets_sample.level)
  time_selection = {day_of_year_dim: targets_sample[time_dim].dt.dayofyear}
  climatology_chunk = climatology_chunk.sel(time_selection)
  climatology_chunk = climatology_chunk.assign_coords({time_dim: targets_sample[time_dim]})
  climatology_chunk = climatology_chunk.drop(day_of_year_dim)
  return climatology_chunk


def spatial_average(
  dataset: xr.Dataset,
  skipna: bool,
  preserved_dims: list[str] | None = None,
  level_dim="level",
) -> xr.Dataset:
  dataset = dataset.mean(level_dim, skipna=skipna)
  preserved_dims = preserved_dims or ["batch", "time"]
  dummy_da = xr.DataArray(coords=dataset.coords)
  weights = normalized_latitude_weights(dummy_da)
  return dataset.weighted(weights).mean(
    [dim for dim in dataset.dims if dim not in preserved_dims], skipna=skipna
  )


def get_acc(
  forecast: xr.Dataset,
  truth: xr.Dataset,
  climatology_ref: xr.Dataset,
  skipna: bool = False,
) -> xr.Dataset:
  forecast_anom = forecast - climatology_ref
  truth_anom = truth - climatology_ref
  norm_sq = spatial_average(forecast_anom**2, skipna=skipna) * spatial_average(
    truth_anom**2, skipna=skipna
  )
  norm = xr.apply_ufunc(np.sqrt, norm_sq)
  return spatial_average(forecast_anom * truth_anom, skipna=skipna) / norm


def get_rmse(
  forecast: xr.Dataset,
  truth: xr.Dataset,
  skipna: bool = False,
) -> xr.Dataset:
  mse = spatial_average((forecast - truth) ** 2, skipna=skipna)
  return xr.apply_ufunc(np.sqrt, mse)


def get_nrmse(
  forecast: xr.Dataset,
  truth: xr.Dataset,
  climatology_ref: xr.Dataset,
  skipna: bool = False,
) -> xr.Dataset:
  norm = get_rmse(climatology_ref, truth, skipna=skipna)
  return get_rmse(forecast, truth, skipna=skipna) / norm


def get_bias(
  forecast: xr.Dataset,
  truth: xr.Dataset,
  skipna: bool = False,
) -> xr.Dataset:
  return spatial_average(forecast - truth, skipna=skipna)
