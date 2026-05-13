from collections.abc import Sequence
from typing import SupportsIndex

import grain.python as grain
import numpy as np
import xarray as xr

from graphcast.data_utils import (
  TargetLeadTimes,
  _get_steps_per_window,
  extract_inputs_targets_forcings,
)
from graphcast.model import TaskConfig

InputsTargetsForcings = tuple[xr.Dataset, xr.Dataset, xr.Dataset]


class ARCODataSource(grain.RandomAccessDataSource):
  """A data source for analysis-ready cloud-optimized datasets containing time-series."""

  def __init__(
    self,
    dataset: xr.Dataset,
    task: TaskConfig,
    target_lead_times: TargetLeadTimes = "1d",
    fill_value: float = 0.0,
    valid_dates: Sequence[np.datetime64] | None = None,
  ):
    self._dataset = dataset
    self._task = task
    self._target_lead_times = target_lead_times
    self._fill_value = fill_value
    self._timesteps = _get_steps_per_window(
      dataset=self._dataset,
      input_duration=self._task.input_duration,
      target_lead_times=self._target_lead_times,
    )
    # TODO: check that rollouts starting from valid dates are within dataset
    self._valid_dates = valid_dates

  def __len__(self):
    if self._valid_dates is None:
      num_samples = len(self._dataset["time"]) - self._timesteps + 1
    else:
      num_samples = len(self._valid_dates)
    return num_samples

  def __getitem__(self, record_key: SupportsIndex) -> InputsTargetsForcings:
    """A single element drawn from the ARCODataSource is a time-series starting from `record_key` and followed by `_timesteps` timesteps."""
    idx = record_key.__index__()
    if self._valid_dates is not None:
      date = self._valid_dates[idx]
      idx = self._dataset.get_index("time").get_loc(date)
    # FIXME: If target_lead_times is far away in the future, the datasource will load the full slice increasing memory
    #  usage. Change the implementation to load in memory (and compute derived vars) only for the needed times.
    dataset = self._dataset.isel(time=slice(idx, idx + self._timesteps))
    dataset = dataset.expand_dims(dim="batch", axis=0)
    dataset = dataset.assign_coords({"datetime": dataset["time"].expand_dims(dim="batch", axis=0)})
    dataset["time"] = dataset["time"] - dataset["time"][0]
    # It is crucial to slice the dataset before filling missing values, or to fill missing values after
    # extract_inputs_targets_forcings. Otherwise, each reader would need to load the entire dataset into memory (and
    # even a single copy might be too large, e.g., for ARCO-OCEAN).
    dataset = dataset.fillna(value=self._fill_value)
    dataset = dataset.astype(np.float32)
    inputs, targets, forcings = extract_inputs_targets_forcings(
      dataset=dataset,
      input_variables=self._task.input_variables,
      target_variables=self._task.target_variables,
      forcing_variables=self._task.forcing_variables,
      levels=self._task.levels,
      input_duration=self._task.input_duration,
      target_lead_times=self._target_lead_times,
      to_jax=False,
    )
    return inputs, targets, forcings

  def __repr__(self):
    return (
      f"{self.__class__.__name__}(dataset={self._dataset}, "
      f"task={self._task}, "
      f"target lead times={self._target_lead_times} "
      f"fill value={self._fill_value})"
    )

  @property
  def xarray_dataset(self):
    return self._dataset
