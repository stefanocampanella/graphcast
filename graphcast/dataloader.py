import pathlib
from typing import SupportsIndex, Tuple

import grain.python as grain
import xarray as xr

from graphcast.data_utils import TargetLeadTimes, _get_steps_per_window
from graphcast.data_utils import extract_inputs_targets_forcings
from graphcast.model import TaskConfig


class ARCODataSource(grain.RandomAccessDataSource):
  """A data source for analysis-ready cloud-optimized datasets containing time-series."""
  def __init__(self,
               path: pathlib.Path,
               task: TaskConfig,
               target_lead_times: TargetLeadTimes = "1d",
               fill_value: float = 0.0,
               from_date: str | None = None,
               to_date: str | None = None,
               ):
    self._dataset = xr.open_dataset(path, engine='zarr').sel(time=slice(from_date, to_date))
    self._task = task
    self._target_lead_times = target_lead_times
    self._fill_value = fill_value
    self._timesteps = _get_steps_per_window(dataset=self._dataset,
                                            input_duration=self._task.input_duration,
                                            target_lead_times=self._target_lead_times)

  def __len__(self):
    return len(self._dataset['time']) - self._timesteps + 1

  def __getitem__(self, record_key: SupportsIndex) -> Tuple[xr.Dataset, xr.Dataset, xr.Dataset]:
    """A single element drawn from the ARCODataSource is a time-series starting from `record_key` and followed by `_timesteps` timesteps. """
    idx = record_key.__index__()
    if idx < 0 or idx >= len(self):
      raise IndexError(f'Index {idx} is out of bounds.')
    dataset = self._dataset.isel(time=slice(idx, idx + self._timesteps))
    dataset = dataset.expand_dims(dim='batch', axis=0)
    dataset = dataset.assign_coords({'datetime': dataset['time'].expand_dims(dim='batch', axis=0)})
    dataset['time'] = dataset['time'] - dataset['time'][0]
    # It is crucial to slice the dataset before filling missing values, or to fill missing values after
    # extract_inputs_targets_forcings. Otherwise, each reader would need to load the entire dataset into memory (and
    # even a single copy might be too large, e.g., for ARCO-OCEAN).
    dataset = dataset.fillna(value=self._fill_value)
    inputs, targets, forcings = extract_inputs_targets_forcings(dataset=dataset,
                                                                input_variables=self._task.input_variables,
                                                                target_variables=self._task.target_variables,
                                                                forcing_variables=self._task.forcing_variables,
                                                                levels=self._task.levels,
                                                                input_duration=self._task.input_duration,
                                                                target_lead_times=self._target_lead_times,
                                                                to_jax=False)
    return inputs, targets, forcings

  def __repr__(self):
    return (f'{self.__class__.__name__}(dataset={self._dataset}, '
            f'task={self._task}, '
            f'target lead times={self._target_lead_times} '
            f'fill value={self._fill_value})')

  @property
  def xarray_dataset(self):
    return self._dataset

  def get_sample(self, batch_size: int) -> Tuple[xr.Dataset, xr.Dataset, xr.Dataset]:
    assert 0 < batch_size <= len(self)
    samples = [self[n] for n in range(batch_size)]
    inputs = xr.concat([inputs for inputs, _, _ in samples], dim='batch')
    targets = xr.concat([targets for _, targets, _ in samples], dim='batch')
    forcings = xr.concat([forcings for _, _, forcings in samples], dim='batch')
    return inputs, targets, forcings