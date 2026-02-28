import pathlib
from typing import SupportsIndex

import grain.python as grain
import jax
import numpy as np
import xarray as xr

from graphcast import xarray_jax
from graphcast.data_utils import extract_inputs_targets_forcings


class ARCODataSource(grain.RandomAccessDataSource):
  """A data source for analysis-ready cloud-optimized datasets containing time-series."""
  def __init__(self, path: pathlib.Path, timesteps=3, mask_name='glorys_mask'):
    self._dataset = xr.open_dataset(path, engine='zarr')
    self._timesteps = timesteps
    self._mask_name = mask_name

  def __len__(self):
    return len(self._dataset['time']) - self._timesteps + 1

  def __getitem__(self, record_key: SupportsIndex):
    """A single element drawn from the ARCODataSource is a time-series starting from `record_key` and followed by `_timesteps` timesteps. """
    idx = record_key.__index__()
    if idx < 0 or idx >= len(self):
      raise IndexError(f'Index {idx} is out of bounds.')
    dataset = self._dataset.isel(time=slice(idx, idx + self._timesteps))
    dataset = dataset.expand_dims(dim='batch', axis=0)
    dataset = dataset.assign_coords({'datetime': dataset['time'].expand_dims(dim='batch', axis=0)})
    dataset['time'] = dataset['time'] - dataset['time'][0]

    return dataset

  def __repr__(self):
    return f'{self.__class__.__name__}(dataset={self._dataset}, timesteps={self._timesteps}, mask_name={self._mask_name})'

  @property
  def mask(self):
    return self._dataset[self._mask_name].isel(level=0, drop=True)

  @property
  def dataset(self):
    return self._dataset


class AddLogDepthCoordinate(grain.MapTransform):

  def __init__(self, log_depth_name='log-depth', depth_name='depth'):
    self.log_depth_name = log_depth_name
    self.depth_name = depth_name

  def map(self, dataset: xr.Dataset) -> xr.Dataset:
    return dataset.assign_coords({self.log_depth_name: - np.log(dataset[self.depth_name])})


class FillNans(grain.MapTransform):

  def __init__(self, value=0.0):
    self.value = value

  def map(self, dataset: xr.Dataset) -> xr.Dataset:
    # Notice: as a side-effect, boolean variables get casted to float32 (which is useful)
    dataset = dataset.fillna(value=self.value)
    return dataset


class ExtractInputsTargetsForcings(grain.MapTransform):

  def __init__(self, task, target_lead_times="1d", derived_vars_device=None):
    self.task = task
    self.target_lead_times = target_lead_times
    self.derived_vars_device = derived_vars_device

  def map(self, dataset: xr.Dataset) -> tuple[xr.Dataset, xr.Dataset, xr.Dataset]:
    inputs, targets, forcings = extract_inputs_targets_forcings(dataset=dataset,
                                                                **self.task,
                                                                target_lead_times=self.target_lead_times,
                                                                to_jax=False)
    return inputs, targets, forcings


class WrapData(grain.MapTransform):
  """Wraps data in a jax.tree_util compatible structure to allow data movements between processes and work with JAX
  arrays."""

  def map(self, element):

    def _wrap_data(dataset: xr.Dataset) -> xr.Dataset:
      # The main reason for using WrapData is to ensure that the data is contiguous. Otherwise, when using
      # multiprocessing, uncontiguous arrays would not be converted to SharedMemoryArrays and instead would be pickled
      # and transferred to the main process.
      dataset = xarray_jax.Dataset(data_vars={var: (data.dims, np.ascontiguousarray(data.data)) for var, data in dataset.data_vars.items()},
                                   coords=dataset.coords,
                                   jax_coords={},
                                   attrs=dataset.attrs)
      return dataset

    return jax.tree_util.tree_map(_wrap_data, element, is_leaf=lambda x: isinstance(x, xr.Dataset))