import pathlib
from collections import OrderedDict
from typing import SupportsIndex

import grain.python as grain
import jax
import numpy as np
import xarray as xr
from grain.sharding import ShardOptions
from jax.experimental import multihost_utils

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

  @property
  def mask(self):
    return self._dataset[self._mask_name].isel(level=0, drop=True)

  @property
  def dataset(self):
    return self._dataset


class ToXarrayJax(grain.MapTransform):

  def __init__(self, datetime_coord_name='datetime'):
    self.datetime_coord_name = datetime_coord_name

  def map(self, dataset: xr.Dataset) -> xr.Dataset:
    datetime_coord = dataset[self.datetime_coord_name]
    datetime_coord.data = datetime_coord.data.astype("datetime64[s]").astype(np.int64)
    dataset = dataset.drop(self.datetime_coord_name)
    dataset = xarray_jax.Dataset(data_vars={var: (data.dims, data.data) for var, data in dataset.data_vars.items()},
                                 coords=dataset.coords,
                                 jax_coords={self.datetime_coord_name: datetime_coord},
                                 attrs=dataset.attrs)
    return dataset


class DevicePut(grain.MapTransform):
  def __init__(self, sharding, multi_host=False):
    self.sharding = sharding
    self.multi_host = multi_host

  def map(self, dataset: xr.Dataset) -> xr.Dataset:
    if self.multi_host:
      dataset = dataset.map(
        lambda da: jax.tree_util.tree_map(
          lambda xs: jax.make_array_from_process_local_data(sharding=self.sharding, local_data=xs),
        da))
    else:
      dataset = xarray_jax.tree_map_with_dims(lambda xs, _: jax.device_put(xs, self.sharding), dataset)
    return dataset


class AddLogDepthCoordinate(grain.MapTransform):

  def __init__(self, log_depth_name='log-depth', depth_name='depth'):
    self.log_depth_name = log_depth_name
    self.depth_name = depth_name

  def map(self, dataset: xr.Dataset) -> xr.Dataset:
    return dataset.assign_coords({self.log_depth_name: - np.log(dataset[self.depth_name])})


class RestoreDatetimeCoordinate(grain.MapTransform):

  def __init__(self, datetime_coord_name='datetime', datetime_dims_name=('batch', 'time'), multi_host=False):
    self.gather_from_all_processes = multi_host
    self.datetime_coord_name = datetime_coord_name
    self.datetime_dims_name = datetime_dims_name

  def map(self, dataset: xr.Dataset) -> xr.Dataset:
    datetime_coordinate = dataset[self.datetime_coord_name]
    # The datetime coordinate should be reverted to a simple coordinate (and not jax_coordinate with an underlying jax
    # array spanning multiple processes/devices): this requires calling
    # `jax.experimental.multihost_utils.process_allgather`, then unwrapping/converting to numpy.ndarray and calling
    # `xarray_jax.Dataset`, and finally recasting to datetime64[ns] (the easy part).
    if self.gather_from_all_processes:
      datetime_coordinate = multihost_utils.process_allgather(datetime_coordinate)
    # In the case of single-host setups it might happen that datetime is still a numpy array
    datetime_coordinate = xarray_jax.unwrap_data(datetime_coordinate, require_jax=False)
    datetime_coordinate = np.asarray(datetime_coordinate).astype('datetime64[s]')
    dataset = dataset.drop(self.datetime_coord_name)
    dataset = dataset.assign_coords({self.datetime_coord_name: (self.datetime_dims_name, datetime_coordinate)})
    return dataset


class FillNans(grain.MapTransform):

  def __init__(self, value=0.0):
    self.value = value

  def map(self, dataset: xr.Dataset) -> xr.Dataset:
    # Notice: as a side-effect, boolean variables get casted to float32 (which is useful)
    dataset = dataset.fillna(value=jax.numpy.float32(self.value))
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
                                                                to_jax=True,
                                                                derived_vars_device=self.derived_vars_device)
    return inputs, targets, forcings


# class DevicePut(grain.MapTransform):
#
#   def __init__(self, mesh, replicate_along_batch=False, batch_dim_name='batch'):
#
#     self.mesh = mesh
#     self.replicate_along_batch = replicate_along_batch
#     self.batch_dim_name = batch_dim_name
#
#   def map(self, dataset: xr.Dataset) -> xr.Dataset:
#     if self.replicate_along_batch:
#       sharding = NamedSharding(self.mesh, PartitionSpec())
#     else:
#       sharding = NamedSharding(self.mesh, PartitionSpec(self.batch_dim_name))
#
#     def _put_dataarray(data_array):
#       return jax.tree_util.tree_map(lambda xs: jax.device_put(xs, sharding), data_array)
#
#     return dataset.map(lambda da: _put_dataarray(da))

# FIXME: the following seems to be broken for a multi-host-each-with-multiple-devices setup.
class BatchParallelShardOptions(ShardOptions):

  def __init__(self, sharding, batch_dim_name='batch', drop_remainder=False):

    def addressable_device_mesh_indices_map(device):
      global_shape = tuple(size for (_, size) in sharding.mesh.shape_tuple)
      slices = sharding.addressable_devices_indices_map(global_shape)[device]
      indices = OrderedDict((name, s.start) for (name, s) in zip(sharding.mesh.axis_names, slices))
      return indices

    local_device_batch_indices = [addressable_device_mesh_indices_map(device)[batch_dim_name]
                                  for device in sharding.addressable_devices]
    assert all(n == local_device_batch_indices[0] for n in local_device_batch_indices)
    shard_index = local_device_batch_indices[0]
    shard_count = sharding.mesh.shape[batch_dim_name]
    super().__init__(shard_count=shard_count, shard_index=shard_index, drop_remainder=drop_remainder)
