import pathlib
from typing import SupportsIndex

import grain.python as grain
import xarray as xr
import numpy as np

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


class AddLogDepthCoordinate(grain.MapTransform):

  def map(self, dataset: xr.Dataset) -> xr.Dataset:
    return dataset.assign_coords({'log-depth': - np.log(dataset['depth'])})


# TODO: experiments in the past used the following torch dataloader, check that all features have been implemented.
# from graphcast import solar_radiation
# from graphcast import model
# from graphcast import xarray_jax
# from torch.utils.data import Dataset, DataLoader as TorchDataLoader, RandomSampler, BatchSampler
#
# class FakeGraphcastDemoDataset(Dataset):
#
#   def __init__(self, dataset_path: pathlib.Path, task_config: model.TaskConfig, fake_len: int = 16, steps: int = 1):
#     self.dataset_path = dataset_path
#     self.task_config = task_config
#     self.fake_len = fake_len
#     self.steps = steps
#
#     with self.dataset_path.open("rb") as dataset_file:
#       example_batch = xarray.load_dataset(dataset_file).compute()
#
#     assert example_batch.sizes["time"] >= 3  # 2 for input, >=1 for targets
#
#     self.example_batch = example_batch
#
#   def __len__(self):
#     return self.fake_len
#
#   def __getitem__(self, idx):
#     inputs, targets, forcings = extract_inputs_targets_forcings(
#       self.example_batch, target_lead_times=slice("6h", f"{self.steps * 6}h"),
#       **dataclasses.asdict(self.task_config))
#     return inputs, targets, forcings
#
#
# class ERA5Dataset(Dataset):
#
#   def __init__(self, dataset_path: Union[pathlib.Path, str], task_config: model.TaskConfig, steps: int = 1):
#     self.dataset_path = dataset_path
#     self.task_config = task_config
#     self.steps = steps
#
#     ds = xarray.open_zarr(dataset_path)
#     assert ds.sizes["time"] >= 3  # at least 2 for input, >=1 for targets
#
#     ds = ds.drop_vars(var for var in ds.data_vars.keys() if var not in task_config.input_variables)
#     ds = ds.expand_dims(dim='batch', axis=0)
#     ds = ds.assign_coords({'datetime': ds['time'].expand_dims(dim='batch', axis=0)})
#     ds['time'] = ds['time'] - ds['time'][0]
#
#     ds = ds.swap_dims(latitude='lat', longitude='lon')
#     ds = ds.rename_vars(latitude='lat', longitude='lon')
#     ds = ds.set_index(lat='lat', lon='lon', level='level', time='time')
#     #TODO: The code should comply with the usual convention for longitudes.
#     # However, this would require retraining of original GraphCast weights.
#     #ds['lon'] = np.where(ds.lon <= 180, ds.lon, ds.lon - 360)
#
#     #TODO: Find out why transpose in accord to demo data breaks experiments.
#     # ds = ds.transpose("batch", "time", "level", "lat", "lon")
#     self.dataset = ds
#
#
#   def __len__(self):
#     return self.dataset.sizes["time"] - 2
#
#
#   def __getitem__(self, idx):
#     ds = self.dataset.sel(time=self.dataset.time[idx:])
#     inputs, targets, forcings = extract_inputs_targets_forcings(ds,
#                                            target_lead_times=slice("6h", f"{self.steps * 6}h"),
#                                            **dataclasses.asdict(self.task_config))
#     return inputs, targets, forcings
#
#
# def device_put(ds, *args, **kwargs):
#
#   def _move_data_array(var, name=None, jax_coords=None):
#     return xarray_jax.DataArray(jax.device_put(var.data, *args, **kwargs),
#                                 coords=var.coords,
#                                 dims=var.dims,
#                                 name=name,
#                                 attrs=var.attrs,
#                                 jax_coords=jax_coords)
#
#   data_variables_names = set(ds.variables.keys()) - set(ds.coords.keys())
#   variables = {name: _move_data_array(ds[name], name=name) for name in data_variables_names}
#   return xarray_jax.Dataset(variables, coords=ds.coords, attrs=ds.attrs)
#
#
# def default_collate_fn(batch):
#   if len(batch) > 1:
#     data = map(lambda datasets: xarray.concat(datasets, dim='batch'), zip(*batch))
#   else:
#     data = batch[0]
#   inputs, targets, forcings = map(lambda ds: ds.compute(), data)
#   return inputs, targets, forcings
#
#
# class DataLoader(TorchDataLoader):
#
#   def __init__(self, dataset: Dataset, batch_size=None, num_samples=None, sharding=None, collate_fn=default_collate_fn, **kwargs):
#     sampler = RandomSampler(dataset, replacement=True, num_samples=num_samples * batch_size, generator=kwargs.get('generator'))
#     batch_sampler = BatchSampler(sampler=sampler, batch_size=batch_size, drop_last=True)
#     kwargs.update({'shuffle': None, 'drop_last': None, 'sampler': None, 'batch_sampler': batch_sampler})
#     super().__init__(dataset, collate_fn=collate_fn, **kwargs)
#     self.sharding = sharding
#
#   def __next__(self):
#     next_elem = super().__next__()
#     if self.sharding is None:
#       inputs, targets, forcings = next_elem
#     else:
#       inputs, targets, forcings = map(lambda x: device_put(x, self.sharding), next_elem)
#     return inputs, targets, forcings
#
#   @property
#   def random_item(self):
#     return next(iter(self))
