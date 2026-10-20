import calendar
import operator
from collections.abc import Sequence
from typing import SupportsIndex

import grain.python as grain
import jax
import numpy as np
import xarray as xr

InputsTargetsForcings = tuple[xr.Dataset, xr.Dataset, xr.Dataset]


class ARCODataSource(grain.RandomAccessDataSource):
  """A data source returning normalized time-series windows as single datasets."""

  def __init__(
    self,
    dataset: xr.Dataset,
      climatology: xr.Dataset,
      anomaly_std: xr.Dataset,
      *,
      window_size: int,
    fill_value: float = 0.0,
    valid_dates: Sequence[np.datetime64] | None = None,
  ):
    window_size = operator.index(window_size)
    if not 0 < window_size <= dataset.sizes["time"]:
      raise ValueError("window_size must be positive and no larger than the dataset time dimension.")
    if valid_dates is not None:
      indices = dataset.get_index("time").get_indexer(valid_dates)
      if np.any(indices < 0) or np.any(indices + window_size > dataset.sizes["time"]):
        raise ValueError("valid_dates must occur in the dataset and leave enough timesteps for a full window.")
    self._dataset = dataset
    self._climatology = climatology
    self._anomaly_std = anomaly_std
    self._fill_value = fill_value
    self._window_size = window_size
    self._valid_dates = valid_dates

  def __len__(self):
    if self._valid_dates is None:
      num_samples = len(self._dataset["time"]) - self._window_size + 1
    else:
      num_samples = len(self._valid_dates)
    return num_samples

  def __getitem__(self, record_key: SupportsIndex) -> xr.Dataset:
    """Return a normalized window of `_window_size` timesteps starting at `record_key`."""
    idx = record_key.__index__()
    if self._valid_dates is not None:
      date = self._valid_dates[idx]
      idx = self._dataset.get_index("time").get_loc(date)
    dataset = self._dataset.isel(time=slice(idx, idx + self._window_size))

    def get_climatology_index(dt) -> list[int]:

      def _handle_leap_years(dayofyear: int, year: int):
        if calendar.isleap(year):
          if dayofyear < 58:
            return dayofyear - 1
          else:
            return dayofyear - 2
        else:
          return dayofyear - 1

      return list(map(_handle_leap_years, dt.dayofyear, dt.year))

    # It is crucial to slice datasets before filling missing values, or to fill missing values after
    # extract_inputs_targets_forcings. Otherwise, each reader would need to load the entire of it into memory (and
    # even a single copy might be too large, e.g., for ARCO-OCEAN).

    # Slice climatology and fill missing values
    climatology = self._climatology.isel(time=get_climatology_index(dataset["time"].dt))
    climatology = climatology.fillna(value=0.0)

    # Slice anomaly_std, fill missing values and clip to avoid division by zero
    anomaly_std = self._anomaly_std.isel(time=get_climatology_index(dataset["time"].dt))
    anomaly_std = anomaly_std.fillna(value=1.0)
    anomaly_std = anomaly_std.clip(min=1e-16)

    # Compute anomaly z-score and replace original datase with it
    time_coord = dataset["time"]
    dataset_wo_time_coord = dataset.drop_vars("time")
    anomaly_zscore = (dataset_wo_time_coord - climatology) / anomaly_std
    anomaly_zscore = anomaly_zscore.assign_coords(time=time_coord)
    # Reintroduce static variables into dataset and drop depth coordinate
    dataset = dataset.drop_vars(anomaly_zscore.data_vars)
    # Normalize static numeric variables using min-max, preserving boolean masks as 0/1.
    for name, values in dataset.data_vars.items():
      if np.issubdtype(values.dtype, np.bool_):
        dataset[name] = values.astype(np.float32)
      else:
        minimum = values.min()
        span = values.max() - minimum
        dataset[name] = (values - minimum) / span.where(span != 0, 1)
    dataset = dataset.merge(anomaly_zscore)
    dataset = dataset.drop_vars("depth")

    # Introduce batch dimension and datetime coordinate
    dataset = dataset.expand_dims(dim="batch", axis=0)
    dataset = dataset.assign_coords({"datetime": dataset["time"].expand_dims(dim="batch", axis=0)})
    dataset["time"] = dataset["time"] - dataset["time"][0]
    dataset = dataset.fillna(value=self._fill_value)
    dataset = dataset.astype(np.float32)
    return dataset

  def __repr__(self):
    return (
      f"{self.__class__.__name__}(dataset={self._dataset}, "
      f"window_size={self._window_size}, "
      f"fill value={self._fill_value})"
    )

  @property
  def xarray_dataset(self):
    return self._dataset


class _FlatSource(grain.RandomAccessDataSource):
  """Expose xarray sample arrays to Grain's multiprocessing transport.

  Grain's shared-memory handling does not use the custom JAX pytree
  registrations for xarray objects. Passing datasets directly would therefore
  hide their arrays from shared-memory transport and restoration. Flattening
  with JAX exposes those arrays as ordinary list elements, while the tree
  definition retains the structure and xarray metadata needed to rebuild them.

  Pair this adapter with `_unflatten_sample` in the pipeline before applying
  transformations that expect datasets. Random access and length are unchanged.
  """

  def __init__(self, source: grain.RandomAccessDataSource):
    """Wrap a source whose samples are registered JAX pytrees."""
    self._source = source

  def __len__(self):
    return len(self._source)

  def __getitem__(self, record_key):
    """Return array leaves and reconstruction metadata for one source sample."""
    # Use JAX's registry to traverse xarray containers, not Grain's traversal.
    return jax.tree_util.tree_flatten(self._source[record_key])

  def __repr__(self) -> str:
    return f"{self.__class__.__name__}(source={self._source!r})"


def _unflatten_sample(flat_sample):
  """Rebuild a sample exposed by `_FlatSource` using its JAX tree definition."""
  leaves, treedef = flat_sample
  # The tree definition restores dataset structure and coordinates around leaves.
  return jax.tree_util.tree_unflatten(treedef, leaves)
