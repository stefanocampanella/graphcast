import operator
from typing import SupportsIndex

import grain.python as grain
import jax
import tensorstore
import xarray as xr
import xarray_tensorstore

InputsTargetsForcings = tuple[xr.Dataset, xr.Dataset, xr.Dataset]

# Zero-based depth positions, in the order of the selected levels.
LevelSelection = list[int] | slice


def get_tensorstore_context(io_threads: int, aws_credentials: dict | None = None) -> tensorstore.Context:
  """TensorStore context limiting file I/O, S3 requests and decoding/encoding to io_threads each."""
  resources = {
    "data_copy_concurrency": {"limit": io_threads},
    "file_io_concurrency": {"limit": io_threads},
    "s3_request_concurrency": {"limit": io_threads},
  }
  if aws_credentials is not None:
    resources["aws_credentials"] = aws_credentials
  return tensorstore.Context(resources)


def select_levels(dataset: xr.Dataset, levels: LevelSelection) -> xr.Dataset:
  """Label the original depth positions as the `level` dimension, then select them in the given order."""
  dataset = dataset.assign_coords(level=("depth", range(dataset.sizes["depth"])))
  dataset = dataset.swap_dims({"depth": "level"})
  dataset = dataset.sel(level=levels)
  if dataset.sizes["level"] == 0:
    raise ValueError(f"Level selection {levels!r} resolved to zero levels")
  return dataset


def open_standardized_anomalies(
  path: str,
  levels: LevelSelection,
  end_date: str | None = None,
  io_threads: int = 16,
) -> xr.Dataset:
  """Open a standardized anomaly store (written by graphcast download) lazily, with TensorStore.

  Only the selected levels and the timesteps up to end_date (included) are kept.
  """
  context = get_tensorstore_context(io_threads)
  dataset = xarray_tensorstore.open_zarr(path, context=context, mask_and_scale=False, consolidated=True)
  return select_levels(dataset, levels).sel(time=slice(None, end_date))


class StandardizedAnomalySource(grain.RandomAccessDataSource):
  """A data source returning windows of a standardized anomaly store, with a batch dimension and relative time.

  The store is opened lazily in each process: pickled sources (e.g. sent to Grain workers) reopen it on first access.
  """

  def __init__(
    self,
    path: str,
    *,
    levels: LevelSelection,
    window_size: int,
    end_date: str | None = None,
    io_threads: int = 16,
  ):
    self._path = path
    self._levels = levels
    self._end_date = end_date
    self._io_threads = io_threads
    self._dataset = None
    dataset = self.dataset
    window_size = operator.index(window_size)
    if not 0 < window_size <= dataset.sizes["time"]:
      raise ValueError("window_size must be positive and no larger than the dataset time dimension.")
    self._window_size = window_size

  @property
  def dataset(self) -> xr.Dataset:
    """The lazily opened store, with the selected levels and timesteps."""
    if self._dataset is None:
      self._dataset = open_standardized_anomalies(self._path, self._levels, self._end_date, self._io_threads)
    return self._dataset

  def __getstate__(self):
    return {**self.__dict__, "_dataset": None}

  def __len__(self):
    return self.dataset.sizes["time"] - self._window_size + 1

  def __getitem__(self, record_key: SupportsIndex) -> xr.Dataset:
    """Return the window of `_window_size` timesteps starting at `record_key`."""
    idx = record_key.__index__()
    window = self.dataset.isel(time=slice(idx, idx + self._window_size))
    # Read all variables concurrently.
    window = xarray_tensorstore.read(window).compute()
    window = window.expand_dims(dim="batch", axis=0)
    window = window.assign_coords({"datetime": window["time"].expand_dims(dim="batch", axis=0)})
    return window.assign_coords(time=window["time"] - window["time"][0])

  def __repr__(self):
    return (
      f"{self.__class__.__name__}(path={self._path!r}, levels={self._levels!r}, end_date={self._end_date!r}, "
      f"window_size={self._window_size})"
    )


class FlatSource(grain.RandomAccessDataSource):
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
