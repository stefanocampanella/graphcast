# SPDX-FileCopyrightText: 2026 Stefano Campanella
# SPDX-License-Identifier: MIT
"""Grain helpers for xarray pipelines.

Grain reader and multiprocessing options are runtime settings read from GRAPHCAST_GRAIN_* environment variables (see
`GrainRuntimeConfig`), not from training configuration files.

Importing this module patches Grain's output-spec recording (used by `py_debug_mode` and dataset visualization) so
that it supports xarray objects. It relies on private Grain APIs (checked with grain 0.2.18).
"""

import dataclasses
import functools
import os

import grain.python as grain
import xarray as xr
from grain._src.core import tree_lib
from grain._src.python.dataset import dataset as grain_dataset
from grain._src.python.dataset import stats as grain_stats
from grain.python import DatasetIterator
from pydantic import NonNegativeInt, PositiveInt, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

GRAIN_ENV_PREFIX = "GRAPHCAST_GRAIN_"


@dataclasses.dataclass(frozen=True)
class GrainOptions:
  """Grain options used to build a dataset iterator; no multiprocessing options means no mp_prefetch."""

  read_options: grain.ReadOptions
  multiprocessing_options: grain.MultiprocessingOptions | None = None

  @property
  def num_workers(self) -> int:
    return (
      0
      if self.multiprocessing_options is None
      else self.multiprocessing_options.num_workers
    )


class GrainRuntimeConfig(BaseSettings):
  """Grain reader and multiprocessing options, read from GRAPHCAST_GRAIN_* environment variables.

  Grain's iterator state depends on num_workers: checkpoints must be restored with the value they were saved with.
  """

  model_config = SettingsConfigDict(env_prefix=GRAIN_ENV_PREFIX, extra="forbid")

  num_threads: NonNegativeInt = 2
  prefetch_buffer_size: NonNegativeInt = 2
  num_workers: NonNegativeInt = 0
  per_worker_buffer_size: PositiveInt = 2
  enable_profiling: bool = False

  @model_validator(mode="before")
  @classmethod
  def reject_unknown_variables(cls, values):
    # Environment variables not matching a field are otherwise silently ignored, e.g. misspelled names.
    unknown = sorted(
      name
      for name in os.environ
      if name.upper().startswith(GRAIN_ENV_PREFIX)
      and name[len(GRAIN_ENV_PREFIX):].lower() not in cls.model_fields
    )
    if unknown:
      expected = ", ".join(
        GRAIN_ENV_PREFIX + field.upper() for field in cls.model_fields
      )
      raise ValueError(
        f"Unknown environment variables {', '.join(unknown)}; expected any of {expected}"
      )
    return values

  def options(self) -> GrainOptions:
    """Return Grain options; num_workers = 0 disables mp_prefetch."""
    read_options = grain.ReadOptions(
      num_threads=self.num_threads, prefetch_buffer_size=self.prefetch_buffer_size
    )
    if self.num_workers == 0:
      return GrainOptions(read_options)
    return GrainOptions(
      read_options,
      grain.MultiprocessingOptions(
        num_workers=self.num_workers,
        per_worker_buffer_size=self.per_worker_buffer_size,
        enable_profiling=self.enable_profiling,
      ),
    )


_XARRAY_TYPES = (xr.Dataset, xr.DataArray, xr.Variable)


def _xarray_spec(value: xr.Variable | xr.DataArray) -> str:
  return f"{value.dtype}{list(value.shape)} {tuple(value.dims)}"


@functools.wraps(tree_lib.spec_like)
def _spec_like(structure):
  # Grain's spec_like maps leaves to strings with jax.tree_util, then rebuilds the pytree. xarray_jax registers xarray
  # types as pytrees, and xarray rejects string data on unflatten, so describe xarray objects directly instead.
  if isinstance(structure, xr.Dataset):
    return {
      "data_vars": {name: _xarray_spec(v) for name, v in structure.data_vars.items()},
      "coords": {name: _xarray_spec(v) for name, v in structure.coords.items()},
    }
  if isinstance(structure, (xr.DataArray, xr.Variable)):
    return _xarray_spec(structure)
  if isinstance(structure, dict):
    return {k: _spec_like(v) for k, v in structure.items()}
  if isinstance(structure, (list, tuple)) and any(
      isinstance(v, _XARRAY_TYPES) for v in structure
  ):
    return type(structure).__name__, [_spec_like(v) for v in structure]
  return _original_spec_like(structure)


_original_spec_like = tree_lib.spec_like
tree_lib.spec_like = _spec_like


def format_execution_summary(iterator: DatasetIterator) -> str:
  """Formats the execution summary of a Grain iterator, sorted by wait time.

  Requires execution statistics, e.g. `grain_config.update("py_debug_mode", True)`. Grain itself only logs the summary
  every 60 s, which short runs may never reach.
  """
  summary = grain_dataset.get_execution_summary(iterator)
  summary = grain_stats.stats_utils.sort_nodes_by_wait_time_ratio(summary)
  return grain_stats.pretty_format_summary(summary)
