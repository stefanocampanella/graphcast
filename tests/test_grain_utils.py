# Copyright 2026 Stefano Campanella.
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
"""Tests of the Grain helpers (grain_utils.py): runtime settings from the environment, options, xarray specs."""

import grain.python as grain
import numpy as np
import pydantic
import pytest
import xarray as xr
from absl import flags
from grain._src.core import tree_lib

from graphcast import grain_utils


def test_settings_from_the_environment(monkeypatch):
  """GRAPHCAST_GRAIN_* variables set the reader and worker settings."""
  monkeypatch.setenv("GRAPHCAST_GRAIN_NUM_WORKERS", "3")
  monkeypatch.setenv("GRAPHCAST_GRAIN_NUM_THREADS", "5")
  monkeypatch.setenv("GRAPHCAST_GRAIN_ENABLE_PROFILING", "true")
  runtime = grain_utils.GrainRuntimeConfig()
  assert (runtime.num_workers, runtime.num_threads, runtime.enable_profiling) == (3, 5, True)


def test_unknown_variables_are_rejected(monkeypatch):
  """A misspelled variable is an error naming the expected ones, not silently ignored."""
  monkeypatch.setenv("GRAPHCAST_GRAIN_NUM_WORKER", "3")
  with pytest.raises(pydantic.ValidationError, match="GRAPHCAST_GRAIN_NUM_WORKER.*expected any of"):
    grain_utils.GrainRuntimeConfig()


@pytest.mark.parametrize("num_workers", [0, 2])
def test_options(num_workers):
  """Reader options always; multiprocessing options only with workers (0 disables mp_prefetch)."""
  options = grain_utils.GrainRuntimeConfig(num_workers=num_workers, num_threads=3, per_worker_buffer_size=4).options()
  assert options.read_options.num_threads == 3
  assert options.num_workers == num_workers
  if num_workers:
    assert options.multiprocessing_options.per_worker_buffer_size == 4
  else:
    assert options.multiprocessing_options is None


def test_spec_of_xarray_objects():
  """Grain's output specs describe xarray objects by dtype, shape and dimensions (Grain's own version failed on them),
  inside dicts and tuples; other structures keep Grain's description."""
  dataset = xr.Dataset({"x": (("time", "lon"), np.zeros((2, 3), np.float32))}, coords={"lon": [0, 1, 2]})
  spec = tree_lib.spec_like({"sample": (dataset, dataset["x"]), "n": np.zeros(4)})
  assert spec["sample"] == (
    "tuple",
    [
      {"data_vars": {"x": "float32[2, 3] ('time', 'lon')"}, "coords": {"lon": "int64[3] ('lon',)"}},
      "float32[2, 3] ('time', 'lon')",
    ],
  )
  assert spec["n"] == tree_lib.spec_like(np.zeros(4)) == grain_utils._original_spec_like(np.zeros(4))


def test_format_execution_summary():
  """The execution summary of an iterator with statistics (Grain's debug mode, an absl flag) lists its stages."""
  flags.FLAGS.mark_as_parsed()
  debug_mode = flags.FLAGS.grain_py_debug_mode
  flags.FLAGS.grain_py_debug_mode = True
  try:
    iterator = iter(grain.MapDataset.range(8).map(lambda x: x * 2).to_iter_dataset())
    assert [next(iterator) for _ in range(3)] == [0, 2, 4]
    summary = grain_utils.format_execution_summary(iterator)
  finally:
    flags.FLAGS.grain_py_debug_mode = debug_mode
  assert "Map" in summary
