# SPDX-FileCopyrightText: 2023 DeepMind Technologies Limited
# SPDX-FileCopyrightText: 2026 Stefano Campanella
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
#
# Ported from the absltest tests of the original GraphCast code to pytest.
"""Tests for xarray_tree."""

import numpy as np
import xarray

from mobydick import xarray_tree

TEST_DATASET = xarray.Dataset(
  data_vars={
    "foo": (("x", "y"), np.zeros((2, 3))),
    "bar": (("x",), np.zeros((2,))),
  },
  coords={
    "x": [1, 2],
    "y": [10, 20, 30],
  },
)


def test_map_structure_maps_over_leaves_but_preserves_dataset_type():
  """Mapping over a `Dataset` visits each variable and returns a `Dataset`, restoring the names even if the function
  drops them."""

  def fn(leaf):
    assert isinstance(leaf, xarray.DataArray)
    result = leaf + 1
    # Removing the name from the returned DataArray to test that we don't rely
    # on it being present to restore the correct names in the result:
    result = result.rename(None)
    return result

  result = xarray_tree.map_structure(fn, TEST_DATASET)
  assert isinstance(result, xarray.Dataset)
  assert set(result.keys()) == {"foo", "bar"}


def test_map_structure_on_data_arrays():
  """Mapping over a dict of `DataArray`s returns a dict."""
  data_arrays = dict(TEST_DATASET)
  result = xarray_tree.map_structure(lambda x: x + 1, data_arrays)
  assert isinstance(result, dict)
  assert set(result.keys()) == {"foo", "bar"}


def test_map_structure_on_dataset_plain_dict_when_coords_incompatible():
  """If the results cannot be merged back into a `Dataset` (conflicting coordinates), a dict is returned."""

  def fn(leaf):
    # Returns DataArrays that can't be exactly merged back into a Dataset
    # due to the coordinates not matching:
    if leaf.name == "foo":
      return xarray.DataArray(data=np.zeros(2), dims=("x",), coords={"x": [1, 2]})
    else:
      return xarray.DataArray(data=np.zeros(2), dims=("x",), coords={"x": [3, 4]})

  result = xarray_tree.map_structure(fn, TEST_DATASET)
  assert isinstance(result, dict)
  assert set(result.keys()) == {"foo", "bar"}


def test_map_structure_on_dataset_drops_vars_with_none_return_values():
  """Variables for which the function returns `None` are dropped from the resulting `Dataset`."""

  def fn(leaf):
    return leaf if leaf.name == "foo" else None

  result = xarray_tree.map_structure(fn, TEST_DATASET)
  assert isinstance(result, xarray.Dataset)
  assert set(result.keys()) == {"foo"}


def test_map_structure_on_dataset_returns_plain_dict_other_return_types():
  """If the function returns something other than `DataArray`s, a dict of the results is returned."""

  def fn(leaf):
    assert isinstance(leaf, xarray.DataArray)
    return "not a DataArray"

  result = xarray_tree.map_structure(fn, TEST_DATASET)
  assert result == {"foo": "not a DataArray", "bar": "not a DataArray"}


def test_map_structure_two_args_different_variable_orders():
  """With two `Dataset`s, variables are matched by name, whatever their order."""
  dataset_different_order = TEST_DATASET[["bar", "foo"]]

  def fn(arg1, arg2):
    assert arg1.name == arg2.name

  xarray_tree.map_structure(fn, TEST_DATASET, dataset_different_order)
