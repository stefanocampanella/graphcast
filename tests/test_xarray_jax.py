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
# Ported from the absltest tests of the original GraphCast code to pytest and updated for this fork.

import chex
import jax
import jax.numpy as jnp
import numpy as np
import pytest
import xarray

from graphcast import xarray_jax


def test_jax_array_wrapper_with_numpy_api():
  """numpy operations on a `JaxArrayWrapper` stay in JAX (the result is still wrapped); `np.asarray` leaves it."""
  # This is just a side benefit of making things work with xarray, but the
  # JaxArrayWrapper does allow you to manipulate JAX arrays using the
  # standard numpy API, without converting them to numpy in the process:
  ones = jnp.ones((3, 4), dtype=np.float32)
  x = xarray_jax.JaxArrayWrapper(ones)
  x = np.abs((x + 2) * (x - 3))
  x = x[:-1, 1:3]
  x = np.concatenate([x, x + 1], axis=0)
  x = np.transpose(x, (1, 0))
  x = np.reshape(x, (-1,))
  x = x.astype(np.int32)
  assert isinstance(x, xarray_jax.JaxArrayWrapper)
  # An explicit conversion gets us out of JAX-land however:
  assert isinstance(np.asarray(x), np.ndarray)


def test_jax_xarray_variable():
  """A chain of xarray operations on a JAX-backed `Variable` keeps JAX data, also under `jit` and `grad`."""

  def ops_via_xarray(inputs):
    x = xarray_jax.Variable(("lat", "lon"), inputs)
    # We'll apply a sequence of operations just to test that the end result is
    # still a JAX array, i.e. we haven't converted to numpy at any point.
    x = np.abs((x + 2) * (x - 3))
    x = x.isel({"lat": slice(0, -1), "lon": slice(1, 3)})  # ty: ignore[unresolved-attribute]  # np.abs of xarray
    x = xarray.Variable.concat([x, x + 1], dim="lat")
    x = x.transpose("lon", "lat")
    x = x.stack(channels=("lon", "lat"))
    x = x.sum()
    return xarray_jax.jax_data(x)

  # Check it doesn't leave jax-land when passed concrete values:
  ones = jnp.ones((3, 4), dtype=np.float32)
  result = ops_via_xarray(ones)
  assert isinstance(result, jax.Array)

  # And that you can JIT it and compute gradients through it. These will
  # involve passing jax tracers through the xarray computation:
  jax.jit(ops_via_xarray)(ones)
  jax.grad(ops_via_xarray)(ones)


def test_jax_xarray_data_array():
  """A chain of xarray operations (selection, concat, stack) on a JAX-backed `DataArray` keeps JAX data, also under
  `jit` and `grad`."""

  def ops_via_xarray(inputs):
    x = xarray_jax.DataArray(
      dims=("lat", "lon"),
      data=inputs,
      coords={"lat": np.arange(3) * 10, "lon": np.arange(4) * 10},
    )
    x = np.abs((x + 2) * (x - 3))
    x = x.sel({"lat": slice(0, 20)})  # ty: ignore[unresolved-attribute]  # np.abs of xarray
    y = xarray_jax.DataArray(
      dims=("lat", "lon"),
      data=ones,
      coords={"lat": np.arange(3, 6) * 10, "lon": np.arange(4) * 10},
    )
    x = xarray.concat([x, y], dim="lat")
    x = x.transpose("lon", "lat")
    x = x.stack(channels=("lon", "lat"))
    x = x.unstack()
    x = x.sum()
    return xarray_jax.jax_data(x)

  ones = jnp.ones((3, 4), dtype=np.float32)
  result = ops_via_xarray(ones)
  assert isinstance(result, jax.Array)

  jax.jit(ops_via_xarray)(ones)
  jax.grad(ops_via_xarray)(ones)


def test_jax_xarray_dataset():
  """A chain of xarray operations on a JAX-backed `Dataset` keeps JAX data, also under `jit` and `grad`."""

  def ops_via_xarray(foo, bar):
    x = xarray_jax.Dataset(
      data_vars={"foo": (("lat", "lon"), foo), "bar": (("time", "lat", "lon"), bar)},
      coords={"time": np.arange(2), "lat": np.arange(3) * 10, "lon": np.arange(4) * 10},
    )
    x = np.abs((x + 2) * (x - 3))
    x = x.sel({"lat": slice(0, 20)})
    y = xarray_jax.Dataset(
      data_vars={"foo": (("lat", "lon"), foo), "bar": (("time", "lat", "lon"), bar)},
      coords={
        "time": np.arange(2),
        "lat": np.arange(3, 6) * 10,
        "lon": np.arange(4) * 10,
      },
    )
    x = xarray.concat([x, y], dim="lat")
    x = x.transpose("lon", "lat", "time")
    x = x.stack(channels=("lon", "lat"))
    x = (x.foo + x.bar).sum()
    return xarray_jax.jax_data(x)

  foo = jnp.ones((3, 4), dtype=np.float32)
  bar = jnp.ones((2, 3, 4), dtype=np.float32)
  result = ops_via_xarray(foo, bar)
  assert isinstance(result, jax.Array)

  jax.jit(ops_via_xarray)(foo, bar)
  jax.grad(ops_via_xarray)(foo, bar)


def test_jit_function_with_xarray_variable_arguments_and_return(subtests):
  """Jitted functions take and return `Variable`s (JAX- or numpy-backed) and can be called twice without retracing
  problems (the pytree aux data is comparable)."""
  function = jax.jit(lambda v: v + 1)
  with subtests.test("jax input"):
    inputs = xarray_jax.Variable(("lat", "lon"), jnp.ones((3, 4), dtype=np.float32))
    _ = function(inputs)
    # We test running the jitted function a second time, to exercise logic in
    # jax which checks if the structure of the inputs (including dimension
    # names and coordinates) is the same as it was for the previous call and
    # so whether it needs to re-trace-and-compile a new version of the
    # function or not. This can run into problems if the 'aux' structure
    # returned by the registered flatten function is not hashable/comparable.
    outputs = function(inputs)
    assert outputs.dims == inputs.dims
  with subtests.test("numpy input"):
    inputs = xarray.Variable(("lat", "lon"), np.ones((3, 4), dtype=np.float32))
    _ = function(inputs)
    outputs = function(inputs)
    assert outputs.dims == inputs.dims


def test_jit_ahead_of_time_compile_with_xarray():
  """Ahead-of-time compilation (`lower(...).compile()`) works with xarray arguments."""
  # This needs jax.stages.ArgInfo to be wrapped, since .lower maps the xarray
  # to a pytree of ArgInfo under the hood.
  function = jax.jit(lambda v: v + 1)
  inputs = xarray_jax.Variable(("lat", "lon"), jnp.ones((3, 4), dtype=np.float32))
  compiled_function = function.lower(inputs).compile()
  outputs = compiled_function(inputs)
  assert outputs.dims == inputs.dims


def test_jit_problem_if_convert_to_plain_numpy_array():
  """`.values` inside `jit` converts to numpy and fails with a tracer conversion error."""
  inputs = xarray_jax.DataArray(data=jnp.ones((2,), dtype=np.float32), dims=("foo",))
  with pytest.raises(jax.errors.TracerArrayConversionError):
    # Calling .values on a DataArray converts its values to numpy:
    jax.jit(lambda data_array: data_array.values)(inputs)


def test_grad_function_with_xarray_variable_arguments():
  """`grad` differentiates through a `Variable` argument."""
  x = xarray_jax.Variable(("lat", "lon"), jnp.ones((3, 4), dtype=np.float32))
  # For grad we still need a JAX scalar as the output:
  jax.grad(lambda v: xarray_jax.jax_data(v.sum()))(x)


def test_jit_function_with_xarray_data_array_arguments_and_return():
  """Jitted functions take and return `DataArray`s, keeping dimensions and coordinates."""
  inputs = xarray_jax.DataArray(
    data=jnp.ones((3, 4), dtype=np.float32),
    dims=("lat", "lon"),
    coords={"lat": np.arange(3), "lon": np.arange(4) * 10},
  )
  fn = jax.jit(lambda v: v + 1)
  _ = fn(inputs)
  outputs = fn(inputs)
  assert outputs.dims == inputs.dims
  chex.assert_trees_all_equal(outputs.coords, inputs.coords)


def test_jit_function_with_data_array_and_jax_coords():
  """JAX coordinates of a `DataArray` stay JAX arrays without an index, inside and out of `jit`, and can be used and
  updated in the computation; numpy coordinates keep their index."""
  inputs = xarray_jax.DataArray(
    data=jnp.ones((3, 4), dtype=np.float32),
    dims=("lat", "lon"),
    coords={"lat": np.arange(3)},
    jax_coords={"lon": jnp.arange(4) * 10},
  )
  # Verify the jax_coord 'lon' retains jax data, and has not been created
  # as an index coordinate:
  assert isinstance(inputs.coords["lon"].data, xarray_jax.JaxArrayWrapper)
  assert "lon" not in inputs.indexes

  @jax.jit
  def fn(v):
    # The non-JAX coord is passed with numpy array data and an index:
    assert isinstance(v.coords["lat"].data, np.ndarray)
    assert "lat" in v.indexes

    # The jax_coord is passed with JAX array data:
    assert isinstance(v.coords["lon"].data, xarray_jax.JaxArrayWrapper)
    assert "lon" not in v.indexes

    # Use the jax coord in the computation:
    v = v + v.coords["lon"]

    # Return with an updated jax coord:
    return xarray_jax.assign_jax_coords(v, lon=v.coords["lon"] + 1)

  _ = fn(inputs)
  outputs = fn(inputs)

  # Verify the jax_coord 'lon' has jax data in the output too:
  assert isinstance(outputs.coords["lon"].data, xarray_jax.JaxArrayWrapper)
  assert "lon" not in outputs.indexes

  assert outputs.dims == inputs.dims
  chex.assert_trees_all_equal(outputs.coords["lat"], inputs.coords["lat"])
  # Check our computations with the coordinate values worked:
  chex.assert_trees_all_equal(outputs.coords["lon"].data, (inputs.coords["lon"] + 1).data)
  chex.assert_trees_all_equal(outputs.data, (inputs + inputs.coords["lon"]).data)


def test_jit_function_with_xarray_dataset_arguments_and_return():
  """Jitted functions take and return `Dataset`s, keeping variables, dimensions and coordinates."""
  foo = jnp.ones((3, 4), dtype=np.float32)
  bar = jnp.ones((2, 3, 4), dtype=np.float32)
  inputs = xarray_jax.Dataset(
    data_vars={"foo": (("lat", "lon"), foo), "bar": (("time", "lat", "lon"), bar)},
    coords={"time": np.arange(2), "lat": np.arange(3) * 10, "lon": np.arange(4) * 10},
  )
  fn = jax.jit(lambda v: v + 1)
  _ = fn(inputs)
  outputs = fn(inputs)
  assert outputs.data_vars.keys() == {"foo", "bar"}
  assert inputs.foo.dims == outputs.foo.dims
  assert inputs.bar.dims == outputs.bar.dims
  chex.assert_trees_all_equal(outputs.coords, inputs.coords)


def test_jit_function_with_xarray_datatree_arguments_and_return():
  """Jitted functions take and return `DataTree`s, with JAX coordinates inherited from the parent."""
  parent_dataset = xarray_jax.Dataset(
    jax_coords={"time": xarray_jax.Variable(("time",), np.arange(2))},
    coords={"lon": xarray.Variable(("lon",), np.arange(4) * 10)},
  )

  bar = jnp.ones((2, 3, 4), dtype=np.float32)
  child_dataset = xarray_jax.Dataset({"bar": (("time", "lat", "lon"), bar)}, coords={"lat": np.arange(3)})

  inputs = xarray.DataTree(dataset=parent_dataset, children={"child": xarray.DataTree(dataset=child_dataset)})

  @jax.jit
  def fn(inputs):
    return xarray.DataTree(
      dataset=xarray_jax.assign_jax_coords(inputs.to_dataset(), time=inputs.time + 1),
      children={"child": xarray.DataTree(dataset=inputs.children["child"].to_dataset() + 1)},
    )

  _ = fn(inputs)
  outputs = fn(inputs)
  assert outputs.children.keys() == {"child"}
  assert outputs.coords.keys() == {"time", "lon"}
  assert isinstance(outputs.coords["time"].data, xarray_jax.JaxArrayWrapper)
  assert outputs.child.data_vars.keys() == {"bar"}


def test_jit_function_with_dataset_and_jax_coords():
  """JAX coordinates of a `Dataset` stay JAX arrays without an index, inside and out of `jit`, and can be used and
  updated in the computation."""
  foo = jnp.ones((3, 4), dtype=np.float32)
  bar = jnp.ones((2, 3, 4), dtype=np.float32)
  inputs = xarray_jax.Dataset(
    data_vars={"foo": (("lat", "lon"), foo), "bar": (("time", "lat", "lon"), bar)},
    coords={
      "time": np.arange(2),
      "lat": np.arange(3) * 10,
    },
    jax_coords={"lon": jnp.arange(4) * 10},
  )
  # Verify the jax_coord 'lon' retains jax data, and has not been created
  # as an index coordinate:
  assert isinstance(inputs.coords["lon"].data, xarray_jax.JaxArrayWrapper)
  assert "lon" not in inputs.indexes

  @jax.jit
  def fn(v):
    # The non-JAX coords are passed with numpy array data and an index:
    assert isinstance(v.coords["lat"].data, np.ndarray)
    assert "lat" in v.indexes

    # The jax_coord is passed with JAX array data:
    assert isinstance(v.coords["lon"].data, xarray_jax.JaxArrayWrapper)
    assert "lon" not in v.indexes

    # Use the jax coord in the computation:
    v = v + v.coords["lon"]

    # Return with an updated jax coord:
    return xarray_jax.assign_jax_coords(v, lon=v.coords["lon"] + 1)

  _ = fn(inputs)
  outputs = fn(inputs)

  # Verify the jax_coord 'lon' has jax data in the output too:
  assert isinstance(outputs.coords["lon"].data, xarray_jax.JaxArrayWrapper)
  assert "lon" not in outputs.indexes

  assert outputs.sizes == inputs.sizes
  chex.assert_trees_all_equal(outputs.coords["lat"], inputs.coords["lat"])
  # Check our computations with the coordinate values worked:
  chex.assert_trees_all_equal(
    (outputs.coords["lon"]).data,
    (inputs.coords["lon"] + 1).data,
  )
  outputs_dict = {key: outputs[key].data for key in outputs}
  inputs_and_inputs_coords_dict = {
    key: (inputs + inputs.coords["lon"])[key].data for key in inputs + inputs.coords["lon"]
  }
  chex.assert_trees_all_equal(outputs_dict, inputs_and_inputs_coords_dict)


def test_flatten_unflatten_variable():
  """A `Variable` round-trips through flatten/unflatten, with hashable, comparable aux data (for `jit`'s cache)."""
  variable = xarray_jax.Variable(("lat", "lon"), jnp.ones((3, 4), dtype=np.float32))
  children, aux = xarray_jax._flatten_variable(variable)
  # Check auxiliary info is hashable/comparable (important for jax.jit):
  hash(aux)
  assert aux == aux  # noqa: PLR0124 (checks that aux supports ==)
  roundtrip = xarray_jax._unflatten_variable(aux, children)
  assert variable.equals(roundtrip)


def test_flatten_unflatten_data_array():
  """A `DataArray` with numpy and JAX coordinates round-trips through flatten/unflatten, with hashable aux data."""
  data_array = xarray_jax.DataArray(
    data=jnp.ones((3, 4), dtype=np.float32),
    dims=("lat", "lon"),
    coords={"lat": np.arange(3)},
    jax_coords={"lon": np.arange(4) * 10},
  )
  children, aux = xarray_jax._flatten_data_array(data_array)
  # Check auxiliary info is hashable/comparable (important for jax.jit):
  hash(aux)
  assert aux == aux  # noqa: PLR0124 (checks that aux supports ==)
  roundtrip = xarray_jax._unflatten_data_array(aux, children)
  assert data_array.equals(roundtrip)


def test_flatten_unflatten_dataset():
  """A `Dataset` with numpy and JAX coordinates round-trips through flatten/unflatten, with hashable aux data."""
  foo = jnp.ones((3, 4), dtype=np.float32)
  bar = jnp.ones((2, 3, 4), dtype=np.float32)
  dataset = xarray_jax.Dataset(
    data_vars={"foo": (("lat", "lon"), foo), "bar": (("time", "lat", "lon"), bar)},
    coords={"time": np.arange(2), "lat": np.arange(3) * 10},
    jax_coords={"lon": np.arange(4) * 10},
  )
  children, aux = xarray_jax._flatten_dataset(dataset)
  # Check auxiliary info is hashable/comparable (important for jax.jit):
  hash(aux)
  assert aux == aux  # noqa: PLR0124 (checks that aux supports ==)
  roundtrip = xarray_jax._unflatten_dataset(aux, children)
  assert dataset.equals(roundtrip)


def test_flatten_unflatten_datatree():
  """A `DataTree` with inherited numpy and JAX coordinates round-trips through flatten/unflatten."""
  # Coords to be inherited from the parent dataset, we include one jax
  # coord and one not to check both code paths
  parent_dataset = xarray_jax.Dataset(
    jax_coords={"time": xarray_jax.Variable(("time",), np.arange(2))},
    coords={"lon": xarray.Variable(("lon",), np.arange(4) * 10)},
  )

  bar = jnp.ones((2, 3, 4), dtype=np.float32)
  child_dataset = xarray_jax.Dataset({"bar": (("time", "lat", "lon"), bar)}, coords={"lat": np.arange(3)})

  datatree = xarray.DataTree(dataset=parent_dataset, children={"child": xarray.DataTree(dataset=child_dataset)})

  children, aux = xarray_jax._flatten_datatree(datatree)
  # Check auxiliary info is hashable/comparable (important for jax.jit):
  hash(aux)
  assert aux == aux  # noqa: PLR0124 (checks that aux supports ==)
  roundtrip = xarray_jax._unflatten_datatree(aux, children)
  assert datatree.equals(roundtrip)


def test_flatten_unflatten_added_dim():
  """`dims_change_on_unflatten` names a leading dimension added to the leaves between flatten and unflatten."""
  data_array = xarray_jax.DataArray(
    data=jnp.ones((3, 4), dtype=np.float32),
    dims=("lat", "lon"),
    coords={"lat": np.arange(3), "lon": np.arange(4) * 10},
  )
  leaves, treedef = jax.tree_util.tree_flatten(data_array)
  leaves = [jnp.expand_dims(x, 0) for x in leaves]
  with xarray_jax.dims_change_on_unflatten(lambda dims: ("new",) + dims):
    with_new_dim = jax.tree_util.tree_unflatten(treedef, leaves)
  assert with_new_dim.dims == ("new", "lat", "lon")
  xarray.testing.assert_identical(jax.device_get(data_array), jax.device_get(with_new_dim.isel(new=0)))


def test_map_added_dim():
  """`tree_map` adding a leading axis gives a `DataArray` with the new dimension named by `dims_change_on_unflatten`."""
  data_array = xarray_jax.DataArray(
    data=jnp.ones((3, 4), dtype=np.float32),
    dims=("lat", "lon"),
    coords={"lat": np.arange(3), "lon": np.arange(4) * 10},
  )
  with xarray_jax.dims_change_on_unflatten(lambda dims: ("new",) + dims):
    with_new_dim = jax.tree_util.tree_map(lambda x: jnp.expand_dims(x, 0), data_array)
  assert with_new_dim.dims == ("new", "lat", "lon")
  xarray.testing.assert_identical(jax.device_get(data_array), jax.device_get(with_new_dim.isel(new=0)))


def test_map_remove_dim():
  """`tree_map` removing a leading axis gives a `Dataset` without that dimension and its coordinate."""
  foo = jnp.ones((1, 3, 4), dtype=np.float32)
  bar = jnp.ones((1, 2, 3, 4), dtype=np.float32)
  dataset = xarray_jax.Dataset(
    data_vars={
      "foo": (("batch", "lat", "lon"), foo),
      "bar": (("batch", "time", "lat", "lon"), bar),
    },
    coords={
      "batch": np.array([123]),
      "time": np.arange(2),
      "lat": np.arange(3) * 10,
      "lon": np.arange(4) * 10,
    },
  )
  with xarray_jax.dims_change_on_unflatten(lambda dims: dims[1:]):
    with_removed_dim = jax.tree_util.tree_map(lambda x: jnp.squeeze(x, 0), dataset)
  assert with_removed_dim["foo"].dims == ("lat", "lon")
  assert with_removed_dim["bar"].dims == ("time", "lat", "lon")
  assert "batch" not in with_removed_dim.dims
  assert "batch" not in with_removed_dim.coords
  xarray.testing.assert_identical(jax.device_get(dataset.isel(batch=0, drop=True)), jax.device_get(with_removed_dim))


def test_pmap():
  """`xarray_jax.pmap` maps over a named leading dimension of a `Dataset`, and can be called again with a different
  structure."""
  devices = jax.local_device_count()
  foo = jnp.zeros((devices, 3, 4), dtype=np.float32)
  bar = jnp.zeros((devices, 2, 3, 4), dtype=np.float32)
  dataset = xarray_jax.Dataset({"foo": (("device", "lat", "lon"), foo), "bar": (("device", "time", "lat", "lon"), bar)})

  def func(d):
    assert "device" not in d.dims
    return d + 1

  func = xarray_jax.pmap(func, dim="device")  # ty: ignore[invalid-assignment]

  result = func(dataset)
  xarray.testing.assert_identical(jax.device_get(dataset + 1), jax.device_get(result))

  # Can call it again with a different argument structure (it will recompile
  # under the hood but should work):
  dataset = dataset.drop_vars("foo")
  result = func(dataset)
  xarray.testing.assert_identical(jax.device_get(dataset + 1), jax.device_get(result))


def test_pmap_with_jax_coords():
  """JAX coordinates with the device dimension are split across devices too, and stay JAX arrays without an index inside
  the mapped function."""
  devices = jax.local_device_count()
  foo = jnp.zeros((devices, 3, 4), dtype=np.float32)
  bar = jnp.zeros((devices, 2, 3, 4), dtype=np.float32)
  time = jnp.zeros((devices, 2), dtype=np.float32)
  dataset = xarray_jax.Dataset(
    {
      "foo": (("device", "lat", "lon"), foo),
      "bar": (("device", "time", "lat", "lon"), bar),
    },
    coords={
      "lat": np.arange(3),
      "lon": np.arange(4),
    },
    jax_coords={
      # Currently any jax_coords need a leading device dimension to use
      # with pmap, same as for data_vars.
      # TODO(matthjw): have pmap automatically broadcast to all devices
      # where the device dimension not present.
      "time": xarray_jax.Variable(("device", "time"), time),
    },
  )

  def func(d):
    assert "device" not in d.dims
    assert "device" not in d.coords["time"].dims

    # The jax_coord 'time' should be passed in backed by a JAX array, but
    # not as an index coordinate.
    assert isinstance(d.coords["time"].data, xarray_jax.JaxArrayWrapper)
    assert "time" not in d.indexes

    return d + 1

  func = xarray_jax.pmap(func, dim="device")  # ty: ignore[invalid-assignment]

  result = func(dataset)
  xarray.testing.assert_identical(jax.device_get(dataset + 1), jax.device_get(result))

  # Can call it again with a different argument structure (it will recompile
  # under the hood but should work):
  dataset = dataset.drop_vars("foo")
  result = func(dataset)
  xarray.testing.assert_identical(jax.device_get(dataset + 1), jax.device_get(result))


def test_pmap_with_tree_mix_of_xarray_and_jax_array():
  """`pmap` accepts trees mixing `DataArray`s and plain JAX arrays."""
  devices = jax.local_device_count()
  data_array = xarray_jax.DataArray(data=jnp.ones((devices, 3, 4), dtype=np.float32), dims=("device", "lat", "lon"))
  plain_array = jnp.ones((devices, 2), dtype=np.float32)
  inputs = {"foo": data_array, "bar": plain_array}

  def func(x):
    return x["foo"] + 1, x["bar"] + 1

  func = xarray_jax.pmap(func, dim="device")  # ty: ignore[invalid-assignment]
  result_foo, result_bar = func(inputs)
  xarray.testing.assert_identical(jax.device_get(inputs["foo"] + 1), jax.device_get(result_foo))
  np.testing.assert_array_equal(jax.device_get(inputs["bar"] + 1), jax.device_get(result_bar))


@pytest.mark.parametrize("structure", ["data_array", "dataset_jax_coord"])
def test_pmap_complains_when_dim_not_first(structure):
  """`pmap` rejects a mapped dimension that is not the leading one with its own message, before calling `jax.pmap`
  (inside the mapped function the check came after jax had already failed with its own error)."""
  devices = jax.local_device_count()
  if structure == "data_array":
    inputs = xarray_jax.DataArray(data=jnp.ones((3, devices, 4), dtype=np.float32), dims=("lat", "device", "lon"))
  else:
    inputs = xarray_jax.Dataset(
      {"foo": (("device", "lat"), jnp.ones((devices, 3), dtype=np.float32))},
      jax_coords={"time": xarray_jax.Variable(("time", "device"), jnp.zeros((2, devices), dtype=np.float32))},
    )

  func = xarray_jax.pmap(lambda x: x + 1, dim="device")
  with pytest.raises(ValueError, match="Expected dim device at index 0, found at 1"):
    func(inputs)


def test_apply_ufunc():
  """`apply_ufunc` applies a JAX function over a core dimension and returns a `DataArray`."""
  inputs = xarray_jax.DataArray(data=jnp.asarray([[1, 2], [3, 4]]), dims=("x", "y"), coords={"x": [0, 1], "y": [2, 3]})
  result = xarray_jax.apply_ufunc(lambda x: jnp.sum(x, axis=-1), inputs, input_core_dims=[["x"]])
  expected_result = xarray_jax.DataArray(data=[4, 6], dims=("y",), coords={"y": [2, 3]})
  xarray.testing.assert_identical(expected_result, jax.device_get(result))


def test_apply_ufunc_multiple_return_values():
  """`apply_ufunc` with a function returning a tuple gives a tuple of `DataArray`s."""

  def ufunc(array):
    return jnp.min(array, axis=-1), jnp.max(array, axis=-1)

  inputs = xarray_jax.DataArray(data=jnp.asarray([[1, 4], [3, 2]]), dims=("x", "y"), coords={"x": [0, 1], "y": [2, 3]})
  result = xarray_jax.apply_ufunc(ufunc, inputs, input_core_dims=[["x"]], output_core_dims=[[], []])
  expected = (
    # Mins:
    xarray_jax.DataArray(data=[1, 2], dims=("y",), coords={"y": [2, 3]}),
    # Maxes:
    xarray_jax.DataArray(data=[3, 4], dims=("y",), coords={"y": [2, 3]}),
  )
  xarray.testing.assert_identical(expected[0], jax.device_get(result[0]))
  xarray.testing.assert_identical(expected[1], jax.device_get(result[1]))


def test_eval_shape_with_xarray():
  """`jax.eval_shape` on a function of xarray returns xarray objects holding shapes and dtypes."""
  # This needs jax.ShapeDtypeStruct to be wrappable inside xarray types.
  function = jax.jit(lambda v: v + 1)
  inputs = xarray_jax.Variable(("lat", "lon"), jnp.ones((3, 4), dtype=np.float32))
  output_shapes = jax.eval_shape(function, inputs)
  assert isinstance(output_shapes, xarray.Variable)
  assert output_shapes.shape == (3, 4)
  assert output_shapes.dtype == np.float32


def test_scan():
  """`xarray_jax.scan` scans over a named dimension in any position, with mixed xarray and JAX inputs; numpy coordinates
  along it are dropped, JAX coordinates kept."""

  def f(carry, x):
    dataset_input, plain_jax_input = x
    assert plain_jax_input.shape == ()
    assert dataset_input.sizes == {"x": 2}  # No 'time' dimension.
    carry = carry + 1
    y = dataset_input + carry
    return carry, y

  dataset_inputs = xarray_jax.Dataset(
    data_vars={
      # Put the scan dimension (time) second to make sure it transposes
      # the array appropriately for JAX which requires scan dim first.
      "foo": (("x", "time"), jnp.zeros((2, 5))),
    },
    coords={
      "x": np.arange(2),
      "time": np.arange(5) * 10,
    },
    jax_coords={
      "time_extra": (("time",), np.arange(5) * 2),
    },
  )
  # These must have scan dimension first:
  plain_jax_inputs = jnp.zeros((5,))

  carry, result = xarray_jax.scan(f, init=0, xs=(dataset_inputs, plain_jax_inputs), dim="time")
  assert jax.device_get(carry) == 5
  assert result.foo.sizes == {"x": 2, "time": 5}
  assert "x" in result.coords
  # Unfortunately static coordinates along the dimension we scan over will
  # not be preserved on the result:
  assert "time" not in result.coords
  # The jax_coord along time will still be present though:
  np.testing.assert_array_equal(
    jax.device_get(result.foo.coords["time_extra"].data),
    jax.device_get(dataset_inputs.coords["time_extra"].data),
  )


def test_scan_no_inputs():
  """`scan` without inputs runs `length` steps and stacks the outputs, including JAX coordinates set in the step."""

  def f(carry, x):
    assert x is None
    carry = carry + 1
    y = carry * 10
    # Set a jax_coord on the output to check it maps through to the result
    # correctly:
    y = xarray_jax.assign_jax_coords(y, extra_coord=carry[0])
    return carry, y

  init = xarray_jax.DataArray(data=np.zeros(2), dims=("x",))
  carry, result = xarray_jax.scan(f, init=init, length=5, dim="time")
  np.testing.assert_array_equal(jax.device_get(carry), [5, 5])
  assert result.sizes == {"x": 2, "time": 5}
  np.testing.assert_array_equal(jax.device_get(result.extra_coord.data), [1, 2, 3, 4, 5])


def test_assign_coords_arg_types():
  """JAX coordinates can be given in every shorthand xarray accepts (arrays, lists, scalars, Variables, DataArrays,
  tuples)."""
  # Check we can assign coords with a variety of shorthands mirroring those
  # supported by xarray's own APIs:
  result = xarray_jax.assign_coords(
    xarray_jax.Dataset(),
    jax_coords={
      "a": np.arange(2),
      "b": jnp.arange(2),
      "c": [0, 1],
      "d": 123,
      "e": xarray_jax.Variable(("e2",), np.arange(2)),
      "f": xarray_jax.DataArray(data=np.arange(2), dims=("f2",)),
      "g": (("g2",), np.arange(2)),
    },
  )
  xarray.testing.assert_equal(result.a.variable, xarray.Variable(("a",), np.arange(2)))
  xarray.testing.assert_equal(jax.device_get(result.b.variable), xarray.Variable(("b",), np.arange(2)))
  xarray.testing.assert_equal(result.c.variable, xarray.Variable(("c",), np.arange(2)))
  xarray.testing.assert_equal(result.d.variable, xarray.Variable((), 123))
  xarray.testing.assert_equal(result.e.variable, xarray.Variable(("e2",), np.arange(2)))
  xarray.testing.assert_equal(result.f.variable, xarray.Variable(("f2",), np.arange(2)))
  xarray.testing.assert_equal(result.g.variable, xarray.Variable(("g2",), np.arange(2)))
