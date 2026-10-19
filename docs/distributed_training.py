from collections.abc import Hashable, Mapping

import chex
import haiku as hk
import jax
import jax.numpy as jnp
import xarray as xr
from jax.sharding import AxisType, NamedSharding, PartitionSpec

from graphcast import losses, model_utils, xarray_jax, xarray_tree

jax.distributed.initialize()

default_rng_key = jax.random.key(0)

target = xarray_jax.Dataset(
  data_vars={
    "ones": (("batch", "lat"), jnp.ones((16, 100))),
    "zeros": (("batch", "lat"), jnp.zeros((16, 100))),
  },
  coords={"lat": range(100)},
)
inputs = xarray_jax.Dataset(
  data_vars={
    "a": (("batch", "lat"), jax.random.uniform(default_rng_key, (16, 100))),
    "b": (("batch", "lat"), jax.random.uniform(default_rng_key, (16, 100))),
  },
  coords={"lat": range(100)},
)


def predict(inputs: xr.Dataset, targets_template: xr.Dataset) -> xr.Dataset:
  def predict_fn(xs):
    mlp = hk.nets.MLP([4, 2])
    ans = mlp(xs)
    return ans

  # Now input is a DataArray with dimensions ['batch', 'lat', 'channels'].
  inputs_stacked = model_utils.dataset_to_stacked(inputs, preserved_dims=("batch", "lat"))

  # Convert inputs_stacked to an unwrapped jax array.
  inputs_stacked_data = xarray_jax.unwrap(inputs_stacked.data, require_jax=True)

  # Add some noise to inputs, to demonstrate use of rng in apply
  key = hk.next_rng_key()
  inputs_stacked_data += jax.random.normal(key, inputs_stacked_data.shape)

  # Compute predictions
  outputs_stacked_data = predict_fn(inputs_stacked_data)

  # Wrap predictions
  outputs_stacked_data = xarray_jax.wrap(outputs_stacked_data)

  # Get back a DataArray
  outputs_stacked = xarray_jax.DataArray(
    data=outputs_stacked_data, dims=("batch", "lat", "channels")
  )

  # Unstack predictions
  outputs = model_utils.stacked_to_dataset(
    outputs_stacked.variable, targets_template, preserved_dims=("batch", "lat")
  )

  return outputs


def loss(inputs: xr.Dataset, template: xr.Dataset) -> tuple[xr.DataArray, Mapping[Hashable, xr.DataArray]]:
  # Get predictions
  predictions = predict(inputs, template)

  # Compute MSE preserving batch dimension
  squared_errors = (predictions - template) ** 2
  mse = squared_errors.mean([d for d in squared_errors.dims if d != "batch"])

  # Compute loss
  loss, diagnostics = losses.sum_per_variable_losses(mse)

  return loss, diagnostics


@hk.transform
def loss_fn(
    inputs: xr.Dataset, template: xr.Dataset
) -> tuple[chex.Array, Mapping[str, chex.Array]]:
  # Compute the losses per batch element
  loss_and_diagnostics_with_batch_dim = loss(inputs, template)

  # Compute the mean along batch, and unwrap data
  loss_and_diagnostics = xarray_tree.map_structure(
    lambda x: xarray_jax.unwrap_data(x.mean(dim="batch")), loss_and_diagnostics_with_batch_dim
  )

  return loss_and_diagnostics


params = loss_fn.init(default_rng_key, inputs, target)
local_grad_fn = jax.value_and_grad(loss_fn.apply, has_aux=True)

print(local_grad_fn(params, default_rng_key, inputs, target))

# Data parallel section

mesh = jax.make_mesh((jax.device_count(),), ("batch",), axis_types=(AxisType.Explicit,), devices=jax.devices())


@jax.jit
def grad_fn(params, rng_key, inputs, target):
  def _fold_in_axis_index(rng_key, axis_name):
    return jax.random.fold_in(rng_key, jax.lax.axis_index(axis_name))

  def _local_grad_fn(params, rng_key, inputs, target):
    local_rng_key = _fold_in_axis_index(rng_key, "batch")
    return local_grad_fn(params, local_rng_key, inputs, target)

  def _parallel_grad_fn(params, rng_key, inputs, target):
    return jax.lax.pmean(_local_grad_fn(params, rng_key, inputs, target), axis_name="batch")

  _grad_fn = jax.shard_map(
    _parallel_grad_fn,
    mesh=mesh,
    in_specs=(PartitionSpec(), None, PartitionSpec("batch"), PartitionSpec("batch")),
    out_specs=PartitionSpec(),
    check_vma=True,
  )

  return _grad_fn(params, rng_key, inputs, target)


with jax.sharding.set_mesh(mesh):
  sharding_along_batch = NamedSharding(mesh, PartitionSpec("batch"))
  sharding_replicated = NamedSharding(mesh, PartitionSpec())


  # rng_key = jax.random.key(0)

  def device_put_dataset(dataset: xr.Dataset) -> xr.Dataset:
    return jax.tree_util.tree_map(
      xarray_jax._wrapped(lambda array: jax.reshard(array, out_shardings=sharding_along_batch)),
      dataset,
    )


  def device_put_params(params):
    return jax.reshard(params, out_shardings=sharding_replicated)


  params = device_put_params(params)
  inputs = device_put_dataset(inputs)
  target = device_put_dataset(target)

  print(f"Gradient computed from {jax.process_index()} (local devices {jax.local_devices()}): ",
        grad_fn(params, default_rng_key, inputs, target))

if jax.distributed.is_initialized():
  jax.distributed.shutdown()
