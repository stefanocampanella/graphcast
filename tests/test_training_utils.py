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
"""Tests of the data parallel map of the training loop (training_utils.data_parallel), over 4 CPU devices on a mesh with
Explicit axes as in training (see conftest.py)."""

import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import AxisType, Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

from graphcast import training_utils, xarray_jax
from graphcast.mask import Mask


@pytest.fixture(scope="module")
def mesh() -> Mesh:
  devices = jax.devices("cpu")
  assert len(devices) >= 4, "conftest.py sets 4 CPU devices"
  return Mesh(np.array(devices[:4]), ("batch",), axis_types=(AxisType.Explicit,))


@hk.transform
def _noisy_loss(data, static):
  """Per-sample loss with noise from the rng: [local batch]."""
  w = hk.get_parameter("w", (), init=jnp.ones)
  noise = jax.random.normal(hk.next_rng_key(), data.shape)
  return jnp.mean((w * data + static + noise) ** 2, axis=-1), {"w": jnp.full(data.shape[:1], w)}


@pytest.mark.parametrize("check_vma", [False, True])
def test_each_device_gets_its_own_rng_key(mesh, check_vma):
  """Each device runs the loss on its shard with the step's key folded with its index along the batch axis: the losses
  and gradients equal those computed shard by shard with these keys, and another key gives other losses. The key
  is an argument of shard_map: closing over a key computed outside fails on Explicit axes, the error that kept the
  previous helper from passing keys."""
  batch_size = 8
  data = jnp.arange(batch_size * 3, dtype=jnp.float32).reshape(batch_size, 3)
  static = jnp.float32(0.5)
  params = _noisy_loss.init(jax.random.key(0), data, static)
  key = jax.random.key(1)
  loss_fn = training_utils.data_parallel(_noisy_loss.apply, mesh, check_vma=check_vma)

  def mean_loss(params, key, data, static):
    loss, _ = loss_fn(params, key, data, static)
    return loss.mean()

  sharded = jax.device_put(data, NamedSharding(mesh, P("batch")))
  with jax.sharding.set_mesh(mesh):
    loss, grads = jax.jit(jax.value_and_grad(mean_loss))(params, key, sharded, static)
    other = jax.jit(mean_loss)(params, jax.random.key(2), sharded, static)

  shards = np.asarray(data).reshape(4, batch_size // 4, 3)

  def reference(params):
    losses = [_noisy_loss.apply(params, jax.random.fold_in(key, i), shards[i], static)[0] for i in range(4)]
    return jnp.concatenate(losses).mean()

  expected, expected_grads = jax.value_and_grad(reference)(params)
  np.testing.assert_allclose(loss, expected, rtol=1e-6)
  np.testing.assert_allclose(grads["~"]["w"], expected_grads["~"]["w"], rtol=1e-5)
  assert abs(float(other) - float(loss)) > 1e-3


@pytest.mark.parametrize("scan", [False, True], ids=["unrolled", "scan"])
def test_graphcast_loss_with_check_vma(mesh, tiny_model, scan):
  """The training loss (GraphCast behind Mask, loss masks as unmapped static data) gives the same loss and gradients
  with and without shard_map's check of varying axes. With it, the scanned processor failed: its carry starts with
  mesh edge features that are the same on every device and comes back varying along the batch axis."""
  data = tiny_model.batch
  static = xarray_jax.wrap_data(tiny_model.mask, to_jax=True, np_contiguous=False)

  @hk.transform
  def loss_fn(data, static):
    (loss, diagnostics), _ = Mask(tiny_model.graphcast(scan), mask=static).loss_and_predictions(*data)
    return xarray_jax.unwrap_data(loss, require_jax=True), xarray_jax.jax_vars(diagnostics)

  params = loss_fn.init(jax.random.key(0), data, static)
  sharded = jax.device_put(data, NamedSharding(mesh, P("batch")))
  results = []
  for check_vma in (False, True):
    parallel = training_utils.data_parallel(loss_fn.apply, mesh, check_vma=check_vma)

    def mean_loss(params, data, parallel=parallel):
      loss, _ = parallel(params, jax.random.key(1), data, static)
      return loss.mean()

    with jax.sharding.set_mesh(mesh):
      results.append(jax.jit(jax.value_and_grad(mean_loss))(params, sharded))
  (loss, grads), (vma_loss, vma_grads) = results
  np.testing.assert_allclose(vma_loss, loss, rtol=1e-6)
  jax.tree.map(lambda a, b: np.testing.assert_allclose(a, b, rtol=1e-4, atol=1e-6), vma_grads, grads)
