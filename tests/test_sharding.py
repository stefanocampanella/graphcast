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
"""Tests of the data parallel wrapper (sharding.py) on a tiny GraphCast, over 4 CPU devices (see conftest.py)."""

import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

from graphcast import predictor_base, xarray_jax
from graphcast.mask import Mask
from graphcast.sharding import FullyShardedDataParallel


@pytest.fixture(scope="module")
def mesh() -> Mesh:
  devices = jax.devices("cpu")
  assert len(devices) >= 4, "conftest.py sets 4 CPU devices"
  return Mesh(np.array(devices[:4]), ("batch",))


def _predictor(tiny_model, scan: bool, mesh: Mesh | None) -> Mask:
  """GraphCast behind loss masks, wrapped for data parallelism if a mesh is given."""
  if mesh is None:
    predictor = tiny_model.graphcast(scan)
  else:
    predictor = FullyShardedDataParallel(lambda: tiny_model.graphcast(scan), mesh)
  return Mask(predictor, mask=tiny_model.mask)


@pytest.mark.parametrize("scan", [False, True], ids=["unrolled", "scan"])
def test_data_parallel_loss_and_gradients_equal_the_plain_ones(mesh, tiny_model, scan):
  """Split over 4 devices, the predictor has the same parameters (names and shapes, so checkpoints are shared) and
  gives the same per-sample losses and the same gradients of the mean loss as without the wrapper. The previous
  version created the parameters inside shard_map (leaking tracers) and asked for an rng in `apply`."""
  inputs, targets, forcings = tiny_model.batch

  def loss_fn(wrapped: bool):
    def fn(inputs, targets, forcings):
      loss, _ = _predictor(tiny_model, scan, mesh if wrapped else None).loss(inputs, targets, forcings)
      return xarray_jax.unwrap_data(loss, require_jax=True)

    return hk.transform(fn)

  plain, wrapped = loss_fn(wrapped=False), loss_fn(wrapped=True)
  params = plain.init(jax.random.key(0), inputs, targets, forcings)
  wrapped_params = wrapped.init(jax.random.key(0), inputs, targets, forcings)
  assert jax.tree.structure(params) == jax.tree.structure(wrapped_params)
  assert jax.tree.map(np.shape, params) == jax.tree.map(np.shape, wrapped_params)

  sharded = jax.device_put((inputs, targets, forcings), NamedSharding(mesh, P("batch")))

  def mean_loss(transformed):
    return jax.jit(jax.value_and_grad(lambda p, data: transformed.apply(p, None, *data).mean()))

  loss, grads = mean_loss(plain)(params, sharded)
  wrapped_loss, wrapped_grads = mean_loss(wrapped)(params, sharded)
  per_sample = jax.jit(lambda p, data: wrapped.apply(p, None, *data))(params, sharded)

  assert per_sample.shape == (tiny_model.batch_size,)
  np.testing.assert_allclose(wrapped_loss, loss, rtol=1e-5)
  jax.tree.map(lambda a, b: np.testing.assert_allclose(a, b, rtol=1e-4, atol=1e-6), wrapped_grads, grads)


def test_data_parallel_predictions_equal_the_plain_ones(mesh, tiny_model):
  """Predictions computed shard by shard equal those of the whole batch at once."""
  inputs, targets, forcings = tiny_model.batch

  def predict(wrapped: bool):
    return hk.transform(lambda i, t, f: _predictor(tiny_model, False, mesh if wrapped else None)(i, t, f))

  plain, wrapped = predict(False), predict(True)
  params = plain.init(jax.random.key(0), inputs, targets, forcings)
  expected = plain.apply(params, None, inputs, targets, forcings)
  actual = jax.jit(lambda p, i, t, f: wrapped.apply(p, None, i, t, f))(params, inputs, targets, forcings)
  np.testing.assert_allclose(actual.zos.data, expected.zos.data, rtol=1e-5, atol=1e-6)


class _NoisyPredictor(predictor_base.Predictor):
  """Adds noise from the rng to the last input: a stand-in for a probabilistic predictor."""

  def __call__(self, inputs, targets_template, forcings, **kwargs):
    last = xarray_jax.unwrap_data(inputs["zos"].isel(time=-1), require_jax=False)
    scale = hk.get_parameter("scale", (), init=jnp.ones)
    noisy = last + scale * jax.random.normal(hk.next_rng_key(), last.shape)
    return targets_template.assign(zos=targets_template["zos"].copy(data=noisy[:, None]))

  def loss(self, inputs, targets, forcings, **kwargs):
    raise NotImplementedError

  def loss_and_predictions(self, inputs, targets, forcings, **kwargs):
    raise NotImplementedError


def test_data_parallel_predictor_gets_a_key_per_device(mesh, tiny_model):
  """With an rng, each device runs the predictor with the key folded with its index along the batch axis (the key is
  an argument of shard_map: see training_utils.data_parallel)."""
  inputs, targets, forcings = tiny_model.batch
  wrapped = hk.transform(lambda i, t, f: FullyShardedDataParallel(_NoisyPredictor, mesh)(i, t, f))
  params = wrapped.init(jax.random.key(0), inputs, targets, forcings)
  key = jax.random.key(1)

  actual = jax.jit(lambda p, k, i, t, f: wrapped.apply(p, k, i, t, f))(params, key, inputs, targets, forcings)

  plain = hk.transform(lambda i, t, f: _NoisyPredictor()(i, t, f))
  # The wrapper's key: the first one drawn in the enclosing transform from the key given to apply.
  step_key = hk.transform(hk.next_rng_key).apply({}, key)
  shard = tiny_model.batch_size // 4
  for device in range(4):
    rows = slice(device * shard, (device + 1) * shard)
    expected = plain.apply(
      params,
      jax.random.fold_in(step_key, device),
      inputs.isel(batch=rows),
      targets.isel(batch=rows),
      forcings.isel(batch=rows),
    )
    np.testing.assert_allclose(actual.zos.data[rows], expected.zos.data, rtol=1e-6, atol=1e-6)
