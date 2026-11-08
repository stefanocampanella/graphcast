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
"""Tests of the bfloat16 wrapper (casting.py) with a predictor that records the dtypes it sees."""

import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
import pytest
import xarray as xr

from mobydick import casting, predictor_base, xarray_jax


class _Recorder(predictor_base.Predictor):
  """Next x = w * last x + forcing, with a parameter and a state requested in bfloat16; records the dtypes it sees."""

  def __init__(self, output_dtype=None):
    self.seen = {}
    self._output_dtype = output_dtype

  def __call__(self, inputs, targets_template, forcings, **kwargs):
    self.seen.update(inputs=inputs["x"].dtype, forcings=forcings["f"].dtype, template=targets_template["x"].dtype)
    w = hk.get_parameter("w", (), dtype=jnp.bfloat16, init=jnp.ones)
    self.seen["parameter"] = w.dtype
    hk.set_state("count", jnp.ones((), jnp.bfloat16))
    value = w * xarray_jax.unwrap_data(inputs["x"].isel(time=[-1])) + xarray_jax.unwrap_data(forcings["f"])
    if self._output_dtype is not None:
      value = value.astype(self._output_dtype)
    return xarray_jax.Dataset({"x": (targets_template["x"].dims, value)}, coords=targets_template.coords)

  def loss(self, inputs, targets, forcings, **kwargs):
    loss_and_diagnostics, _ = self.loss_and_predictions(inputs, targets, forcings, **kwargs)
    return loss_and_diagnostics

  def loss_and_predictions(self, inputs, targets, forcings, **kwargs):
    predictions = self(inputs, targets, forcings)
    # With jax.numpy: xarray's mean does not take bfloat16.
    error = xarray_jax.unwrap_data(predictions["x"]) - xarray_jax.unwrap_data(targets["x"])
    loss = jnp.mean(error**2, axis=(1, 2))
    if self._output_dtype is not None:
      loss = loss.astype(self._output_dtype)
    loss = xarray_jax.DataArray(loss, dims=("batch",))
    return (loss, {"x": loss}), predictions


def _data(dtype=np.float32):
  rng = np.random.default_rng(0)
  inputs = xr.Dataset({"x": (("batch", "time", "lon"), rng.normal(size=(2, 2, 3)).astype(dtype))})
  targets = xr.Dataset({"x": (("batch", "time", "lon"), rng.normal(size=(2, 1, 3)).astype(dtype))})
  forcings = xr.Dataset({"f": (("batch", "time", "lon"), rng.normal(size=(2, 1, 3)).astype(dtype))})
  return inputs, targets, forcings


def _run(method: str, recorder: _Recorder, enabled: bool = True):
  inputs, targets, forcings = _data()
  transformed = hk.transform_with_state(
    lambda i, t, f: getattr(casting.Bfloat16Cast(recorder, enabled=enabled), method)(i, t, f)
  )
  params, state = transformed.init(jax.random.key(0), inputs, targets, forcings)
  result, state = transformed.apply(params, state, None, inputs, targets, forcings)
  return result, params, state


def test_predictions_in_bfloat16_returned_in_the_targets_dtype():
  """The wrapped predictor sees bfloat16 inputs, forcings and template, and bfloat16 parameters stored in float32
  (also its state); its bfloat16 predictions are returned in the targets' float32."""
  recorder = _Recorder()
  predictions, params, state = _run("__call__", recorder)

  assert recorder.seen == {k: jnp.bfloat16 for k in ("inputs", "forcings", "template", "parameter")}
  assert predictions["x"].dtype == np.float32
  assert jax.tree.leaves(params)[0].dtype == np.float32
  assert jax.tree.leaves(state)[0].dtype == np.float32


@pytest.mark.parametrize("method", ["loss", "loss_and_predictions"])
def test_losses_returned_in_the_targets_dtype(method):
  """Losses, diagnostics (and predictions) computed in bfloat16 are returned in float32."""
  recorder = _Recorder()
  result, _, _ = _run(method, recorder)

  assert recorder.seen["inputs"] == jnp.bfloat16
  assert {leaf.dtype for leaf in jax.tree.leaves(result)} == {np.dtype(np.float32)}


@pytest.mark.parametrize("method", ["__call__", "loss", "loss_and_predictions"])
def test_disabled(method):
  """Disabled, the wrapper passes everything through in float32."""
  recorder = _Recorder()
  result, _, _ = _run(method, recorder, enabled=False)
  assert recorder.seen["inputs"] == np.float32 and recorder.seen["parameter"] == jnp.bfloat16
  assert {leaf.dtype for leaf in jax.tree.leaves(result)} == {np.dtype(np.float32)}


@pytest.mark.parametrize("method", ["__call__", "loss", "loss_and_predictions"])
def test_outputs_must_be_bfloat16(method):
  """A wrapped predictor that does not compute in bfloat16 is reported."""
  with pytest.raises(ValueError, match="Expected bfloat16"):
    _run(method, _Recorder(output_dtype=jnp.float32))


def test_infer_floating_dtype():
  """The one floating dtype of the variables (integers ignored); several are an error."""
  assert casting.infer_floating_dtype({"a": np.zeros(2, np.float32), "i": np.zeros(2, np.int32)}) == np.float32
  with pytest.raises(ValueError, match="exactly one floating dtype"):
    casting.infer_floating_dtype({"a": np.zeros(2, np.float32), "b": np.zeros(2, np.float64)})


def test_tree_map_cast():
  """Only the leaves of the input dtype are cast."""
  tree = {"a": np.zeros(2, np.float32), "b": np.zeros(2, np.float64)}
  cast = casting.tree_map_cast(tree, input_dtype=np.dtype(np.float32), output_dtype=jnp.dtype(jnp.bfloat16))
  assert cast["a"].dtype == jnp.bfloat16 and cast["b"].dtype == np.float64
