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
"""Tests of the autoregressive wrapper (autoregressive.py) with a one-step predictor whose results are known exactly."""

import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
import pytest
import xarray as xr

from mobydick import autoregressive, predictor_base, xarray_jax

BATCH, LON, STEPS = 2, 3, 4
DAY = np.timedelta64(1, "D")


class _OneStep(predictor_base.Predictor):
  """Next x = w * last x + forcing f + static c, with a parameter w initialized to 1."""

  def __call__(self, inputs, targets_template, forcings, **kwargs):
    w = hk.get_parameter("w", (), init=jnp.ones)
    # Aligned by dimension names (inside the scan, forcings have time first).
    last = inputs["x"].isel(time=-1, drop=True)
    rest = (forcings["f"].isel(time=0, drop=True) + inputs["c"]).transpose(*last.dims)
    value = w * xarray_jax.unwrap_data(last) + xarray_jax.unwrap_data(rest)
    prediction = xarray_jax.DataArray(value, dims=last.dims).expand_dims(time=targets_template["time"].values)
    return xarray_jax.Dataset({"x": prediction.transpose(*targets_template["x"].dims)})

  def loss(self, inputs, targets, forcings, **kwargs):
    loss_and_diagnostics, _ = self.loss_and_predictions(inputs, targets, forcings, **kwargs)
    return loss_and_diagnostics

  def loss_and_predictions(self, inputs, targets, forcings, **kwargs):
    predictions = self(inputs, targets, forcings)
    loss = ((predictions["x"] - targets["x"]) ** 2).mean(("time", "lon"))
    return (loss, {"x": loss}), predictions


def _data(steps: int = STEPS):
  rng = np.random.default_rng(0)
  inputs = xr.Dataset(
    {
      "x": (("batch", "time", "lon"), rng.normal(size=(BATCH, 2, LON)).astype(np.float32)),
      "c": (("batch", "lon"), rng.normal(size=(BATCH, LON)).astype(np.float32)),
    },
    coords={"time": np.array([-1, 0]) * DAY},
  )
  time = np.arange(1, steps + 1) * DAY
  targets = xr.Dataset({"x": (("batch", "time", "lon"), rng.normal(size=(BATCH, steps, LON)).astype(np.float32))})
  targets = targets.assign_coords(time=time)
  forcings = xr.Dataset({"f": (("batch", "time", "lon"), rng.normal(size=(BATCH, steps, LON)).astype(np.float32))})
  return inputs, targets, forcings.assign_coords(time=time)


def _expected(inputs, forcings) -> np.ndarray:
  """The trajectory of x unrolled by hand: [batch, time, lon]."""
  x = inputs["x"].values[:, -1]
  steps = []
  for k in range(forcings.sizes["time"]):
    x = x + forcings["f"].values[:, k] + inputs["c"].values
    steps.append(x)
  return np.stack(steps, axis=1)


@pytest.mark.parametrize("gradient_checkpointing", [False, True])
def test_predictions_feed_back_as_inputs(gradient_checkpointing):
  """Each step's prediction is the next step's input (with the forcings of that step and the static inputs), for as
  many steps as the targets template has."""
  inputs, targets, forcings = _data()
  transformed = hk.transform(
    lambda i, t, f: autoregressive.Predictor(_OneStep(), gradient_checkpointing=gradient_checkpointing)(i, t, f)
  )
  params = transformed.init(jax.random.key(0), inputs, targets, forcings)
  predictions = transformed.apply(params, None, inputs, targets, forcings)

  assert predictions["x"].dims == ("time", "batch", "lon")
  np.testing.assert_allclose(
    predictions["x"].transpose("batch", "time", "lon").data, _expected(inputs, forcings), rtol=1e-5
  )
  np.testing.assert_array_equal(predictions["time"], targets["time"])


def test_loss_is_the_mean_of_the_per_step_losses():
  """The loss rolls out the predictions (not the targets) and averages the one-step losses over the steps."""
  inputs, targets, forcings = _data()
  transformed = hk.transform(lambda i, t, f: autoregressive.Predictor(_OneStep()).loss(i, t, f))
  params = transformed.init(jax.random.key(0), inputs, targets, forcings)
  loss, diagnostics = transformed.apply(params, None, inputs, targets, forcings)

  per_step = ((_expected(inputs, forcings) - targets["x"].values) ** 2).mean(axis=-1)  # [batch, time]
  np.testing.assert_allclose(loss.data, per_step.mean(axis=1), rtol=1e-5)
  np.testing.assert_allclose(diagnostics["x"].data, per_step.mean(axis=1), rtol=1e-5)


def test_one_step_loss_is_the_wrapped_loss():
  """With one target step the wrapped predictor's loss is returned as is."""
  inputs, targets, forcings = _data(steps=1)
  transformed = hk.transform(lambda i, t, f: autoregressive.Predictor(_OneStep()).loss(i, t, f))
  params = transformed.init(jax.random.key(0), inputs, targets, forcings)
  loss, _ = transformed.apply(params, None, inputs, targets, forcings)
  expected = ((_expected(inputs, forcings) - targets["x"].values) ** 2).mean(axis=(1, 2))
  np.testing.assert_allclose(loss.data, expected, rtol=1e-5)


def test_noise_is_added_to_the_inputs():
  """With noise_level, the time-dependent inputs get normal noise from the rng: other keys give other losses."""
  inputs, targets, forcings = _data()
  transformed = hk.transform(lambda i, t, f: autoregressive.Predictor(_OneStep(), noise_level=0.1).loss(i, t, f))
  params = transformed.init(jax.random.key(0), inputs, targets, forcings)
  first, _ = transformed.apply(params, jax.random.key(1), inputs, targets, forcings)
  second, _ = transformed.apply(params, jax.random.key(2), inputs, targets, forcings)
  noiseless = ((_expected(inputs, forcings) - targets["x"].values) ** 2).mean(axis=(1, 2))
  first, second = (np.asarray(xarray_jax.unwrap_data(loss)) for loss in (first, second))
  assert not np.allclose(first, second)
  assert not np.allclose(first, noiseless)


@pytest.mark.parametrize(
  ("change", "message"),
  [
    ("time_dependent_static", "must either be a forcing"),
    ("static_target", "must be time-dependent"),
    ("static_forcing", "must be time-dependent"),
    ("target_and_forcing", "both targets and forcings"),
  ],
)
def test_invalid_variables_are_rejected(change, message):
  """Time-dependent inputs must be predicted or forced, targets and forcings time-dependent and distinct."""
  inputs, targets, forcings = _data()
  if change == "time_dependent_static":
    inputs = inputs.assign(c=inputs["x"])
  elif change == "static_target":
    targets = targets.assign(s=inputs["c"])
  elif change == "static_forcing":
    forcings = forcings.assign(s=inputs["c"])
  else:
    forcings = forcings.assign(x=targets["x"])
  transformed = hk.transform(lambda i, t, f: autoregressive.Predictor(_OneStep())(i, t, f))
  with pytest.raises(ValueError, match=message):
    transformed.init(jax.random.key(0), inputs, targets, forcings)
