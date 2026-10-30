"""Tests of the normalization wrappers: InputNormalization (inputs and forcings) and OutputNormalization (residuals or
direct targets), alone and nested, with a recording inner predictor in place of the model."""

import logging

import numpy as np
import pytest
import xarray as xr

from graphcast import predictor_base
from graphcast.normalization import InputNormalization, OutputNormalization


class _Recorder(predictor_base.Predictor):
  """Inner predictor that records what it is given and predicts constant normalized values."""

  def __init__(self, value: float):
    self.value = value
    self.inputs = self.targets = self.forcings = None

  def __call__(self, inputs, targets_template, forcings, **kwargs):
    self.inputs, self.forcings = inputs, forcings
    return xr.full_like(targets_template, self.value)

  def loss_and_predictions(self, inputs, targets, forcings, **kwargs):
    self.inputs, self.targets, self.forcings = inputs, targets, forcings
    predictions = xr.full_like(targets, self.value)
    loss = ((predictions - targets) ** 2).mean()["x"]
    return (loss, {}), predictions


@pytest.fixture
def data():
  """Inputs of two timesteps, a one-step target, residual scales per latitude, and a forcing."""
  inputs = xr.Dataset({"x": (("time", "lat"), np.array([[1.0, 2.0], [3.0, 5.0]]))}, coords={"time": [0, 1]})
  targets = xr.Dataset({"x": (("time", "lat"), np.array([[4.0, 9.0]]))}, coords={"time": [2]})
  residual_scales = xr.Dataset({"x": ("lat", np.array([0.5, 2.0]))})
  forcings = xr.Dataset({"f": (("time", "lat"), np.array([[10.0, 20.0]]))}, coords={"time": [2]})
  return inputs, targets, residual_scales, forcings


class TestOutputNormalization:
  def test_residuals_normalized_by_residual_scales(self, data):
    """The model is trained on (target - last input) / residual scale; predictions are mapped back."""
    inputs, targets, residual_scales, forcings = data
    inner = _Recorder(value=1.0)
    predictor = OutputNormalization(inner, residual_scales=residual_scales)
    _, predictions = predictor.loss_and_predictions(inputs, targets, forcings=forcings)
    assert inner.targets is not None
    np.testing.assert_allclose(inner.targets["x"].values, [[(4 - 3) / 0.5, (9 - 5) / 2.0]])
    # Normalized predictions of 1 become last input + 1 * residual scale.
    np.testing.assert_allclose(predictions["x"].values, [[3 + 0.5, 5 + 2.0]])
    np.testing.assert_allclose(predictor(inputs, targets, forcings=forcings)["x"].values, [[3.5, 7.0]])
    # Inputs and forcings reach the model unchanged.
    xr.testing.assert_identical(inner.inputs, inputs)
    xr.testing.assert_identical(inner.forcings, forcings)

  def test_zero_residual_scales_keep_the_input(self, data):
    """Where the residual scale is 0 (constant series, excluded from the loss), targets are divided by 1 and
    predictions keep the last input."""
    inputs, targets, residual_scales, forcings = data
    residual_scales["x"][1] = 0.0
    inner = _Recorder(value=1.0)
    _, predictions = OutputNormalization(inner, residual_scales=residual_scales).loss_and_predictions(
      inputs, targets, forcings=forcings
    )
    assert inner.targets is not None
    np.testing.assert_allclose(inner.targets["x"].values, [[(4 - 3) / 0.5, 9 - 5]])
    np.testing.assert_allclose(predictions["x"].values, [[3 + 0.5, 5]])

  def test_targets_predicted_directly_without_residual_scales(self, data):
    """Without residual scales the model predicts the targets themselves."""
    inputs, targets, _, forcings = data
    inner = _Recorder(value=1.0)
    _, predictions = OutputNormalization(inner, residual_scales=None).loss_and_predictions(
      inputs, targets, forcings=forcings
    )
    xr.testing.assert_identical(inner.targets, targets)
    np.testing.assert_allclose(predictions["x"].values, [[1.0, 1.0]])

  def test_direct_targets_normalized_with_scales(self, data):
    """Targets that are not inputs (or all, without residual scales) use the scales and locations, if given."""
    inputs, targets, residual_scales, forcings = data
    targets["y"] = targets["x"] * 0 + 7.0
    scales = xr.Dataset({"y": 2.0, "x": 1.0})
    locations = xr.Dataset({"y": 1.0, "x": 0.0})
    inner = _Recorder(value=1.0)
    predictor = OutputNormalization(inner, residual_scales=residual_scales, scales=scales, locations=locations)
    _, predictions = predictor.loss_and_predictions(inputs, targets, forcings=forcings)
    assert inner.targets is not None
    np.testing.assert_allclose(inner.targets["y"].values, [[(7 - 1) / 2, (7 - 1) / 2]])
    np.testing.assert_allclose(predictions["y"].values, [[1 * 2 + 1, 1 * 2 + 1]])

  def test_log_messages(self, data, caplog):
    """The logged message says whether targets are predicted as residuals or directly."""
    _, _, residual_scales, _ = data
    with caplog.at_level(logging.INFO, logger="graphcast.normalization"):
      OutputNormalization(_Recorder(0.0), residual_scales=None)
      OutputNormalization(_Recorder(0.0), residual_scales=residual_scales)
    assert "predicted directly" in caplog.text
    assert "predicted as residuals" in caplog.text


class TestInputNormalization:
  def test_inputs_and_forcings_normalized(self, data):
    """Inputs and forcings are normalized for the model; targets and predictions pass unchanged."""
    inputs, targets, _, forcings = data
    scales = xr.Dataset({"x": 2.0, "f": 10.0})
    locations = xr.Dataset({"x": 1.0, "f": 0.0})
    inner = _Recorder(value=1.0)
    predictor = InputNormalization(inner, scales=scales, locations=locations)
    _, predictions = predictor.loss_and_predictions(inputs, targets, forcings=forcings)
    assert inner.inputs is not None and inner.forcings is not None
    np.testing.assert_allclose(inner.inputs["x"].values, (inputs["x"].values - 1) / 2)
    np.testing.assert_allclose(inner.forcings["f"].values, forcings["f"].values / 10)
    xr.testing.assert_identical(inner.targets, targets)
    np.testing.assert_allclose(predictions["x"].values, [[1.0, 1.0]])

  def test_skip_names(self, data):
    """Variables in skip_names (e.g. derived features) are passed unchanged."""
    inputs, targets, _, forcings = data
    inner = _Recorder(value=1.0)
    InputNormalization(inner, scales=xr.Dataset({"x": 2.0}), skip_names=["f"]).loss_and_predictions(
      inputs, targets, forcings=forcings
    )
    xr.testing.assert_identical(inner.forcings, forcings)

  def test_needs_statistics(self):
    """Without scales or locations there is nothing to normalize: the wrapper is not created."""
    with pytest.raises(ValueError, match="needs scales or locations"):
      InputNormalization(_Recorder(0.0), scales=None, locations=None)


def test_nested(data):
  """OutputNormalization outside InputNormalization: the model sees normalized inputs and normalized residuals,
  computed from the original inputs; predictions come back in the original domain."""
  inputs, targets, residual_scales, forcings = data
  inner = _Recorder(value=1.0)
  predictor = OutputNormalization(
    InputNormalization(inner, scales=xr.Dataset({"x": 10.0, "f": 1.0})), residual_scales=residual_scales
  )
  _, predictions = predictor.loss_and_predictions(inputs, targets, forcings=forcings)
  assert inner.inputs is not None and inner.targets is not None
  np.testing.assert_allclose(inner.inputs["x"].values, inputs["x"].values / 10)
  np.testing.assert_allclose(inner.targets["x"].values, [[(4 - 3) / 0.5, (9 - 5) / 2.0]])
  np.testing.assert_allclose(predictions["x"].values, [[3.5, 7.0]])
