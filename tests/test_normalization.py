import numpy as np
import pytest
import xarray as xr

from graphcast import predictor_base
from graphcast.normalization import InputsAndResiduals


class _Recorder(predictor_base.Predictor):
  """Inner predictor that records the targets it is trained on and predicts constant normalized values."""

  def __init__(self, value: float):
    self.value = value
    self.targets = None

  def __call__(self, inputs, targets_template, forcings, **kwargs):
    return xr.full_like(targets_template, self.value)

  def loss_and_predictions(self, inputs, targets, forcings, **kwargs):
    self.targets = targets
    predictions = xr.full_like(targets, self.value)
    loss = ((predictions - targets) ** 2).mean()["x"]
    return (loss, {}), predictions


@pytest.fixture
def data():
  inputs = xr.Dataset({"x": (("time", "lat"), np.array([[1.0, 2.0], [3.0, 5.0]]))}, coords={"time": [0, 1]})
  targets = xr.Dataset({"x": (("time", "lat"), np.array([[4.0, 9.0]]))}, coords={"time": [2]})
  residual_scales = xr.Dataset({"x": ("lat", np.array([0.5, 2.0]))})
  return inputs, targets, residual_scales


def test_residuals_normalized_by_residual_scales(data):
  inputs, targets, residual_scales = data
  inner = _Recorder(value=1.0)
  predictor = InputsAndResiduals(inner, stddev_by_level=None, mean_by_level=None, diffs_stddev_by_level=residual_scales)
  _, predictions = predictor.loss_and_predictions(inputs, targets, forcings=xr.Dataset())
  # The inner predictor is trained on (target - last input) / residual scale.
  np.testing.assert_allclose(inner.targets["x"].values, [[(4 - 3) / 0.5, (9 - 5) / 2.0]])
  # Its normalized predictions are mapped back: last input + prediction * residual scale.
  np.testing.assert_allclose(predictions["x"].values, [[3 + 0.5, 5 + 2.0]])
  np.testing.assert_allclose(predictor(inputs, targets, forcings=xr.Dataset())["x"].values, [[3.5, 7.0]])


def test_targets_predicted_directly_without_residual_scales(data):
  inputs, targets, _ = data
  inner = _Recorder(value=1.0)
  predictor = InputsAndResiduals(inner, stddev_by_level=None, mean_by_level=None, diffs_stddev_by_level=None)
  _, predictions = predictor.loss_and_predictions(inputs, targets, forcings=xr.Dataset())
  xr.testing.assert_identical(inner.targets, targets)
  np.testing.assert_allclose(predictions["x"].values, [[1.0, 1.0]])
  np.testing.assert_allclose(predictor(inputs, targets, forcings=xr.Dataset())["x"].values, [[1.0, 1.0]])
