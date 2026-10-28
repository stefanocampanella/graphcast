import numpy as np
import xarray as xr

from graphcast import predictor_base
from graphcast.mask import Mask


class _Constant(predictor_base.Predictor):
  """Inner predictor predicting a constant, recording the mask it is given for the loss."""

  def __init__(self, value: float):
    self.value = value
    self.loss_mask = None

  def __call__(self, inputs, targets_template, forcings, **kwargs):
    return xr.full_like(targets_template, self.value)

  def loss_and_predictions(self, inputs, targets, forcings, **kwargs):
    self.loss_mask = kwargs.get("mask")
    return (xr.DataArray([0.0], dims="batch"), {}), xr.full_like(targets, self.value)


def _data():
  dims = ("time", "level", "lat")
  inputs = xr.Dataset(
    {"x": (dims, np.array([[[1.0, 2.0], [3.0, np.nan]]])), "y": (("time", "lat"), np.array([[5.0, 6.0]]))},
    coords={"time": [0]},
  )
  targets = xr.Dataset(
    {
      "x": (dims, np.zeros((1, 2, 2))),
      "y": (("time", "lat"), np.zeros((1, 2))),
      "z": (("time", "lat"), np.zeros((1, 2))),
    },
    coords={"time": [1]},
  )
  return inputs, targets


def test_predictions_keep_the_input_outside_per_variable_masks():
  inputs, targets = _data()
  masks = xr.Dataset(
    {
      "x": (("level", "lat"), np.array([[True, False], [False, True]])),
      "y": ("lat", np.array([False, True])),
      "z": ("lat", np.array([True, False])),
    }
  )
  inner = _Constant(9.0)
  predictor = Mask(inner, mask=masks, value=0.0)
  (_, _), predictions = predictor.loss_and_predictions(inputs, targets, forcings=xr.Dataset())
  np.testing.assert_array_equal(predictions["x"].values, [[[9.0, 2.0], [3.0, 9.0]]])
  np.testing.assert_array_equal(predictions["y"].values, [[5.0, 9.0]])
  # A variable that is not an input has no value to keep: NaN.
  np.testing.assert_array_equal(predictions["z"].values, [[9.0, np.nan]])
  assert predictions["x"]["time"].values.tolist() == [1]
  xr.testing.assert_identical(inner.loss_mask, masks)
  xr.testing.assert_identical(predictor(inputs, targets, forcings=xr.Dataset()), predictions)


def test_a_single_mask_for_all_variables():
  inputs, targets = _data()
  mask = xr.DataArray(np.array([True, False]), dims="lat")
  (_, _), predictions = Mask(_Constant(9.0), mask=mask).loss_and_predictions(inputs, targets, forcings=xr.Dataset())
  np.testing.assert_array_equal(predictions["y"].values, [[9.0, 6.0]])
  # Missing inputs are filled (0) before being kept.
  np.testing.assert_array_equal(predictions["x"].values, [[[9.0, 2.0], [9.0, 0.0]]])
