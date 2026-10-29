"""Tests of the masked, latitude- and level-weighted losses: masked points do not count, per-variable masks, empty
levels, and bfloat16 inputs (as inside the model).
"""

import numpy as np
import pytest
import xarray as xr

from graphcast import losses

LAT = np.linspace(-90.0, 90.0, 5)


def _data(levels: int = 2, value: float = 1.0) -> tuple[xr.Dataset, xr.Dataset]:
  """Predictions and targets with errors equal to the level index + 1 (squared: 1, 4, ...)."""
  shape = (1, 1, levels, LAT.size, 2)
  errors = np.broadcast_to(np.arange(1, levels + 1, dtype=np.float32)[None, None, :, None, None], shape)
  coords = {"lat": LAT, "lon": [0.0, 180.0], "level": np.arange(levels), "depth": ("level", 10.0 ** np.arange(levels))}
  dims = ("batch", "time", "level", "lat", "lon")
  targets = xr.Dataset({"x": (dims, np.full(shape, value, np.float32))}, coords=coords)
  predictions = xr.Dataset({"x": (dims, targets["x"].values + errors)}, coords=coords)
  return predictions, targets


def _loss(predictions, targets, mask=None):
  loss, diagnostics = losses.weighted_mse(predictions, targets, levels_normalization_coord="depth", mask=mask)
  return float(loss.values[0]), {name: float(value.values[0]) for name, value in diagnostics.items()}


def test_a_full_mask_is_the_unmasked_mean():
  """A mask selecting all points gives the unmasked loss."""
  predictions, targets = _data()
  full = xr.DataArray(np.ones((LAT.size, 2), dtype=bool), dims=("lat", "lon"))
  assert _loss(predictions, targets, full)[0] == pytest.approx(_loss(predictions, targets)[0])


def test_masked_points_do_not_count():
  """A level with masked points keeps its per-point mean: masked points are not averaged in as zeros."""
  predictions, targets = _data()
  mask = xr.DataArray(np.ones((2, LAT.size, 2), dtype=bool), dims=("level", "lat", "lon"))
  mask[1, :, 1] = False
  # Corrupt the masked points: they must not matter.
  predictions["x"][0, 0, 1, :, 1] = 1e6
  masked, _ = _loss(predictions, targets, mask)
  unmasked, _ = _loss(*_data())
  assert masked == pytest.approx(unmasked)


def test_per_variable_masks():
  """Each variable is averaged over the points of its own mask."""
  predictions, targets = _data()
  predictions["y"], targets["y"] = predictions["x"] * 2, targets["x"]
  masks = xr.Dataset(
    {
      "x": xr.DataArray(np.ones((LAT.size, 2), dtype=bool), dims=("lat", "lon")),
      "y": xr.DataArray(np.array([[True, False]] * LAT.size), dims=("lat", "lon")),
    }
  )
  predictions["y"][..., 1] = 1e6
  _, diagnostics = _loss(predictions, targets, masks)
  _, reference = _loss(*_data())
  assert diagnostics["x"] == pytest.approx(reference["x"])
  assert np.isfinite(diagnostics["y"]) and diagnostics["y"] < 1e3


def test_a_level_without_points_contributes_zero():
  """A level without points adds 0 to the level-weighted mean."""
  predictions, targets = _data()
  mask = xr.DataArray(np.ones((2, LAT.size, 2), dtype=bool), dims=("level", "lat", "lon"))
  mask[1] = False
  only_first, _ = _loss(predictions, targets, mask)
  level_weights = losses.normalized_level_weights(targets["x"], coord="depth").values
  assert only_first == pytest.approx(level_weights[0] * 1.0 / 2)


def test_bfloat16_losses():
  """Inside the model, predictions and targets are bfloat16: the masked mean is summed in float32 and returned in
  bfloat16, as the unmasked one (Bfloat16Cast expects it)."""
  import ml_dtypes

  predictions, targets = _data()
  mask = xr.DataArray(np.ones((LAT.size, 2), dtype=bool), dims=("lat", "lon"))
  as_bf16 = [data.astype(ml_dtypes.bfloat16) for data in (predictions, targets)]
  loss, _ = losses.weighted_mse(*as_bf16, levels_normalization_coord="depth", mask=mask)
  assert loss.dtype == ml_dtypes.bfloat16
  assert float(loss.values[0]) == pytest.approx(_loss(predictions, targets)[0], rel=1e-2)
