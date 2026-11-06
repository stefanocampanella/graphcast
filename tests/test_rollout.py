"""Tests of chunked rollouts (rollout.py) with a predictor whose results are known exactly."""

import dask.array
import jax
import numpy as np
import pandas as pd
import pytest
import xarray as xr

from graphcast import rollout

BATCH, LON, STEPS = 2, 3, 6
DAY = np.timedelta64(1, "D")


def _predictor_fn(rng, inputs, targets_template, forcings):
  """Each step adds that step's forcing to the last x (several steps per call in a chunk)."""
  del rng
  x = inputs["x"].isel(time=-1).values
  steps = []
  for k in range(targets_template.sizes["time"]):
    x = x + forcings["f"].isel(time=k).values
    steps.append(x)
  return xr.Dataset(
    {"x": (("batch", "time", "lon"), np.stack(steps, axis=1))}, coords={"time": targets_template["time"]}
  )


def _data(with_datetime: bool = False):
  rng = np.random.default_rng(0)
  inputs = xr.Dataset(
    {
      "x": (("batch", "time", "lon"), rng.normal(size=(BATCH, 2, LON))),
      "c": (("batch", "lon"), rng.normal(size=(BATCH, LON))),
    },
    coords={"time": np.array([-1, 0]) * DAY},
  )
  time = np.arange(1, STEPS + 1) * DAY
  targets = xr.Dataset({"x": (("batch", "time", "lon"), np.zeros((BATCH, STEPS, LON)))}, coords={"time": time})
  forcings = xr.Dataset({"f": (("batch", "time", "lon"), rng.normal(size=(BATCH, STEPS, LON)))}, coords={"time": time})
  if with_datetime:
    datetime = pd.date_range("2020-01-02", periods=STEPS, freq="D").values
    targets = targets.assign_coords(datetime=("time", datetime))
    inputs = inputs.assign_coords(datetime=("time", pd.date_range("2019-12-31", periods=2, freq="D").values))
    forcings = forcings.assign_coords(datetime=("time", datetime))
  return inputs, targets, forcings


def _expected(inputs, forcings) -> np.ndarray:
  return inputs["x"].values[:, -1:, :] + np.cumsum(forcings["f"].values, axis=1)


@pytest.mark.parametrize("num_steps_per_chunk", [1, 2, 3, 6])
def test_chunked_prediction_continues_from_the_previous_chunk(num_steps_per_chunk):
  """The rollout predicts the whole template chunk by chunk, each chunk starting from the predictions of the
  previous one: the trajectory is the same whatever the chunk size, with the template's times."""
  inputs, targets, forcings = _data()
  predictions = rollout.chunked_prediction(
    _predictor_fn, jax.random.key(0), inputs, targets, forcings, num_steps_per_chunk=num_steps_per_chunk
  )
  np.testing.assert_allclose(predictions["x"].transpose("batch", "time", "lon").values, _expected(inputs, forcings))
  np.testing.assert_array_equal(predictions["time"], targets["time"])


def test_chunks_get_the_relative_times_of_the_first_chunk_and_their_own_rng():
  """Each call sees the template, forcings and inputs with the times of the first chunk (as in training) and a new
  key; the datetime coordinate of the targets is restored on the predictions."""
  inputs, targets, forcings = _data(with_datetime=True)
  calls = []

  def recording_fn(rng, inputs, targets_template, forcings):
    calls.append((rng, targets_template["time"].values, forcings["time"].values, inputs["time"].values))
    assert "datetime" not in inputs.coords and "datetime" not in forcings.coords
    return _predictor_fn(rng, inputs, targets_template, forcings)

  chunks = list(
    rollout.chunked_prediction_generator(
      recording_fn, jax.random.key(0), inputs, targets, forcings, num_steps_per_chunk=2
    )
  )
  assert len(chunks) == 3
  for _, template_time, forcing_time, input_time in calls:
    np.testing.assert_array_equal(template_time, targets["time"].values[:2])
    np.testing.assert_array_equal(forcing_time, targets["time"].values[:2])
    np.testing.assert_array_equal(input_time, inputs["time"].values)
  keys = [jax.random.key_data(rng).tolist() for rng, *_ in calls]
  assert len({str(key) for key in keys}) == 3
  for index, chunk in enumerate(chunks):
    np.testing.assert_array_equal(chunk["datetime"].values, targets["datetime"].values[2 * index : 2 * index + 2])


def test_chunks_must_divide_the_steps():
  inputs, targets, forcings = _data()
  with pytest.raises(ValueError, match="evenly divide"):
    rollout.chunked_prediction(_predictor_fn, jax.random.key(0), inputs, targets, forcings, num_steps_per_chunk=4)


def test_target_times_must_be_evenly_spaced():
  inputs, targets, forcings = _data()
  uneven = np.array([1, 2, 3, 5, 6, 7]) * DAY
  targets, forcings = targets.assign_coords(time=uneven), forcings.assign_coords(time=uneven)
  with pytest.raises(ValueError, match="evenly spaced"):
    rollout.chunked_prediction(_predictor_fn, jax.random.key(0), inputs, targets, forcings)


def test_time_dependent_inputs_must_be_predicted_or_forced():
  """An input with a time dimension that is neither predicted nor forced cannot be rolled forward."""
  inputs, targets, forcings = _data()
  inputs = inputs.assign(y=inputs["x"])
  with pytest.raises(ValueError, match="not predicted or forced"):
    rollout.chunked_prediction(_predictor_fn, jax.random.key(0), inputs, targets, forcings)


def test_extend_targets_template():
  """The template is extended lazily (dask zeros) to the required number of steps, with times and datetimes
  continuing at the same step."""
  _, targets, _ = _data(with_datetime=True)
  template = targets.isel(time=[0])

  extended = rollout.extend_targets_template(template, required_num_steps=10)

  assert extended["x"].shape == (BATCH, 10, LON)
  assert isinstance(extended["x"].data, dask.array.Array)
  np.testing.assert_array_equal(extended["time"].values, np.arange(1, 11) * DAY)
  np.testing.assert_array_equal(extended["datetime"].values, pd.date_range("2020-01-02", periods=10, freq="D").values)
