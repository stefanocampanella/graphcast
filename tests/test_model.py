"""Tests of the checks of GraphCast's inputs (model.py) that do not need a mesh."""

import numpy as np
import pytest
import xarray as xr

from graphcast.model import TaskConfig, _check_rollout_inputs


def _task(input_variables: tuple[str, ...]) -> TaskConfig:
  return TaskConfig(
    input_variables=input_variables,
    target_variables=("thetao",),
    forcing_variables=("2t",),
    levels=(0,),
    input_duration="1D",
  )


def _inputs() -> xr.Dataset:
  return xr.Dataset(
    {
      "thetao": (("batch", "time", "lat", "lon"), np.zeros((1, 1, 2, 3))),
      "2t": (("batch", "time", "lat", "lon"), np.zeros((1, 1, 2, 3))),
      "deptho": (("batch", "lat", "lon"), np.zeros((1, 2, 3))),
      "swh": (("batch", "time", "lat", "lon"), np.zeros((1, 1, 2, 3))),
    }
  )


def test_predicted_forced_and_constant_inputs_can_be_rolled_out():
  """Targets, forcings and inputs without a time dimension (static fields) pass."""
  _check_rollout_inputs(_inputs(), _task(("thetao", "2t", "deptho")))


def test_time_dependent_inputs_that_are_not_rolled_out_fail():
  """A time-dependent input that is neither a target nor a forcing would be missing at the next rollout step."""
  with pytest.raises(ValueError, match="swh"):
    _check_rollout_inputs(_inputs(), _task(("thetao", "deptho", "swh")))
