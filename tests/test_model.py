"""Tests of the checks of GraphCast's inputs and mesh (model.py)."""

import dataclasses

import haiku as hk
import jax
import numpy as np
import pytest
import xarray as xr

from graphcast import icosahedral_mesh
from graphcast.model import GraphCast, ModelConfig, TaskConfig, _check_rollout_inputs


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


def test_a_mesh_without_mesh_size_is_rejected():
  """The encoder's query radius is a fraction of the mesh size: a mesh without one fails with a clear error (it was a
  TypeError multiplying None)."""
  mesh_data = dataclasses.replace(icosahedral_mesh.get_multi_mesh_data(1), mesh_size=None)
  model_config = ModelConfig(latent_size=4, gnn_msg_steps=1, hidden_layers=1, radius_query_fraction_edge_length=0.6)

  def build():
    GraphCast(
      _model_config=model_config,
      _task_config=_task(("thetao", "2t")),
      _grid_lat=np.array([-10.0, 10.0]),
      _grid_lon=np.array([0.0, 120.0, 240.0]),
      _grid_mask=None,
      _mesh_data=mesh_data,
    )

  with pytest.raises(ValueError, match="mesh size"):
    hk.transform(build).init(None)


def test_forcings_without_grid_dimensions(tiny_model):
  """Forcings that are not on the grid (e.g. only year progress, a function of time) are broadcast to the grid points,
  as inputs are: stacking them alone failed with a KeyError on `lat`."""
  inputs, targets, _ = tiny_model.batch
  year_progress = xr.DataArray(
    np.full((tiny_model.batch_size, 1), 0.25, dtype=np.float32), dims=("batch", "time"), coords={"time": targets.time}
  )
  forcings = xr.Dataset({"year_progress_sin": year_progress})
  task = dataclasses.replace(tiny_model.task, forcing_variables=("year_progress_sin",))

  def predict(inputs, targets, forcings):
    model = GraphCast(
      _model_config=tiny_model.model_config,
      _task_config=task,
      _grid_lat=tiny_model.lat,
      _grid_lon=tiny_model.lon,
      _grid_mask=None,
      _mesh_data=tiny_model.mesh_data,
    )
    return model(inputs, targets, forcings)

  transformed = hk.transform(predict)
  params = transformed.init(jax.random.key(0), inputs, targets, forcings)
  predictions = transformed.apply(params, None, inputs, targets, forcings)
  assert predictions["zos"].shape == targets["zos"].shape
