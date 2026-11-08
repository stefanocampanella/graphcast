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
"""Tests of the Leviathan model (model.py): the checks of its inputs and mesh, scan, initialization."""

import dataclasses

import haiku as hk
import jax
import numpy as np
import pytest
import xarray as xr

from mobydick import icosahedral_mesh
from mobydick.leviathan.model import Leviathan, ModelConfig, TaskConfig, _check_rollout_inputs


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
    Leviathan(
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
    model = Leviathan(
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


def _leviathan(tiny_model, mask=None, **options):
  return Leviathan(
    _model_config=tiny_model.model_config,
    _task_config=tiny_model.task,
    _grid_lat=tiny_model.lat,
    _grid_lon=tiny_model.lon,
    _grid_mask=mask,
    _mesh_data=tiny_model.mesh_data,
    **options,
  )


@pytest.mark.parametrize("options", [{"_remat": True}, {"_edge_chunk_size": 50}], ids=["remat", "edge_chunks"])
def test_scan_and_unrolled_processor_are_interchangeable(tiny_model, options):
  """The scanned and unrolled processors (and with rematerialization or edge chunks) have the same parameters and give
  the same predictions with them, so checkpoints work with either."""
  inputs, targets, forcings = tiny_model.batch
  unrolled = hk.transform(lambda i, t, f: _leviathan(tiny_model, _scan=False)(i, t, f))
  scanned = hk.transform(lambda i, t, f: _leviathan(tiny_model, _scan=True, **options)(i, t, f))
  params = unrolled.init(jax.random.key(0), inputs, targets, forcings)
  assert jax.tree.structure(scanned.init(jax.random.key(0), inputs, targets, forcings)) == jax.tree.structure(params)

  expected = unrolled.apply(params, None, inputs, targets, forcings)
  actual = scanned.apply(params, None, inputs, targets, forcings)
  np.testing.assert_allclose(actual["zos"].data, expected["zos"].data, rtol=1e-5, atol=1e-6)


def test_connected_grid_nodes_only_drop_unconnected_points(tiny_model):
  """With connected_grid_nodes, grid points connected to no mesh vertex (here the masked ones) are not graph nodes:
  predictions at the ocean points are unchanged, and zero at the masked ones."""
  inputs, targets, forcings = tiny_model.batch
  mask = tiny_model.mask.values
  config = dataclasses.replace(tiny_model.model_config, connected_grid_nodes=True)

  def predict(connected):
    return Leviathan(
      _model_config=config if connected else tiny_model.model_config,
      _task_config=tiny_model.task,
      _grid_lat=tiny_model.lat,
      _grid_lon=tiny_model.lon,
      _grid_mask=mask,
      _mesh_data=tiny_model.mesh_data,
    )

  everything = hk.transform(lambda i, t, f: predict(False)(i, t, f))
  connected = hk.transform(lambda i, t, f: predict(True)(i, t, f))
  params = everything.init(jax.random.key(0), inputs, targets, forcings)

  expected = np.asarray(everything.apply(params, None, inputs, targets, forcings)["zos"].data)
  actual = np.asarray(connected.apply(params, None, inputs, targets, forcings)["zos"].data)
  np.testing.assert_allclose(actual[..., mask], expected[..., mask], rtol=1e-5, atol=1e-6)
  np.testing.assert_array_equal(actual[..., ~mask], 0.0)


def test_mesh_nodes_are_embedded_from_their_positions_only(tiny_model):
  """The mesh node embedder sees the 4 position features of the mesh nodes, not zeros for the grid node input channels
  (upstream GraphCast's padding, which changed only the initialization)."""
  inputs, targets, forcings = tiny_model.batch
  transformed = hk.transform(lambda i, t, f: _leviathan(tiny_model)(i, t, f))
  params = transformed.init(jax.random.key(0), inputs, targets, forcings)
  (name,) = [name for name in params if "mesh_node_embedder" in name and name.endswith("linear_0")]
  assert params[name]["w"].shape[0] == 4
