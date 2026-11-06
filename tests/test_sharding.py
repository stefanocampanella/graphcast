"""Tests of the data parallel wrapper (sharding.py) on a tiny GraphCast, over 4 CPU devices (see conftest.py)."""

import haiku as hk
import jax
import numpy as np
import pytest
import xarray as xr
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

from graphcast import icosahedral_mesh, xarray_jax
from graphcast.mask import Mask
from graphcast.model import GraphCast, ModelConfig, TaskConfig
from graphcast.sharding import FullyShardedDataParallel

LAT = np.arange(-75.0, 90.0, 30.0)
LON = np.arange(0.0, 360.0, 45.0)
BATCH = 8
TASK = TaskConfig(
  input_variables=("zos",), target_variables=("zos",), forcing_variables=("2t",), levels=(0,), input_duration="1D"
)
MODEL_CONFIG = ModelConfig(latent_size=4, gnn_msg_steps=1, hidden_layers=1, radius_query_fraction_edge_length=0.6)
MESH_DATA = icosahedral_mesh.get_multi_mesh_data(1)


@pytest.fixture(scope="module")
def mesh() -> Mesh:
  devices = jax.devices("cpu")
  assert len(devices) >= 4, "conftest.py sets 4 CPU devices"
  return Mesh(np.array(devices[:4]), ("batch",))


@pytest.fixture(scope="module")
def batch() -> tuple[xr.Dataset, xr.Dataset, xr.Dataset]:
  rng = np.random.default_rng(0)

  def field(day: int) -> xr.DataArray:
    return xr.DataArray(
      rng.normal(size=(BATCH, 1, LAT.size, LON.size)).astype(np.float32),
      dims=("batch", "time", "lat", "lon"),
      coords={"time": np.array([day], dtype="timedelta64[D]"), "lat": LAT, "lon": LON},
    )

  return xr.Dataset({"zos": field(0)}, {}), xr.Dataset({"zos": field(1)}), xr.Dataset({"2t": field(1)})


def _predictor(scan: bool, mesh: Mesh | None) -> Mask:
  """GraphCast on the icosahedral multi-mesh behind loss masks, wrapped for data parallelism if a mesh is given."""

  def graphcast() -> GraphCast:
    return GraphCast(
      _model_config=MODEL_CONFIG,
      _task_config=TASK,
      _grid_lat=LAT,
      _grid_lon=LON,
      _grid_mask=None,
      _mesh_data=MESH_DATA,
      _scan=scan,
    )

  predictor = graphcast() if mesh is None else FullyShardedDataParallel(graphcast, mesh)
  # The northernmost row is land.
  ocean = np.broadcast_to(LAT[:, None] < 70.0, (LAT.size, LON.size))
  mask = xr.DataArray(ocean, dims=("lat", "lon"), coords={"lat": LAT, "lon": LON})
  return Mask(predictor, mask=mask)


@pytest.mark.parametrize("scan", [False, True], ids=["unrolled", "scan"])
def test_data_parallel_loss_and_gradients_equal_the_plain_ones(mesh, batch, scan):
  """Split over 4 devices, the predictor has the same parameters (names and shapes, so checkpoints are shared) and
  gives the same per-sample losses and the same gradients of the mean loss as without the wrapper. The previous
  version created the parameters inside shard_map (leaking tracers) and asked for an rng in `apply`."""
  inputs, targets, forcings = batch

  def loss_fn(wrapped: bool):
    def fn(inputs, targets, forcings):
      loss, _ = _predictor(scan, mesh if wrapped else None).loss(inputs, targets, forcings)
      return xarray_jax.unwrap_data(loss, require_jax=True)

    return hk.transform(fn)

  plain, wrapped = loss_fn(wrapped=False), loss_fn(wrapped=True)
  params = plain.init(jax.random.key(0), inputs, targets, forcings)
  wrapped_params = wrapped.init(jax.random.key(0), inputs, targets, forcings)
  assert jax.tree.structure(params) == jax.tree.structure(wrapped_params)
  assert jax.tree.map(np.shape, params) == jax.tree.map(np.shape, wrapped_params)

  sharded = jax.device_put((inputs, targets, forcings), NamedSharding(mesh, P("batch")))

  def mean_loss(transformed):
    return jax.jit(jax.value_and_grad(lambda p, data: transformed.apply(p, None, *data).mean()))

  loss, grads = mean_loss(plain)(params, sharded)
  wrapped_loss, wrapped_grads = mean_loss(wrapped)(params, sharded)
  per_sample = jax.jit(lambda p, data: wrapped.apply(p, None, *data))(params, sharded)

  assert per_sample.shape == (BATCH,)
  np.testing.assert_allclose(wrapped_loss, loss, rtol=1e-5)
  jax.tree.map(lambda a, b: np.testing.assert_allclose(a, b, rtol=1e-4, atol=1e-6), wrapped_grads, grads)


def test_data_parallel_predictions_equal_the_plain_ones(mesh, batch):
  """Predictions computed shard by shard equal those of the whole batch at once."""
  inputs, targets, forcings = batch

  def predict(wrapped: bool):
    return hk.transform(lambda i, t, f: _predictor(False, mesh if wrapped else None)(i, t, f))

  plain, wrapped = predict(False), predict(True)
  params = plain.init(jax.random.key(0), inputs, targets, forcings)
  expected = plain.apply(params, None, inputs, targets, forcings)
  actual = jax.jit(lambda p, i, t, f: wrapped.apply(p, None, i, t, f))(params, inputs, targets, forcings)
  np.testing.assert_allclose(actual.zos.data, expected.zos.data, rtol=1e-5, atol=1e-6)
