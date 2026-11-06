# Copyright 2026 Stefano Campanella.
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
"""Wrappers that take care of sharding."""

from collections.abc import Callable
from typing import Any

import haiku as hk
import jax
import xarray
from jax.sharding import Mesh
from jax.sharding import PartitionSpec as P

from graphcast import predictor_base


class FullyShardedDataParallel(predictor_base.Predictor):
  """Runs the wrapped predictor on shards of the batch, one per device of `mesh`.

  Inputs, targets, forcings and the results are split along their leading (batch) dimension over the `batch_dim_name`
  axis of the mesh; keyword arguments (e.g. the loss masks of `mask.Mask`) are passed whole to every device. The
  parameters are not sharded yet: each device holds all of them (data parallelism), and they have the same names as
  those of the wrapped predictor, so the two share checkpoints.

  The wrapped predictor runs inside `shard_map`, as a separate transform lifted into the enclosing one
  (`hk.transparent_lift`), so the wrapper is used inside `hk.transform` like any other predictor. Haiku modules (e.g.
  `GraphCast`) cannot be used inside a lifted transform if they were created outside it: the wrapper takes a function
  that creates the predictor, called inside.
  """

  def __init__(
    self,
    predictor_factory: Callable[[], predictor_base.Predictor],
    mesh: Mesh,
    batch_dim_name: str = "batch",
  ):
    """Inits the wrapper.

    Args:
      predictor_factory: creates the predictor being wrapped (called inside the transforms that run it).
      mesh: devices over which the batch is split.
      batch_dim_name: name of the mesh axis over which the batch is split.
    """
    self._predictor_factory = predictor_factory
    self._mesh = mesh
    self._spec = P(batch_dim_name)

  def _sharded(self, method: str, *args: xarray.Dataset, **kwargs) -> Any:
    """Calls `method` of the wrapped predictor on each shard of `args`, with the (replicated) parameters."""
    transformed = hk.without_apply_rng(
      hk.transform(lambda *method_args: getattr(self._predictor_factory(), method)(*method_args, **kwargs))
    )
    in_specs = (P(),) + (self._spec,) * len(args)
    # The same rng key on every device gives the same parameters, which are then taken from any device. Compiled:
    # outside jit, shard_map runs operation by operation (initializing a tiny GraphCast took 46 s instead of 5 s).
    init = jax.jit(jax.shard_map(transformed.init, mesh=self._mesh, in_specs=in_specs, out_specs=P(), check_vma=False))
    apply = jax.jit(
      jax.shard_map(transformed.apply, mesh=self._mesh, in_specs=in_specs, out_specs=self._spec, check_vma=False)
    )
    # Lifted outside shard_map: the parameters are created (or looked up) in the enclosing transform.
    rng = hk.next_rng_key() if hk.running_init() else None
    params = hk.transparent_lift(init)(rng, *args)
    return apply(params, *args)

  def __call__(
    self, inputs: xarray.Dataset, targets_template: xarray.Dataset, forcings: xarray.Dataset, **kwargs
  ) -> xarray.Dataset:
    return self._sharded("__call__", inputs, targets_template, forcings, **kwargs)

  def loss(
    self, inputs: xarray.Dataset, targets: xarray.Dataset, forcings: xarray.Dataset, **kwargs
  ) -> predictor_base.LossAndDiagnostics:
    return self._sharded("loss", inputs, targets, forcings, **kwargs)

  def loss_and_predictions(
    self, inputs: xarray.Dataset, targets: xarray.Dataset, forcings: xarray.Dataset, **kwargs
  ) -> tuple[predictor_base.LossAndDiagnostics, xarray.Dataset]:
    return self._sharded("loss_and_predictions", inputs, targets, forcings, **kwargs)
