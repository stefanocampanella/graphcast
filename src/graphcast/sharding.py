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

from typing import Any

import haiku as hk
import xarray
from jax.experimental.shard_map import shard_map
from jax.sharding import Mesh
from jax.sharding import PartitionSpec as P

from graphcast import predictor_base

PyTree = Any


class FullyShardedDataParallel(predictor_base.Predictor):
  """Wrapper that replicate parameters to devices."""

  def __init__(
    self, predictor: predictor_base.Predictor, mesh: Mesh, batch_dim_name: str = "batch"
  ):
    """Inits the wrapper.

    Args:
      mesh: predictor being wrapped.
      batch_dim_name: disables the wrapper if False, for simpler hyperparameter scans.

    """
    self._predictor = predictor
    self._abstract_mesh = mesh.abstract_mesh
    self._spec = P(batch_dim_name)

  def __call__(
    self,
    inputs: xarray.Dataset,
    targets_template: xarray.Dataset,
    forcings: xarray.Dataset,
    **kwargs,
  ) -> xarray.Dataset:
    @hk.without_apply_rng
    @hk.transform
    def _local_call(inputs, targets_template, forcings):
      return self._predictor(
        inputs=inputs, targets_template=targets_template, forcings=forcings, **kwargs
      )

    _global_call_init = shard_map(
      hk.transparent_lift(_local_call.init),
      mesh=self._abstract_mesh,
      in_specs=(None, self._spec, self._spec, self._spec),
      out_specs=P(),
      check_rep=False,
    )
    params = _global_call_init(hk.next_rng_key(), inputs, targets_template, forcings)

    _global_call_apply = shard_map(
      hk.transparent_lift(_local_call.apply),
      mesh=self._abstract_mesh,
      in_specs=(None, self._spec, self._spec, self._spec),
      out_specs=self._spec,
      check_rep=False,
    )
    predictions = _global_call_apply(params, inputs, targets_template, forcings)

    return predictions

  def loss(
    self,
    inputs: xarray.Dataset,
    targets: xarray.Dataset,
    forcings: xarray.Dataset,
    **kwargs,
  ) -> predictor_base.LossAndDiagnostics:
    @hk.without_apply_rng
    @hk.transform
    def _local_loss(inputs, targets, forcings):
      return self._predictor.loss(inputs=inputs, targets=targets, forcings=forcings, **kwargs)

    _global_loss_init = shard_map(
      hk.transparent_lift(_local_loss.init),
      mesh=self._abstract_mesh,
      in_specs=(None, self._spec, self._spec, self._spec),
      out_specs=P(),
      check_rep=False,
    )
    params = _global_loss_init(hk.next_rng_key(), inputs, targets, forcings)

    _global_loss_apply = shard_map(
      hk.transparent_lift(_local_loss.apply),
      mesh=self._abstract_mesh,
      in_specs=(None, self._spec, self._spec, self._spec),
      out_specs=self._spec,
      check_rep=False,
    )
    loss_and_diagnostics = _global_loss_apply(params, inputs, targets, forcings)

    return loss_and_diagnostics

  def loss_and_predictions(  # pytype: disable=signature-mismatch  # jax-ndarray
    self,
    inputs: xarray.Dataset,
    targets: xarray.Dataset,
    forcings: xarray.Dataset,
    **kwargs,
  ) -> tuple[predictor_base.LossAndDiagnostics, xarray.Dataset]:
    @hk.without_apply_rng
    @hk.transform
    def _local_loss_and_predictions(inputs, targets, forcings):
      return self._predictor.loss_and_predictions(
        inputs=inputs, targets=targets, forcings=forcings, **kwargs
      )

    _global_loss_and_predictions_init = shard_map(
      hk.transparent_lift(_local_loss_and_predictions.init),
      mesh=self._abstract_mesh,
      in_specs=(None, self._spec, self._spec, self._spec),
      out_specs=P(),
      check_rep=False,
    )
    params = _global_loss_and_predictions_init(hk.next_rng_key(), inputs, targets, forcings)

    _global_loss_and_predictions_apply = shard_map(
      hk.transparent_lift(_local_loss_and_predictions.apply),
      mesh=self._abstract_mesh,
      in_specs=(None, self._spec, self._spec, self._spec),
      out_specs=self._spec,
      check_rep=False,
    )
    loss_and_diagnostics, predictions = _global_loss_and_predictions_apply(
      params, inputs, targets, forcings
    )

    return loss_and_diagnostics, predictions
