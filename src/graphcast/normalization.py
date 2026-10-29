# Copyright 2023 DeepMind Technologies Limited.
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
"""Wrappers for Predictors which allow them to work with normalized data.

`InputNormalization` normalizes the inputs and forcings the wrapped predictor sees; `OutputNormalization` makes it
predict normalized residuals (or targets) and translates its predictions back to the original domain. They can be
used separately or nested (`OutputNormalization` outside).
"""

import logging
from collections.abc import Iterable

import xarray

from graphcast import predictor_base, xarray_tree

logger = logging.getLogger(__name__)


def normalize[Values: (xarray.Dataset, xarray.DataArray)](
  values: Values,
  scales: xarray.Dataset | None,
  locations: xarray.Dataset | None,
  skip_names: Iterable[str] | None = None,
) -> Values:
  """Normalize variables using the given scales and (optionally) locations."""
  skip_names = skip_names or {}

  def normalize_array(array):
    if array.name is None:
      raise ValueError("Can't look up normalization constants because array has no name.")
    if locations is not None:
      if array.name in locations:
        array = array - locations[array.name].astype(array.dtype)
      elif array.name not in skip_names:
        logger.warning("No normalization location found for %s", array.name)
    if scales is not None:
      if array.name in scales:
        array = array / scales[array.name].astype(array.dtype)
      elif array.name not in skip_names:
        logger.warning("No normalization scale found for %s", array.name)
    return array

  return xarray_tree.map_structure(normalize_array, values)


def unnormalize[Values: (xarray.Dataset, xarray.DataArray)](
  values: Values,
  scales: xarray.Dataset | None,
  locations: xarray.Dataset | None,
) -> Values:
  """Unnormalize variables using the given scales and (optionally) locations."""

  def unnormalize_array(array):
    if array.name is None:
      raise ValueError("Can't look up normalization constants because array has no name.")
    if scales is not None:
      if array.name in scales:
        array = array * scales[array.name].astype(array.dtype)
      else:
        logger.warning("No normalization scale found for %s", array.name)
    if locations is not None:
      if array.name in locations:
        array = array + locations[array.name].astype(array.dtype)
      else:
        logger.warning("No normalization location found for %s", array.name)
    return array

  return xarray_tree.map_structure(unnormalize_array, values)


class InputNormalization(predictor_base.Predictor):
  """Normalizes inputs and forcings with `scales` and (optionally) `locations`; targets and predictions unchanged.

  Variables in `skip_names` (e.g. derived features already in [-1, 1]) are passed unchanged. Nest it inside
  `OutputNormalization`, which needs the original inputs to compute residuals.
  """

  def __init__(
    self,
    predictor: predictor_base.Predictor,
    scales: xarray.Dataset | None,
    locations: xarray.Dataset | None = None,
    skip_names: Iterable[str] | None = None,
  ):
    if scales is None and locations is None:
      raise ValueError("InputNormalization needs scales or locations: without them, do not wrap the predictor.")
    self._predictor = predictor
    self._scales = scales
    self._locations = locations
    self._skip_names = skip_names

  def _normalize(self, inputs: xarray.Dataset, forcings: xarray.Dataset) -> tuple[xarray.Dataset, xarray.Dataset]:
    return (
      normalize(inputs, self._scales, self._locations, skip_names=self._skip_names),
      normalize(forcings, self._scales, self._locations, skip_names=self._skip_names),
    )

  def __call__(
    self, inputs: xarray.Dataset, targets_template: xarray.Dataset, forcings: xarray.Dataset, **kwargs
  ) -> xarray.Dataset:
    norm_inputs, norm_forcings = self._normalize(inputs, forcings)
    return self._predictor(inputs=norm_inputs, targets_template=targets_template, forcings=norm_forcings, **kwargs)

  def loss(
    self, inputs: xarray.Dataset, targets: xarray.Dataset, forcings: xarray.Dataset, **kwargs
  ) -> predictor_base.LossAndDiagnostics:
    loss_and_diagnostics, _ = self.loss_and_predictions(inputs=inputs, targets=targets, forcings=forcings, **kwargs)
    return loss_and_diagnostics

  def loss_and_predictions(  # pytype: disable=signature-mismatch  # jax-ndarray
    self, inputs: xarray.Dataset, targets: xarray.Dataset, forcings: xarray.Dataset, **kwargs
  ) -> tuple[predictor_base.LossAndDiagnostics, xarray.Dataset]:
    norm_inputs, norm_forcings = self._normalize(inputs, forcings)
    return self._predictor.loss_and_predictions(inputs=norm_inputs, targets=targets, forcings=norm_forcings, **kwargs)


class OutputNormalization(predictor_base.Predictor):
  """Makes the inner predictor predict normalized residuals, or normalized targets, and maps its predictions back.

  - With `residual_scales`, a target variable that is also an input is predicted as the residual (target - last
    input) / residual scale, which roughly standardizes the targets (as GraphCast does). Residual scales are 0 where a
    series is constant (excluded from the loss): residuals are divided by 1 there, and predictions keep the input.
  - Other target variables (or all of them, without `residual_scales`) are predicted directly, normalized with
    `scales` and `locations` if given.

  The loss of the inner predictor is computed on the normalized values, so the normalization also weights the
  variables against each other. It needs the original inputs: wrap `InputNormalization`, not the other way round.
  """

  def __init__(
    self,
    predictor: predictor_base.Predictor,
    residual_scales: xarray.Dataset | None,
    scales: xarray.Dataset | None = None,
    locations: xarray.Dataset | None = None,
  ):
    self._predictor = predictor
    self._residual_scales = residual_scales
    self._scales = scales
    self._locations = locations
    if residual_scales is None:
      self._residual_divisors = None
      logger.info("No residual scales: target variables are predicted directly.")
    else:
      self._residual_divisors = xarray_tree.map_structure(lambda scale: scale.where(scale > 0, 1), residual_scales)
      logger.info("Target variables that are inputs are predicted as residuals divided by the residual scales.")

  def _is_residual(self, name, inputs: xarray.Dataset) -> bool:
    return self._residual_scales is not None and name in inputs

  def _normalize_target(self, inputs: xarray.Dataset, target: xarray.DataArray) -> xarray.DataArray:
    if target.sizes.get("time") != 1:
      raise ValueError("OutputNormalization only supports predicting a single timestep.")
    if self._is_residual(target.name, inputs):
      residual = target - inputs[target.name].isel(time=-1)
      return normalize(residual, self._residual_divisors, None)
    return normalize(target, self._scales, self._locations)

  def _unnormalize_prediction(self, inputs: xarray.Dataset, prediction: xarray.DataArray) -> xarray.DataArray:
    if prediction.sizes.get("time") != 1:
      raise ValueError("OutputNormalization only supports predicting a single timestep.")
    if self._is_residual(prediction.name, inputs):
      residual = unnormalize(prediction, self._residual_scales, None)
      return residual + inputs[prediction.name].isel(time=-1)
    return unnormalize(prediction, self._scales, self._locations)

  def __call__(
    self, inputs: xarray.Dataset, targets_template: xarray.Dataset, forcings: xarray.Dataset, **kwargs
  ) -> xarray.Dataset:
    norm_predictions = self._predictor(inputs=inputs, targets_template=targets_template, forcings=forcings, **kwargs)
    return xarray_tree.map_structure(lambda pred: self._unnormalize_prediction(inputs, pred), norm_predictions)

  def loss(
    self, inputs: xarray.Dataset, targets: xarray.Dataset, forcings: xarray.Dataset, **kwargs
  ) -> predictor_base.LossAndDiagnostics:
    loss_and_diagnostics, _ = self.loss_and_predictions(inputs=inputs, targets=targets, forcings=forcings, **kwargs)
    return loss_and_diagnostics

  def loss_and_predictions(  # pytype: disable=signature-mismatch  # jax-ndarray
    self, inputs: xarray.Dataset, targets: xarray.Dataset, forcings: xarray.Dataset, **kwargs
  ) -> tuple[predictor_base.LossAndDiagnostics, xarray.Dataset]:
    """The loss computed on normalized targets, with predictions mapped back to the original domain."""
    norm_targets = xarray_tree.map_structure(lambda target: self._normalize_target(inputs, target), targets)
    (loss, scalars), norm_predictions = self._predictor.loss_and_predictions(
      inputs=inputs, targets=norm_targets, forcings=forcings, **kwargs
    )
    predictions = xarray_tree.map_structure(lambda pred: self._unnormalize_prediction(inputs, pred), norm_predictions)
    return (loss, scalars), predictions
