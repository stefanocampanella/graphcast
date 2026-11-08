# SPDX-FileCopyrightText: 2025 Stefano Campanella
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
"""Wrappers for Predictors which allow working with masked data"""

import numpy as np
import xarray as xr

from mobydick import predictor_base


class Mask(predictor_base.Predictor):
  """Fills missing inputs and restricts the loss and the predictions to the points of a mask.

  `mask` is one boolean array for all predicted variables, or a dataset with one per variable (e.g. per level). The
  loss only counts the points of the mask; elsewhere, predictions keep the value of the last input (if the variable is
  an input; NaN otherwise).
  """

  def __init__(
    self,
    predictor: predictor_base.Predictor,
    mask: xr.DataArray | xr.Dataset,
    value: np.floating | float = 0.0,
  ):
    self._predictor = predictor
    self._value = np.float32(value)
    self._mask = mask

  def _variable_mask(self, name) -> xr.DataArray:
    return self._mask[name] if isinstance(self._mask, xr.Dataset) else self._mask

  def _restrict(self, predictions: xr.Dataset, inputs: xr.Dataset) -> xr.Dataset:
    restricted = {}
    for name, prediction in predictions.data_vars.items():
      if name in inputs:
        last_input = inputs[name].isel(time=-1, drop=True)
        restricted[name] = prediction.where(self._variable_mask(name), last_input)
      else:
        restricted[name] = prediction.where(self._variable_mask(name), np.nan)
    return predictions.assign(restricted)

  def __call__(self, inputs: xr.Dataset, targets_template: xr.Dataset, forcings: xr.Dataset, **kwargs) -> xr.Dataset:
    inputs = inputs.fillna(value=self._value)
    forcings = forcings.fillna(value=self._value)
    predictions = self._predictor(inputs=inputs, targets_template=targets_template, forcings=forcings, **kwargs)
    return self._restrict(predictions, inputs)

  def loss(
    self, inputs: xr.Dataset, targets: xr.Dataset, forcings: xr.Dataset, **kwargs
  ) -> predictor_base.LossAndDiagnostics:
    loss_and_diagnostics, _ = self.loss_and_predictions(inputs=inputs, targets=targets, forcings=forcings, **kwargs)
    return loss_and_diagnostics

  def loss_and_predictions(
    self, inputs: xr.Dataset, targets: xr.Dataset, forcings: xr.Dataset, **kwargs
  ) -> tuple[predictor_base.LossAndDiagnostics, xr.Dataset]:
    inputs = inputs.fillna(value=self._value)
    forcings = forcings.fillna(value=self._value)
    targets = targets.fillna(value=self._value)
    loss_and_diagnostics, predictions = self._predictor.loss_and_predictions(
      inputs=inputs, targets=targets, forcings=forcings, mask=self._mask, **kwargs
    )
    return loss_and_diagnostics, self._restrict(predictions, inputs)
