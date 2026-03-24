"""Wrappers for Predictors which allow working with masked data"""
from typing import Tuple

import numpy as np
import xarray as xr

from graphcast import predictor_base


class MaskedPredictor(predictor_base.Predictor):

  def __init__(self, predictor: predictor_base.Predictor, mask: xr.DataArray, value: np.floating | float = np.float32(0.0)):
       self._predictor = predictor
       self._value = value
       self._mask = mask

  def __call__(self,
               inputs: xr.Dataset,
               targets_template: xr.Dataset,
               forcings: xr.Dataset,
               **kwargs
               ) -> xr.Dataset:
    inputs = inputs.fillna(value=self._value)
    forcings = forcings.fillna(value=self._value)
    predictions = self._predictor(inputs=inputs, targets_template=targets_template, forcings=forcings, **kwargs)
    return predictions.where(self._mask, np.nan)

  def loss(self,
           inputs: xr.Dataset,
           targets: xr.Dataset,
           forcings: xr.Dataset,
           **kwargs
           ) -> predictor_base.LossAndDiagnostics:
    loss_and_diagnostics, _ = self.loss_and_predictions(inputs=inputs, targets=targets, forcings=forcings, **kwargs)
    return loss_and_diagnostics

  def loss_and_predictions(self,
                           inputs: xr.Dataset,
                           targets: xr.Dataset,
                           forcings: xr.Dataset,
                           **kwargs
                           ) -> Tuple[predictor_base.LossAndDiagnostics, xr.Dataset]:

    inputs = inputs.fillna(value=self._value)
    forcings = forcings.fillna(value=self._value)
    targets = targets.fillna(value=self._value)
    loss_and_diagnostics, predictions  = self._predictor.loss_and_predictions(inputs=inputs,
                                                                              targets=targets,
                                                                              forcings=forcings,
                                                                              mask=self._mask,
                                                                              **kwargs)
    return loss_and_diagnostics, predictions.where(self._mask, np.nan)

