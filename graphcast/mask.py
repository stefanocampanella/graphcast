"""Wrappers for Predictors which allow working with masked data"""
from typing import Any, Tuple

from graphcast import predictor_base
import xarray as xr
import numpy as np

class MaskedPredictor(predictor_base.Predictor):

  def __init__(self, predictor: predictor_base.Predictor, mask: xr.DataArray, value: Any = np.float32(0.0)):
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
    predictions = self._predictor(inputs, targets_template, forcings=forcings, **kwargs)
    return predictions.where(self._mask, np.nan)

  def loss(self,
           inputs: xr.Dataset,
           targets: xr.Dataset,
           forcings: xr.Dataset,
           **kwargs
           ) -> predictor_base.LossAndDiagnostics:
    inputs = inputs.fillna(value=self._value)
    forcings = forcings.fillna(value=self._value)
    targets = targets.fillna(value=self._value)
    return self._predictor.loss(inputs, targets, forcings=forcings, mask=self._mask, **kwargs)

  def loss_and_predictions(self,
                           inputs: xr.Dataset,
                           targets: xr.Dataset,
                           forcings: xr.Dataset,
                           **kwargs
                           ) -> Tuple[predictor_base.LossAndDiagnostics, xr.Dataset]:

    inputs = inputs.fillna(value=self._value)
    forcings = forcings.fillna(value=self._value)
    targets = targets.fillna(value=self._value)
    return self._predictor.loss_and_predictions(inputs, targets, forcings=forcings, mask=self._mask, **kwargs)

