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
"""Loss functions (and terms for use in loss functions) used for weather."""
import functools
from typing import Mapping, Callable, Tuple, Hashable
from typing import Optional

import jax.numpy as jnp
import numpy as np
import xarray as xr
import xarray.ufuncs as xu
from typing import Protocol

from graphcast import xarray_tree

LossAndDiagnostics = Tuple[xr.DataArray, Mapping[Hashable, xr.DataArray]]


class LossFunction(Protocol):
  """A loss function.

  This is a protocol so it's fine to use a plain function which 'quacks like'
  this. This is just to document the interface.
  """

  def __call__(self,
               predictions: xr.Dataset,
               targets: xr.Dataset,
               **optional_kwargs) -> LossAndDiagnostics:
    """Computes a loss function.

    Args:
      predictions: Dataset of predictions.
      targets: Dataset of targets.
      **optional_kwargs: Implementations may support extra optional kwargs.

    Returns:
      loss: A DataArray with dimensions ('batch',) containing losses for each
        element of the batch. These will be averaged to give the final
        loss, locally and across replicas.
      diagnostics: Mapping of additional quantities to log by name alongside the
        loss. These will will typically correspond to terms in the loss. They
        should also have dimensions ('batch',) and will be averaged over the
        batch before logging.
    """


#TODO: The loss could also make use spatial weights.
def weighted_mse(
    predictions: xr.Dataset,
    targets: xr.Dataset,
    per_variable_weights: Optional[Mapping[str, float]] = None,
    levels_normalization_coord: str = 'level',
    weights_decreasing_with_level: bool = False,
    mask: Optional[xr.DataArray] = None,
) -> LossAndDiagnostics:
  """Variable-, latitude- and level-weighted, masked MSE loss."""

  def squared_error(prediction: xr.DataArray, target: xr.DataArray) -> xr.DataArray:
    return (prediction - target)**2

  weighted_mse_fn = get_weighted_loss(squared_error,
                                      per_variable_weights=per_variable_weights,
                                      levels_normalization_coord=levels_normalization_coord,
                                      weights_decreasing_with_level=weights_decreasing_with_level,
                                      mask=mask)
  return weighted_mse_fn(predictions, targets)


def weighted_tweedie_deviance(
    predictions: xr.Dataset,
    targets: xr.Dataset,
    per_variable_weights: Optional[Mapping[str, float]] = None,
    levels_normalization_coord: str = 'level',
    weights_decreasing_with_level: bool = False,
    mask: Optional[xr.DataArray] = None,
    p: float = 0.0,
) -> LossAndDiagnostics:
  """Variable-, latitude- and level-weighted, masked Tweedie deviance loss."""

  def tweedie_deviance(prediction: xr.DataArray, target: xr.DataArray) -> xr.DataArray:

      assert prediction.shape == target.shape, 'Predictions and targets shapes must match'

      return (2 / ((1 - p) * (2 - p))) * (target ** (2 - p)
                                          - 2 * (2 - p) * target * prediction ** (1 - p)
                                          + (1 - p) * prediction ** (2 - p))
  weighted_tweedie_deviance_fn = get_weighted_loss(tweedie_deviance,
                                                   per_variable_weights=per_variable_weights,
                                                   levels_normalization_coord=levels_normalization_coord,
                                                   weights_decreasing_with_level=weights_decreasing_with_level,
                                                   mask=mask)
  return weighted_tweedie_deviance_fn(predictions, targets)


def get_weighted_loss(loss_fn: Callable[[xr.DataArray, xr.DataArray], xr.DataArray],
                      per_variable_weights: Optional[Mapping[str, float]] = None,
                      levels_normalization_coord: str = 'level',
                      weights_decreasing_with_level: bool = False,
                      mask: Optional[xr.DataArray] = None) \
    -> Callable[[xr.Dataset, xr.Dataset], LossAndDiagnostics]:
  """Returns a Dataset function that computes variable-, latitude-, and level-weighted, masked loss for a given loss
  function."""

  @functools.wraps(loss_fn)
  def weighted_loss(predictions: xr.Dataset, targets: xr.Dataset) -> LossAndDiagnostics:
    loss_per_variable_fn = get_weighted_loss_per_variable(loss_fn,
                                                          levels_normalization_coord=levels_normalization_coord,
                                                          weights_decreasing_with_level=weights_decreasing_with_level,
                                                          mask=mask)
    losses: xr.Dataset = xarray_tree.map_structure(loss_per_variable_fn, predictions, targets)
    return sum_per_variable_losses(dict(losses.data_vars), per_variable_weights)

  return weighted_loss


def get_weighted_loss_per_variable(loss_fn: Callable[[xr.DataArray, xr.DataArray], xr.DataArray],
                                   levels_normalization_coord: str = 'level',
                                   weights_decreasing_with_level: bool = False,
                                   mask: Optional[xr.DataArray] = None
                                   ) -> Callable[[xr.DataArray, xr.DataArray], xr.DataArray]:
  """Returns a DataArray function that computes (latitude) area-weighted, masked loss for a given loss function."""

  @functools.wraps(loss_fn)
  def weighted_loss_per_variable_fn(prediction: xr.DataArray, target: xr.DataArray) -> xr.DataArray:
    loss = loss_fn(prediction, target)
    loss = loss * normalized_latitude_weights(target).astype(loss.dtype)
    if 'level' in target.dims:
      loss = loss * normalized_level_weights(target, coord=levels_normalization_coord,
                                       decreasing=weights_decreasing_with_level).astype(loss.dtype)
    return _mean_preserving_batch(loss, mask=mask)

  return weighted_loss_per_variable_fn


def _mean_preserving_batch(x: xr.DataArray, mask: Optional[xr.DataArray]=None) -> xr.DataArray:
  if mask is not None:
    x = x.where(mask, 0.0)
  return x.mean([d for d in x.dims if d != 'batch'], skipna=False)


def sum_per_variable_losses(
    per_variable_losses: Mapping[Hashable, xr.DataArray],
    weights: Optional[Mapping[Hashable, np.floating | float]] = None,
) -> LossAndDiagnostics:
  """Weighted sum of per-variable losses."""
  weights = weights or {}
  if not set(weights.keys()).issubset(set(per_variable_losses.keys())):
    raise ValueError(
        'Passing a weight that does not correspond to any variable '
        f'{set(weights.keys())-set(per_variable_losses.keys())}')

  weighted_per_variable_losses = {
      name: jnp.array(weights.get(name, 1.0), dtype=loss.dtype) * loss
      for name, loss in per_variable_losses.items()
  }
  total = xr.concat(weighted_per_variable_losses.values(), dim='variable', join='exact').sum(dim='variable',
                                                                                             skipna=False)
  return total, per_variable_losses


def normalized_level_weights(data: xr.DataArray,
                             coord: str = 'level',
                             w_min=1e-2,
                             decreasing: bool = False
                             ) -> xr.DataArray:
  """Compute weights from `coord` at each level.

  We ask that the weights are such that:
    1. w_i >= w_min >= 0.
    2. sum(w_i) = 1.
    3. w_i >= w_j iff coord_i >= coord_j when decreasing=False,
       or w_i >= w_j iff coord_i <= coord_j otherwise.

  Among all the possible ways to assign such weights, we choose `w_i = a * coord_i + b` with `min(w_i) = w_min`.
  """
  weights: xr.DataArray = data.coords[coord]
  if decreasing:
    weights = -weights
  assert 0 <= w_min < 1 / len(weights), 'w_min must be in (0, 1/len(coord))'
  weights = (weights - weights.min()) / (weights.max() - weights.min())
  delta = w_min * weights.sum() / (1 - len(weights) * w_min)
  weights = delta + weights
  return weights / weights.sum()


def normalized_latitude_weights(data: xr.DataArray) -> xr.DataArray:
  """Weights based on latitude, roughly proportional to grid cell area.

  This method supports two use cases only (both for equispaced values):
  * Latitude values such that the closest value to the pole is at latitude
    (90 - d_lat/2), where d_lat is the difference between contiguous latitudes.
    For example: [-89, -87, -85, ..., 85, 87, 89]) (d_lat = 2)
    In this case each point with `lat` value represents a sphere slice between
    `lat - d_lat/2` and `lat + d_lat/2`, and the area of this slice would be
    proportional to:
    `sin(lat + d_lat/2) - sin(lat - d_lat/2) = 2 * sin(d_lat/2) * cos(lat)`, and
    we can simply omit the term `2 * sin(d_lat/2)` which is just a constant
    that cancels during normalization.
  * Latitude values that fall exactly at the poles.
    For example: [-90, -88, -86, ..., 86, 88, 90]) (d_lat = 2)
    In this case each point with `lat` value also represents
    a sphere slice between `lat - d_lat/2` and `lat + d_lat/2`,
    except for the points at the poles, that represent a slice between
    `90 - d_lat/2` and `90` or, `-90` and  `-90 + d_lat/2`.
    The areas of the first type of point are still proportional to:
    * sin(lat + d_lat/2) - sin(lat - d_lat/2) = 2 * sin(d_lat/2) * cos(lat)
    but for the points at the poles now is:
    * sin(90) - sin(90 - d_lat/2) = 2 * sin(d_lat/4) ^ 2
    and we will be using these weights, depending on whether we are looking at
    pole cells, or non-pole cells (omitting the common factor of 2 which will be
    absorbed by the normalization).

    It can be shown via a limit, or simple geometry, that in the small angles
    regime, the proportion of area per pole-point is equal to 1/8th
    the proportion of area covered by each of the nearest non-pole point, and we
    test for this in the test.

  Args:
    data: `DataArray` with latitude coordinates.
  Returns:
    Unit mean latitude weights.
  """
  latitude = data.coords['lat']

  if np.any(np.isclose(np.abs(latitude), 90.)):
    weights = _weight_for_latitude_vector_with_poles(latitude)
  else:
    weights = _weight_for_latitude_vector_without_poles(latitude)

  return weights / np.nanmean(weights)


def _weight_for_latitude_vector_without_poles(latitude: xr.DataArray) -> xr.DataArray:
  """Weights for uniform latitudes of the form [+-90-+d/2, ..., -+90+-d/2]."""
  assert latitude.dims == ('lat',), 'Latitude vector must have a single dimension.'
  delta_latitude = np.abs(_check_uniform_spacing_and_get_delta(latitude.to_numpy()))
  if (not np.isclose(np.max(latitude), 90 - delta_latitude/2) or
      not np.isclose(np.min(latitude), -90 + delta_latitude/2)):
    raise ValueError(
        f'Latitude vector {latitude} does not start/end at '
        '+- (90 - delta_latitude/2) degrees.')
  # Use XArray ufuncs only to avoid the type checker complaining about the
  # result being a numpy array.
  weights = xu.cos(xu.deg2rad(latitude))
  return weights


def _weight_for_latitude_vector_with_poles(latitude: xr.DataArray) -> xr.DataArray:
  """Weights for uniform latitudes of the form [+- 90, ..., -+90]."""
  assert latitude.dims == ('lat',), 'Latitude vector must have a single dimension.'
  delta_latitude = np.abs(_check_uniform_spacing_and_get_delta(latitude.to_numpy()))
  if (not np.isclose(np.max(latitude), 90.) or
      not np.isclose(np.min(latitude), -90.)):
    raise ValueError(
        f'Latitude vector {latitude} does not start/end at +- 90 degrees.')
  # Use XArray ufuncs only to avoid the type checker complaining about the
  # result being a numpy array.
  weights: xr.DataArray = xu.cos(xu.deg2rad(latitude)) * np.sin(np.deg2rad(delta_latitude/2))
  # The two checks above enough to guarantee that latitudes are sorted, so
  # the extremes are the poles
  weights.data[[0, -1]] = np.sin(np.deg2rad(delta_latitude/4)) ** 2
  return weights


def _check_uniform_spacing_and_get_delta(vector: np.ndarray) -> np.ndarray:
  diff = np.diff(vector)
  if not np.all(np.isclose(diff[0], diff)):
    raise ValueError(f'Vector {diff} is not uniformly spaced.')
  return diff[0]
