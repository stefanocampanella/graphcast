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
"""Utilities for building models."""

from typing import Any, Mapping, Optional, Tuple, Literal

import numpy as np
import pyproj
import xarray

from graphcast.gis_utils import get_transform, equirectangular_srs, cartesian_unit_sphere_srs

NumpyInterface = Any
TransformInterface = Any


# TODO: this implementation should be updated to include the distance from the coast as a feature
# TODO: compute features by solving the related inverse problems (proj/geod)
def get_graph_spatial_features(
    *, node_lon: np.ndarray, node_lat: np.ndarray,
    senders: np.ndarray, receivers: np.ndarray,
    boundary_nodes: Optional[np.ndarray],
    add_node_positions: bool,
    add_node_latitude: bool,
    add_node_longitude: bool,
    add_edge_length: bool,
    add_edge_direction: bool,
    edge_normalization: Tuple[float, float] | Literal['zscore'] | None = None,
    sine_cosine_encoding: bool = False,
    encoding_num_freqs: int = 10,
    encoding_multiplicative_factor: float = 1.2,
    ) -> Tuple[np.ndarray, np.ndarray]:
  """Computes spatial features for the nodes.

  Args:
    node_lon: Longitudes in the [-180, 180] interval of shape [num_nodes]
    node_lat: Latitudes in the [-90, 90] interval of shape [num_nodes]
    senders: Sender indices of shape [num_edges]
    receivers: Receiver indices of shape [num_edges]
    boundary_nodes: Optional list of boundary node indices.
    add_node_positions: Add unit norm absolute positions.
    add_node_latitude: Add a feature for latitude (cos(90 - lat))
        Note even if this is set to False, the model may be able to infer the
        longitude from relative features, unless
        `relative_latitude_local_coordinates` is also True, or if there is any
        bias on the relative edge sizes for different longitudes.
    add_node_longitude: Add features for longitude (cos(lon), sin(lon)).
        Note even if this is set to False, the model may be able to infer the
        longitude from relative features, unless
        `relative_longitude_local_coordinates` is also True, or if there is any
        bias on the relative edge sizes for different longitudes.
    add_edge_length: Whether to add geodetic length of edges.
    add_edge_direction: Whether to add azimuth of edges using sine and cosine encoding.
    edge_normalization: Allows explicitly controlling edge normalization.
        If None, defaults to max edge length. If 'zscore' use standardization,
        otherwise specify location and scale. This supports using pre-trained
        model weights with a different graph structure to what it was trained.
    sine_cosine_encoding: If True, we will transform the node/edge features
        with sine and cosine functions, similar to NERF.
    encoding_num_freqs: frequency parameter
    encoding_multiplicative_factor: used for calculating the frequency.

  Returns:
    Arrays of shape: [num_nodes, num_features] and [num_edges, num_features].
    with node and edge features.

  """

  num_nodes = node_lat.shape[0]
  num_edges = senders.shape[0]
  dtype = node_lat.dtype
  node_phi, node_theta = lat_lon_deg_to_spherical(node_lon, node_lat)

  # Computing some node features.
  node_features = []

  if boundary_nodes is not None:
    # Set interior nodes to -1, boundary nodes to 1.
    boundary_mask = np.full((num_nodes,), -1.0, dtype=np.float32)
    boundary_mask[boundary_nodes] = 1.0
    node_features.append(boundary_mask)

  if add_node_positions:
    # Already in [-1, 1.] range.
    latlon_to_unit_sphere = get_transform(equirectangular_srs, cartesian_unit_sphere_srs)
    node_features.extend(*latlon_to_unit_sphere((node_lon, node_lat)))

  if add_node_latitude:
    # Using the cos of theta.
    # From 1. (north pole) to -1 (south pole).
    node_features.append(np.cos(node_theta))

  if add_node_longitude:
    # Using the cos and sin, which is already normalized.
    node_features.append(np.cos(node_phi))
    node_features.append(np.sin(node_phi))

  if not node_features:
    node_features = np.zeros([num_nodes, 0], dtype=dtype)
  else:
    node_features = np.stack(node_features, axis=-1)

  # Computing some edge features.
  edge_features = []

  if add_edge_length or add_edge_direction:
    geoid = pyproj.Geod(ellps="WGS84")
    edge_azimuths, _, edge_lengths = geoid.inv(node_lon[senders], node_lat[senders],
                                              node_lon[receivers], node_lat[receivers])
    if add_edge_length:
      if edge_normalization is None:
        # Normalize to the maximum edge length.
        edge_normalization_location = np.zeros((num_edges,), dtype=dtype)
        edge_normalization_scale = edge_lengths.max()
      elif edge_normalization == "zscore":
        edge_normalization_location = np.mean(edge_lengths)
        edge_normalization_scale = np.std(edge_lengths)
      else:
        edge_normalization_location, edge_normalization_scale = edge_normalization
      edge_lengths = (edge_lengths - edge_normalization_location) / edge_normalization_scale
      edge_features.append(edge_lengths / edge_normalization_scale)
    if add_edge_direction:
      edge_azimuths = np.deg2rad(edge_azimuths)
      edge_features.append(np.sin(edge_azimuths))
      edge_features.append(np.cos(edge_azimuths))

  if not edge_features:
    edge_features = np.zeros([num_edges, 0], dtype=dtype)
  else:
    edge_features = np.stack(edge_features, axis=-1)

  if sine_cosine_encoding:
    def sine_cosine_transform(x: np.ndarray) -> np.ndarray:
      freqs = encoding_multiplicative_factor**np.arange(encoding_num_freqs)
      phases = freqs * x[..., None]
      x_sin = np.sin(phases)
      x_cos = np.cos(phases)
      x_cat = np.concatenate([x_sin, x_cos], axis=-1)
      return x_cat.reshape([x.shape[0], -1])

    node_features = sine_cosine_transform(node_features)
    edge_features = sine_cosine_transform(edge_features)

  return node_features, edge_features


def lat_lon_to_leading_axes(
    grid_xarray: xarray.DataArray) -> xarray.DataArray:
  """Reorders xarray so lat/lon axes come first."""
  # leading + ["lat", "lon"] + trailing
  # to
  # ["lat", "lon"] + leading + trailing
  return grid_xarray.transpose("lat", "lon", ...)


def restore_leading_axes(grid_xarray: xarray.DataArray) -> xarray.DataArray:
  """Reorders xarray so batch/time/level axes come first (if present)."""

  # ["lat", "lon"] + [(batch,) (time,) (level,)] + trailing
  # to
  # [(batch,) (time,) (level,)] + ["lat", "lon"] + trailing

  input_dims = list(grid_xarray.dims)
  output_dims = list(input_dims)
  for leading_key in ["level", "time", "batch"]:  # reverse order for insert
    if leading_key in input_dims:
      output_dims.remove(leading_key)
      output_dims.insert(0, leading_key)
  return grid_xarray.transpose(*output_dims)


def lat_lon_deg_to_spherical(node_lon: np.ndarray,
                             node_lat: np.ndarray,
                             np_: NumpyInterface = np,
                            ) -> Tuple[np.ndarray, np.ndarray]:
  phi = np_.deg2rad(node_lon)
  theta = np_.deg2rad(90 - node_lat)
  return phi, theta


# TODO: this implementation could be updated to include the distance from the coast as a feature
def get_bipartite_graph_spatial_features(
    *,
    senders_node_lon: np.ndarray,
    senders_node_lat: np.ndarray,
    senders: np.ndarray,
    receivers_node_lon: np.ndarray,
    receivers_node_lat: np.ndarray,
    receivers: np.ndarray,
    senders_boundary_nodes: Optional[np.ndarray] = None,
    receivers_boundary_nodes: Optional[np.ndarray] = None,
    add_node_positions: bool,
    add_node_latitude: bool,
    add_node_longitude: bool,
    add_edge_length: bool,
    add_edge_direction: bool,
    edge_normalization: Tuple[float, float] | Literal['zscore'] | None = None,
    sine_cosine_encoding: bool = False,
    encoding_num_freqs: int = 10,
    encoding_multiplicative_factor: float = 1.2,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
  """Computes spatial features for the nodes.

  This function is almost identical to `get_graph_spatial_features`. The only
  difference is that sender nodes and receiver nodes can be in different arrays.
  This is necessary to enable combination with typed Graph.

  Args:
    senders_node_lat: Latitudes in the [-90, 90] interval of shape
      [num_sender_nodes]
    senders_node_lon: Longitudes in the [0, 360] interval of shape
      [num_sender_nodes]
    senders: Sender indices of shape [num_edges], indices in [0,
      num_sender_nodes)
    receivers_node_lat: Latitudes in the [-90, 90] interval of shape
      [num_receiver_nodes]
    receivers_node_lon: Longitudes in the [0, 360] interval of shape
      [num_receiver_nodes]
    receivers: Receiver indices of shape [num_edges], indices in [0,
      num_receiver_nodes)
    senders_boundary_nodes: Optional list of boundary node indices.
    receivers_boundary_nodes: Optional list of boundary node indices.
    add_node_positions: Add unit norm absolute positions.
    add_node_latitude: Add a feature for latitude (cos(90 - lat)) Note even if
      this is set to False, the model may be able to infer the longitude from
      relative features, unless `relative_latitude_local_coordinates` is also
      True, or if there is any bias on the relative edge sizes for different
      longitudes.
    add_node_longitude: Add features for longitude (cos(lon), sin(lon)). Note
      even if this is set to False, the model may be able to infer the longitude
      from relative features, unless `relative_longitude_local_coordinates` is
      also True, or if there is any bias on the relative edge sizes for
      different longitudes.
    add_edge_length: Whether to add geodetic length of edges.
    add_edge_direction: Whether to add azimuth of edges using sine and cosine encoding.
    edge_normalization: Allows explicitly controlling edge normalization.
        If None, defaults to max edge length. If 'zscore' use standardization,
        otherwise specify location and scale. This supports using pre-trained
        model weights with a different graph structure to what it was trained.
    sine_cosine_encoding: If True, we will transform the node/edge features
        with sine and cosine functions, similar to NERF.
    encoding_num_freqs: frequency parameter
    encoding_multiplicative_factor: used for calculating the frequency.

  Returns:
    Arrays of shape: [num_nodes, num_features] and [num_edges, num_features].
    with node and edge features.

  """
  num_senders = senders_node_lat.shape[0]
  num_receivers = receivers_node_lat.shape[0]
  num_edges = senders.shape[0]
  dtype = senders_node_lat.dtype
  assert receivers_node_lat.dtype == dtype
  senders_node_phi, senders_node_theta = lat_lon_deg_to_spherical(
      senders_node_lat, senders_node_lon)
  receivers_node_phi, receivers_node_theta = lat_lon_deg_to_spherical(
      receivers_node_lat, receivers_node_lon)

  # Computing some node features.
  senders_node_features = []
  receivers_node_features = []

  if senders_boundary_nodes is not None:
    # Set interior nodes to -1, boundary nodes to 1.
    senders_boundary_mask = np.full((num_senders,), -1.0, dtype=np.float32)
    senders_boundary_mask[senders_boundary_nodes] = 1.0
    senders_node_features.append(senders_boundary_mask)

  if receivers_boundary_nodes is not None:
    # Set interior nodes to -1, boundary nodes to 1.
    receivers_boundary_mask = np.full((num_receivers,), -1.0, dtype=np.float32)
    receivers_boundary_mask[receivers_boundary_nodes] = 1.0
    receivers_node_features.append(receivers_boundary_mask)

  if add_node_positions:
    # Already in [-1, 1.] range.
    latlon_to_unit_sphere = get_transform(equirectangular_srs, cartesian_unit_sphere_srs)
    senders_node_features.extend(
        latlon_to_unit_sphere((senders_node_lon, senders_node_lat)))
    receivers_node_features.extend(
        latlon_to_unit_sphere((receivers_node_lon, receivers_node_lat)))

  if add_node_latitude:
    # Using the cos of theta.
    # From 1. (north pole) to -1 (south pole).
    senders_node_features.append(np.cos(senders_node_theta))
    receivers_node_features.append(np.cos(receivers_node_theta))

  if add_node_longitude:
    # Using the cos and sin, which is already normalized.
    senders_node_features.append(np.cos(senders_node_phi))
    senders_node_features.append(np.sin(senders_node_phi))

    receivers_node_features.append(np.cos(receivers_node_phi))
    receivers_node_features.append(np.sin(receivers_node_phi))

  if not senders_node_features:
    senders_node_features = np.zeros([num_senders, 0], dtype=dtype)
    receivers_node_features = np.zeros([num_receivers, 0], dtype=dtype)
  else:
    senders_node_features = np.stack(senders_node_features, axis=-1)
    receivers_node_features = np.stack(receivers_node_features, axis=-1)

  # Computing some edge features.
  edge_features = []

  if add_edge_length or add_edge_direction:
    geoid = pyproj.Geod(ellps="WGS84")
    edge_azimuths, _, edge_lengths = geoid.inv(senders_node_lon[senders], senders_node_lat[senders],
                                               receivers_node_lon[receivers], receivers_node_lat[receivers])

    if add_edge_length:
      if edge_normalization is None:
        # Normalize to the maximum edge length.
        edge_normalization_location = np.zeros((num_edges,), dtype=dtype)
        edge_normalization_scale = edge_lengths.max()
      elif edge_normalization == 'zscore':
        edge_normalization_location = np.mean(edge_lengths)
        edge_normalization_scale = np.std(edge_lengths)
      else:
        edge_normalization_location, edge_normalization_scale = edge_normalization
      edge_lengths = (edge_lengths - edge_normalization_location) / edge_normalization_scale
      edge_features.append(edge_lengths)
    if add_edge_direction:
      edge_azimuths = np.deg2rad(edge_azimuths)
      edge_features.append(np.sin(edge_azimuths))
      edge_features.append(np.cos(edge_azimuths))

  if not edge_features:
    edge_features = np.zeros([num_edges, 0], dtype=dtype)
  else:
    edge_features = np.stack(edge_features, axis=-1)

  if sine_cosine_encoding:
    def sine_cosine_transform(x: np.ndarray) -> np.ndarray:
      freqs = encoding_multiplicative_factor**np.arange(encoding_num_freqs)
      phases = freqs * x[..., None]
      x_sin = np.sin(phases)
      x_cos = np.cos(phases)
      x_cat = np.concatenate([x_sin, x_cos], axis=-1)
      return x_cat.reshape([x.shape[0], -1])

    senders_node_features = sine_cosine_transform(senders_node_features)
    receivers_node_features = sine_cosine_transform(receivers_node_features)
    edge_features = sine_cosine_transform(edge_features)

  return senders_node_features, receivers_node_features, edge_features


def variable_to_stacked(
    variable: xarray.Variable,
    sizes: Mapping[str, int],
    preserved_dims: Tuple[str, ...] = ("batch", "lat", "lon"),
    ) -> xarray.Variable:
  """Converts an xarray.Variable to preserved_dims + ("channels",).

  Any dimensions other than those included in preserved_dims get stacked into a
  final "channels" dimension. If any of the preserved_dims are missing then they
  are added, with the data broadcast/tiled to match the sizes specified in
  `sizes`.

  Args:
    variable: An xarray.Variable.
    sizes: Mapping including sizes for any dimensions which are not present in
      `variable` but are needed for the output. This may be needed for example
      for a static variable with only ("lat", "lon") dims, or if you want to
      encode just the latitude coordinates (a variable with dims ("lat",)).
    preserved_dims: dimensions of variable to not be folded in channels.

  Returns:
    An xarray.Variable with dimensions preserved_dims + ("channels",).
  """
  stack_to_channels_dims = [
      d for d in variable.dims if d not in preserved_dims]
  if stack_to_channels_dims:
    variable = variable.stack(channels=stack_to_channels_dims)
  dims = {dim: variable.sizes.get(dim) or sizes[dim] for dim in preserved_dims}
  dims["channels"] = variable.sizes.get("channels", 1)
  return variable.set_dims(dims)


def dataset_to_stacked(
    dataset: xarray.Dataset,
    sizes: Optional[Mapping[str, int]] = None,
    preserved_dims: Tuple[str, ...] = ("batch", "lat", "lon"),
) -> xarray.DataArray:
  """Converts an xarray.Dataset to a single stacked array.

  This takes each constituent data_var, converts it into BHWC layout
  using `variable_to_stacked`, then concats them all along the channels axis.

  Args:
    dataset: An xarray.Dataset.
    sizes: Mapping including sizes for any dimensions which are not present in
      the `dataset` but are needed for the output. See variable_to_stacked.
    preserved_dims: dimensions from the dataset that should not be folded in
      the predictions channels.

  Returns:
    An xarray.DataArray with dimensions preserved_dims + ("channels",).
    Existing coordinates for preserved_dims axes will be preserved, however
    there will be no coordinates for "channels".
  """
  data_vars = [
      variable_to_stacked(dataset.variables[name], sizes or dataset.sizes,
                          preserved_dims)
      for name in sorted(dataset.data_vars.keys())
  ]
  coords = {
      dim: coord
      for dim, coord in dataset.coords.items()
      if dim in preserved_dims
  }
  return xarray.DataArray(
      data=xarray.Variable.concat(data_vars, dim="channels"), coords=coords)


def stacked_to_dataset(
    stacked_array: xarray.Variable,
    template_dataset: xarray.Dataset,
    preserved_dims: Tuple[str, ...] = ("batch", "lat", "lon"),
    ) -> xarray.Dataset:
  """The inverse of dataset_to_stacked.

  Requires a template dataset to demonstrate the variables/shapes/coordinates
  required.
  All variables must have preserved_dims dimensions.

  Args:
    stacked_array: Data in BHWC layout, encoded the same as dataset_to_stacked
      would if it was asked to encode `template_dataset`.
    template_dataset: A template Dataset (or other mapping of DataArrays)
      demonstrating the shape of output required (variables, shapes,
      coordinates etc).
    preserved_dims: dimensions from the target_template that were not folded in
      the predictions channels. The preserved_dims need to be a subset of the
      dims of all the variables of template_dataset.

  Returns:
    An xarray.Dataset (or other mapping of DataArrays) with the same shape and
    type as template_dataset.
  """
  unstack_from_channels_sizes = {}
  # noinspection PyTypeChecker
  var_names = sorted(template_dataset.keys())
  for name in var_names:
    template_var = template_dataset[name]
    if not all(dim in template_var.dims for dim in preserved_dims):
      raise ValueError(
          f"stacked_to_dataset requires all Variables to have {preserved_dims} "
          f"dimensions, but found only {template_var.dims}.")
    unstack_from_channels_sizes[name] = {
        dim: size for dim, size in template_var.sizes.items()
        if dim not in preserved_dims}

  channels = {name: np.prod(list(unstack_sizes.values()), dtype=np.int64)
              for name, unstack_sizes in unstack_from_channels_sizes.items()}
  total_expected_channels = sum(channels.values())
  found_channels = stacked_array.sizes["channels"]
  if total_expected_channels != found_channels:
    raise ValueError(
        f"Expected {total_expected_channels} channels but found "
        f"{found_channels}, when trying to convert a stacked array of shape "
        f"{stacked_array.sizes} to a dataset of shape {template_dataset}.")

  data_vars = {}
  index = 0
  for name in var_names:
    template_var = template_dataset[name]
    var = stacked_array.isel({"channels": slice(index, index + channels[name])})
    index += channels[name]
    var = var.unstack({"channels": unstack_from_channels_sizes[name]})
    var = var.transpose(*template_var.dims)
    data_vars[name] = xarray.DataArray(
        data=var,
        coords=template_var.coords,
        # This might not always be the same as the name it's keyed under; it
        # will refer to the original variable name, whereas the key might be
        # some alias e.g. temperature_850 under which it should be logged:
        name=template_var.name,
    )
  return type(template_dataset)(data_vars)  # pytype:disable=not-callable,wrong-arg-count


def fourier_features(
    values: np.ndarray,
    num_frequencies: int,
    base_period: float = 1.0,
    ) -> np.ndarray:
  """Maps values to sin/cos features for a range of frequencies.

  Args:
    values: Values to compute Fourier features for.
    base_period: The base period to use. This should be greater or equal to the
      range of the values, or to the period if the values have periodic
      semantics (e.g. 2pi if they represent angles). Frequencies used will be
      integer multiples of 1/base_period.
    num_frequencies: The number of frequencies to use, we will use integer
      multiples of 1/base_period from 1 up to num_frequencies inclusive. (We
      don't include a zero frequency as this would just give constant features
      which are redundant if a bias term is present).

  Returns:
    Array with same shape as values except with an extra trailing dimension
    of size 2*num_frequencies, which contains a sin and a cos feature for each
    frequency.
  """
  frequencies = np.arange(1, num_frequencies + 1) / base_period
  angular_frequencies = np.array(2 * np.pi * frequencies, dtype=values.dtype)
  values_times_angular_freqs = values[..., None] * angular_frequencies
  return np.concatenate(
      [np.cos(values_times_angular_freqs),
       np.sin(values_times_angular_freqs)],
      axis=-1)

