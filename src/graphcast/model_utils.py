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
"""Utilities for building models. For learnable Fourier features, see:
https://arxiv.org/pdf/2106.02795 and https://arxiv.org/abs/2006.10739."""

import dataclasses
from collections.abc import Callable, Mapping
from typing import Any

import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
import pyproj
import xarray
from jax.ad_checkpoint import checkpoint_name

from graphcast.gis_utils import cartesian_unit_sphere_crs, equirectangular_crs, get_transform

NumpyInterface = Any
TransformInterface = Any


def get_graph_spatial_features(
  *,
  node_lon: np.ndarray,
  node_lat: np.ndarray,
  senders: np.ndarray,
  receivers: np.ndarray,
  add_node_position: bool,
  add_node_coordinates: bool,
  add_edge_fwd_azimuth: bool,
  add_edge_direction: bool,
  add_edge_length: bool,
  add_edge_receiver_coordinates: bool,
  edge_normalization: tuple[float, float] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
  """Computes spatial features for the nodes.

  Args:
    node_lon: Longitudes in the [-180, 180] interval of shape [num_nodes]
    node_lat: Latitudes in the [-90, 90] interval of shape [num_nodes]
    senders: Sender indices of shape [num_edges]
    receivers: Receiver indices of shape [num_edges]
    add_node_position: Add unit norm absolute positions to node features.
    add_node_coordinates: Add nodes longitude and latitude in turns (1 / 2pi radians) to node features.
    add_edge_fwd_azimuth: Add edge forward azimuth in turns (1 / 2pi radians) to edge features.
    add_edge_direction: Add edge unit vector on the tangent plane to edge features
        ((x, y) with respectively x easting and y northing directions).
    add_edge_receiver_coordinates: Add receiver longitude and latitude in turns (1 / 2pi radians) to edge features.
    add_edge_length: Add geodetic length to edge features.
    edge_normalization: Allows explicitly controlling edge normalization.
        If None, defaults to max edge length, otherwise specify location and scale as a pair.
        This supports using pre-trained model weights with a different graph structure to what it was trained.

  Returns:
    Arrays of shape: [num_nodes, num_features] and [num_edges, num_features].
    with node and edge features.

  """

  num_nodes = node_lat.shape[0]
  num_edges = senders.shape[0]
  dtype = node_lat.dtype

  # Computing some node features.
  node_features = []

  if add_node_position:
    # Already in [-1, 1.] range.
    latlon_to_unit_sphere = get_transform(equirectangular_crs, cartesian_unit_sphere_crs)
    node_features.extend(*latlon_to_unit_sphere((node_lon, node_lat)))

  if add_node_coordinates:
    node_phi, node_theta = lat_lon_deg_to_spherical(node_lon, node_lat)
    # Normalize in [-1, 1.] range.
    node_features.append(node_phi / (2 * np.pi))
    node_features.append(node_theta / (2 * np.pi))

  if not node_features:
    node_features = np.zeros([num_nodes, 0], dtype=dtype)
  else:
    node_features = np.stack(node_features, axis=-1)

  # Computing some edge features.
  edge_features = []

  geoid = pyproj.Geod(ellps="WGS84")
  edge_azimuths, _, edge_lengths = geoid.inv(
    node_lon[senders], node_lat[senders], node_lon[receivers], node_lat[receivers]
  )
  edge_azimuths = np.rad2deg(edge_azimuths)
  if edge_normalization is None:
    # Normalize to the maximum edge length.
    edge_normalization_location = np.zeros((num_edges,), dtype=dtype)
    edge_normalization_scale = edge_lengths.max()
  else:
    edge_normalization_location, edge_normalization_scale = edge_normalization
  edge_lengths = (edge_lengths - edge_normalization_location) / edge_normalization_scale

  if add_edge_fwd_azimuth:
    edge_features.append(edge_azimuths / (2 * np.pi))
  if add_edge_direction:
    edge_features.append(np.sin(edge_azimuths))
    edge_features.append(np.cos(edge_azimuths))
  if add_edge_length:
    edge_features.append(edge_lengths)

  # TODO: this should probably be deprecated
  if add_edge_receiver_coordinates:
    edge_phi, edge_theta = lat_lon_deg_to_spherical(node_lon[receivers], node_lat[receivers])
    edge_features.append(edge_phi / (2 * np.pi))
    edge_features.append(edge_theta / (2 * np.pi))

  if not edge_features:
    edge_features = np.zeros([num_edges, 0], dtype=dtype)
  else:
    edge_features = np.stack(edge_features, axis=-1)

  return node_features, edge_features


def lat_lon_to_leading_axes(grid_xarray: xarray.DataArray) -> xarray.DataArray:
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


def lat_lon_deg_to_spherical(
  node_lon: np.ndarray,
  node_lat: np.ndarray,
  np_: NumpyInterface = np,
) -> tuple[np.ndarray, np.ndarray]:
  phi = np_.deg2rad(node_lon)
  theta = np_.deg2rad(90 - node_lat)
  return phi, theta


def get_bipartite_graph_spatial_features(
  *,
  senders_node_lon: np.ndarray,
  senders_node_lat: np.ndarray,
  senders: np.ndarray,
  receivers_node_lon: np.ndarray,
  receivers_node_lat: np.ndarray,
  receivers: np.ndarray,
  add_node_position: bool,
  add_node_coordinates: bool,
  add_edge_fwd_azimuth: bool,
  add_edge_direction: bool,
  add_edge_length: bool,
  add_edge_receiver_coordinates: bool,
  edge_normalization: tuple[float, float] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
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
    add_node_position: Add unit norm absolute positions to node features.
    add_node_coordinates: Add nodes longitude and latitude in turns (1 / 2pi radians) to node features.
    add_edge_fwd_azimuth: Add edge forward azimuth in turns (1 / 2pi radians) to edge features.
    add_edge_direction: Add edge unit vector on the tangent plane to edge features
        ((x, y) with respectively x easting and y northing directions).
    add_edge_receiver_coordinates: Add receiver longitude and latitude in turns (1 / 2pi radians) to edge features.
    add_edge_length: Add geodetic length to edge features.
    edge_normalization: Allows explicitly controlling edge normalization.
        If None, defaults to max edge length, otherwise specify location and scale as a pair.
        This supports using pre-trained model weights with a different graph structure to what it was trained.

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
    senders_node_lat, senders_node_lon
  )
  receivers_node_phi, receivers_node_theta = lat_lon_deg_to_spherical(
    receivers_node_lat, receivers_node_lon
  )

  # Computing some node features.
  senders_node_features = []
  receivers_node_features = []

  if add_node_position:
    # Already in [-1, 1.] range.
    latlon_to_unit_sphere = get_transform(equirectangular_crs, cartesian_unit_sphere_crs)
    senders_node_features.extend(latlon_to_unit_sphere((senders_node_lon, senders_node_lat)))
    receivers_node_features.extend(latlon_to_unit_sphere((receivers_node_lon, receivers_node_lat)))

  if add_node_coordinates:
    # Normalize in [-1, 1.] range.
    senders_node_features.append(senders_node_phi / (2 * np.pi))
    senders_node_features.append(senders_node_theta / (2 * np.pi))
    receivers_node_features.append(receivers_node_phi / (2 * np.pi))
    receivers_node_features.append(receivers_node_theta / (2 * np.pi))

  if not senders_node_features:
    senders_node_features = np.zeros([num_senders, 0], dtype=dtype)
    receivers_node_features = np.zeros([num_receivers, 0], dtype=dtype)
  else:
    senders_node_features = np.stack(senders_node_features, axis=-1)
    receivers_node_features = np.stack(receivers_node_features, axis=-1)

  # Computing some edge features.
  edge_features = []

  geoid = pyproj.Geod(ellps="WGS84")
  edge_azimuths, _, edge_lengths = geoid.inv(
    senders_node_lon[senders],
    senders_node_lat[senders],
    receivers_node_lon[receivers],
    receivers_node_lat[receivers],
  )
  edge_azimuths = np.rad2deg(edge_azimuths)
  if edge_normalization is None:
    # Normalize to the maximum edge length.
    edge_normalization_location = np.zeros((num_edges,), dtype=dtype)
    edge_normalization_scale = edge_lengths.max()
  else:
    edge_normalization_location, edge_normalization_scale = edge_normalization
  edge_lengths = (edge_lengths - edge_normalization_location) / edge_normalization_scale
  if add_edge_fwd_azimuth:
    edge_features.append(edge_azimuths / (2 * np.pi))
  if add_edge_direction:
    edge_features.append(np.sin(edge_azimuths))
    edge_features.append(np.cos(edge_azimuths))
  if add_edge_length:
    edge_features.append(edge_lengths)

  # TODO: this should probably be deprecated
  if add_edge_receiver_coordinates:
    edge_phi, edge_theta = lat_lon_deg_to_spherical(
      receivers_node_lon[receivers], receivers_node_lat[receivers]
    )
    edge_features.append(edge_phi / (2 * np.pi))
    edge_features.append(edge_theta / (2 * np.pi))

  if not edge_features:
    edge_features = np.zeros([num_edges, 0], dtype=dtype)
  else:
    edge_features = np.stack(edge_features, axis=-1)

  return senders_node_features, receivers_node_features, edge_features


def variable_to_stacked(
  variable: xarray.Variable,
  sizes: Mapping[str, int],
  preserved_dims: tuple[str, ...] = ("batch", "lat", "lon"),
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
  stack_to_channels_dims = [d for d in variable.dims if d not in preserved_dims]
  if stack_to_channels_dims:
    variable = variable.stack(channels=stack_to_channels_dims)
  dims = {dim: variable.sizes.get(dim) or sizes[dim] for dim in preserved_dims}
  dims["channels"] = variable.sizes.get("channels", 1)
  return variable.set_dims(dims)


def dataset_to_stacked(
  dataset: xarray.Dataset,
  sizes: Mapping[str, int] | None = None,
  preserved_dims: tuple[str, ...] = ("batch", "lat", "lon"),
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
    variable_to_stacked(dataset.variables[name], sizes or dataset.sizes, preserved_dims)
    for name in sorted(dataset.data_vars.keys())
  ]
  coords = {dim: coord for dim, coord in dataset.coords.items() if dim in preserved_dims}
  return xarray.DataArray(data=xarray.Variable.concat(data_vars, dim="channels"), coords=coords)


def stacked_to_dataset(
  stacked_array: xarray.Variable,
  template_dataset: xarray.Dataset,
  preserved_dims: tuple[str, ...] = ("batch", "lat", "lon"),
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
        f"dimensions, but found only {template_var.dims}."
      )
    unstack_from_channels_sizes[name] = {
      dim: size for dim, size in template_var.sizes.items() if dim not in preserved_dims
    }

  channels = {
    name: np.prod(list(unstack_sizes.values()), dtype=np.int64)
    for name, unstack_sizes in unstack_from_channels_sizes.items()
  }
  total_expected_channels = sum(channels.values())
  found_channels = stacked_array.sizes["channels"]
  if total_expected_channels != found_channels:
    raise ValueError(
      f"Expected {total_expected_channels} channels but found "
      f"{found_channels}, when trying to convert a stacked array of shape "
      f"{stacked_array.sizes} to a dataset of shape {template_dataset}."
    )

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


@dataclasses.dataclass
class FourierFeatures(hk.Module):
  """A simple MLP applied to Fourier features of values. see: https://arxiv.org/abs/2006.10739.

  Args:
    num_frequencies: Number of frequencies to use.
    encoding_dim: Encoding dimension, i.e., output dimension of the MLP.
    mlp_hidden_dim: Hidden dimension of the MLP.
    gamma: Scaling factor for the variance of the frequency random matrix elements, default is 1.0.
    name: Name of the module.
  """

  num_frequencies: int
  encoding_dim: int
  mlp_hidden_dim: int
  gamma: float | None = 1.0
  name: str | None = None

  def __post_init__(self):
    super().__post_init__(name=self.name)

  def __call__(self, values: jnp.ndarray) -> jnp.ndarray:
    frequencies = hk.get_parameter(
      "frequencies",
      shape=(values.shape[-1], self.num_frequencies),
      dtype=values.dtype,
      init=hk.initializers.RandomNormal(stddev=self.gamma**-2),
    )
    # The following initialization (and comment) is taken from GenCast, quote:
    #   "Scale of 2 is appropriate for input layer as sin/cos fourier features
    #   have variance 0.5 for random inputs. Also, reasonable to use for later
    #   layers as relu activation cuts variance in half for inputs to later
    #   layers and gelu something close enough too."
    input_shape = values.shape
    w_init = hk.initializers.VarianceScaling(2.0, mode="fan_in", distribution="uniform")
    mlp = hk.nets.MLP(
      output_sizes=(self.mlp_hidden_dim, self.encoding_dim),
      w_init=w_init,
      activation=jax.nn.gelu,
      activate_final=False,
    )
    # Keep only last dimensions
    values = values.reshape(-1, input_shape[-1])
    features = mlp(self.fourier_features_fn(values, frequencies))
    # Restore all but the last dimension
    features = features.reshape(input_shape[:-1] + (self.encoding_dim,))

    return features

  def fourier_features_fn(self, values: jnp.ndarray, frequencies: jnp.ndarray) -> jnp.ndarray:
    values = 2 * np.pi * values @ frequencies
    return jnp.concatenate([jnp.cos(values), jnp.sin(values)], axis=-1) / jnp.sqrt(
      self.num_frequencies
    )


# TODO: disabling learnable fourier features has not been tested, the following might be broken
def fourier_features(
  values: jnp.ndarray,
  num_frequencies: int,
) -> jnp.ndarray:
  """Maps values to sin/cos features for a range of frequencies.

  Args:
    values: Values to compute Fourier features for.
    num_frequencies: The number of frequencies to use, we will use integer from 1 up
      to num_frequencies inclusive. (We don't include a zero frequency as this would
      just give constant features which are redundant if a bias term is present).

  Returns:
    Array with same shape as values except with an extra trailing dimension
    of size 2*num_frequencies, which contains a sin and a cos feature for each
    frequency.
  """
  frequencies = 2 * jnp.pi * jnp.arange(1, num_frequencies + 1, dtype=values.dtype)
  values_times_freqs = values[..., None] * frequencies
  features = jnp.concatenate([jnp.cos(values_times_freqs), jnp.sin(values_times_freqs)], axis=-1)
  features = features.reshape(values.shape[:-1] + (2 * num_frequencies * values.shape[-1],))
  return features


@dataclasses.dataclass
class PositionalEncoder(hk.Module):
  """Encodes positions using learnable or static Fourier features."""

  learnable_fourier_features: bool
  num_frequencies: int
  encoding_dim: int | None = None
  mlp_hidden_dim: int | None = None
  gamma: float | None = 1.0
  remat: bool = False
  policy: Callable[..., bool] | None = None
  prevent_cse: bool = False
  name: str | None = None

  def __post_init__(self):
    super().__post_init__(name=self.name)

  def __call__(self, node_coordinates: jnp.ndarray) -> jnp.ndarray:
    if self.learnable_fourier_features:
      if self.encoding_dim is None or self.mlp_hidden_dim is None:
        raise ValueError(
          "When using learnable features, encoding_dim and hidden_dim must be specified."
        )
      positional_encoder = FourierFeatures(
        self.num_frequencies,
        self.encoding_dim,
        self.mlp_hidden_dim,
        self.gamma,
        name=self.name + "_fourier_features",
      )
      if self.remat:
        positional_encoder = hk.remat(
          positional_encoder, policy=self.policy, prevent_cse=self.prevent_cse
        )
      codes = positional_encoder(node_coordinates)
    else:
      codes = fourier_features(node_coordinates, self.num_frequencies)
    codes = checkpoint_name(codes, "positional_encoder")
    return codes


@dataclasses.dataclass
class Embedder(hk.Module):
  """Embeds data"""

  mlp_hidden_dim: int
  mlp_num_hidden_layers: int
  output_dim: int
  remat: bool = False
  policy: Callable[..., bool] | None = None
  prevent_cse: bool = False
  name: str | None = None

  def __post_init__(self):
    super().__post_init__(name=self.name)

  def __call__(self, input_features: jnp.ndarray) -> jnp.ndarray:
    network = hk.nets.MLP(
      [self.mlp_hidden_dim] * self.mlp_num_hidden_layers + [self.output_dim],
      name=self.name + "_mlp",
    )
    layer_norm = hk.LayerNorm(
      axis=-1, create_scale=True, create_offset=True, name=self.name + "_layer_norm"
    )
    network = hk.Sequential([network, layer_norm])
    if self.remat:
      network = hk.remat(network, policy=self.policy, prevent_cse=self.prevent_cse)
    output_features = network(input_features)
    output_features = checkpoint_name(output_features, "embedder")
    return output_features
