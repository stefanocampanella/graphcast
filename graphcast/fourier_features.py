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
"""Learnable Fourier features, see: https://arxiv.org/pdf/2106.02795 and https://arxiv.org/abs/2006.10739."""
import dataclasses
from typing import Callable

import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
from jax.ad_checkpoint import checkpoint_name


@dataclasses.dataclass
class FourierFeatures(hk.Module):
  """A simple MLP applied to Fourier features of values. see: https://arxiv.org/abs/2006.10739.

    Args:
      num_frequencies: Number of frequencies to use.
      encoding_dim: Encoding dimension, i.e., output dimension of the MLP.
      hidden_dim: Hidden dimension of the MLP.
      gamma: Scaling factor for the variance of the frequency random matrix elements, default is 1.0.
      name: Name of the module.
    """
  num_frequencies: int
  encoding_dim: int
  hidden_dim: int
  gamma: float | None = 1.0
  name: str | None = None

  def __post_init__(self):
    super().__post_init__(name=self.name)

  def __call__(self, values: jnp.ndarray) -> jnp.ndarray:
    frequencies = hk.get_parameter("frequencies", shape=(values.shape[-1], self.num_frequencies),
                                   dtype=values.dtype,
                                   init=hk.initializers.RandomNormal(stddev=self.gamma ** -2))
    # The following initialization (and comment) is taken from GenCast, quote:
    #   "Scale of 2 is appropriate for input layer as sin/cos fourier features
    #   have variance 0.5 for random inputs. Also, reasonable to use for later
    #   layers as relu activation cuts variance in half for inputs to later
    #   layers and gelu something close enough too."
    input_shape = values.shape
    w_init = hk.initializers.VarianceScaling(2.0, mode="fan_in", distribution="uniform")
    mlp = hk.nets.MLP(output_sizes=(self.hidden_dim, self.encoding_dim), w_init=w_init, activation=jax.nn.gelu,
                      activate_final=False)
    # Keep only last dimensions
    values = values.reshape(-1, input_shape[-1])
    features = mlp(self.fourier_features_fn(values, frequencies))
    # Restore all but the last dimension
    features = features.reshape(input_shape[:-1] + (self.encoding_dim,))

    return features

  def fourier_features_fn(
      self,
      values: jnp.ndarray,
      frequencies: jnp.ndarray) -> jnp.ndarray:
    values = 2 * np.pi * values @  frequencies
    return jnp.concatenate([jnp.cos(values), jnp.sin(values)], axis=-1) / jnp.sqrt(self.num_frequencies)


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
  encoding_dim: int
  hidden_dim: int
  gamma: float | None = 1.0
  remat: bool = False
  policy: Callable[..., bool] | None = None
  prevent_cse: bool = False
  name: str | None = None

  def __post_init__(self):
    super().__post_init__(name=self.name)

  def __call__(self, node_coordinates: jnp.ndarray) -> jnp.ndarray:
    if self.learnable_fourier_features:
      positional_encoder = FourierFeatures(self.num_frequencies, self.encoding_dim, self.hidden_dim,
                                           self.gamma, name=self.name + "_fourier_features")
      if self.remat:
        positional_encoder = hk.remat(positional_encoder, policy=self.policy, prevent_cse=self.prevent_cse)
      codes = positional_encoder(node_coordinates)
    else:
      codes = fourier_features(node_coordinates, self.num_frequencies)
    codes = checkpoint_name(codes, "positional_encoder")
    return codes
