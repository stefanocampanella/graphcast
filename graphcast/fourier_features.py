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
"""Learnable Fourier features, see: https://arxiv.org/abs/2006.10739."""
import dataclasses

import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np

@dataclasses.dataclass
class FourierFeaturesEncoder(hk.Module):
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
    # Keep only leading and last dimensions
    values = values.reshape((input_shape[0], -1, input_shape[-1]))
    features = mlp(self.fourier_features(values, frequencies))
    # Restore all but the last dimension
    features = features.reshape(input_shape[:-1] + (self.encoding_dim,))

    return features

  def fourier_features(
      self,
      values: jnp.ndarray,
      frequencies: jnp.ndarray) -> jnp.ndarray:
    values = values.reshape(-1, values.shape[-1])
    values = 2 * np.pi * values @  frequencies
    return jnp.concatenate([jnp.cos(values), jnp.sin(values)], axis=-1) / jnp.sqrt(self.num_frequencies)