# SPDX-FileCopyrightText: 2026 Stefano Campanella
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
"""Settings of moby-dick graph-statistics."""

from pydantic import BaseModel, ConfigDict, Field, NonNegativeInt, PositiveFloat, PositiveInt


class SquashedGraphConfig(BaseModel):
  """Settings of the squashed grid-to-grid graphs (`squashed`)."""

  model_config = ConfigDict(extra="forbid")

  message_passing_steps: list[PositiveInt] = Field(
    default=[],
    description="One squashed graph per entry, with this many message passing steps on every mesh (none: no squashed "
    "graph)",
  )
  num_sources: PositiveInt = Field(
    default=1024, description="Grid nodes sampled as senders (out-degrees, horizon) and as receivers (in-degrees)"
  )
  max_targets: PositiveInt = Field(
    default=4096,
    description="Edges per sampled sender whose geodesic length is computed (sampled when it reaches more grid nodes)",
  )
  batch_size: PositiveInt = Field(
    default=64,
    description="Sources propagated together (columns of the dense blocks: memory grows as batch_size × grid nodes)",
  )


class GraphStatisticsConfig(BaseModel):
  """Settings of moby-dick graph-statistics (unknown keys are rejected)."""

  model_config = ConfigDict(extra="forbid")

  radius_query_fraction_edge_length: PositiveFloat = Field(
    default=0.6,
    description="Encoder radius as a fraction of the longest triangle edge at each mesh vertex, as the model's "
    "radius_query_fraction_edge_length: keep it equal to the training config's",
  )
  num_sources: NonNegativeInt = Field(
    default=32,
    description="Nodes per connected component from which the shortest paths are computed (0: all nodes, exact, "
    "one breadth-first search per node)",
  )
  seed: int = Field(default=0, description="Seed of the sampled nodes")
  length_bin_width_km: PositiveFloat = Field(default=5.0, description="Width of the edge length histogram bins")
  quality_bins: PositiveInt = Field(
    default=20,
    description="Bins of the triangle quality histograms per unit (radius ratio over [0, 1], volume-length over "
    "[-1, 1] with twice as many, angles over [0°, 180°])",
  )
  squashed: SquashedGraphConfig = Field(
    default_factory=SquashedGraphConfig, description="The squashed grid-to-grid graphs"
  )
