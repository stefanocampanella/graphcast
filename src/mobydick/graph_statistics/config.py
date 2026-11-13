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
"""Settings of moby-dick graph-statistics: the inputs (mask and meshes, absolute paths as in the training configs)
and the settings of the statistics."""

import pathlib

from pydantic import BaseModel, ConfigDict, Field, NonNegativeInt, PositiveFloat, PositiveInt, field_validator


def _absolute(path: pathlib.Path) -> pathlib.Path:
  if not path.is_absolute():
    raise ValueError(f"{path} is not an absolute path")
  return path


class MeshLevelConfig(BaseModel):
  """One mesh of the model, with its processor's message passing steps."""

  model_config = ConfigDict(extra="forbid")

  path: pathlib.Path = Field(description="Mesh file (.msh, from make-ocean-mesh or make-icosahedral-mesh)")
  message_passing_steps: NonNegativeInt = Field(
    description="Message passing steps of the mesh's processor (the model's gnn_msg_steps for one mesh; in a "
    "hierarchy, the steps on this mesh on the way down the V-cycle)"
  )

  _path_is_absolute = field_validator("path")(_absolute)


class SquashedGraphConfig(BaseModel):
  """Settings of the squashed grid-to-grid graph (`squashed`)."""

  model_config = ConfigDict(extra="forbid")

  num_sources: PositiveInt = Field(
    default=1024,
    description="Grid nodes sampled as senders (forward: out-degrees, horizon) and as receivers (backward: in-degrees)",
  )
  max_targets: PositiveInt = Field(
    default=4096,
    description="Edges per sampled node whose geodesic length is computed (sampled when it has more)",
  )
  batch_size: PositiveInt = Field(
    default=64,
    description="Sources propagated together (columns of the dense blocks: memory grows as batch_size × grid nodes)",
  )


class GraphStatisticsConfig(BaseModel):
  """Settings of moby-dick graph-statistics (unknown keys are rejected)."""

  model_config = ConfigDict(extra="forbid")

  mask: pathlib.Path = Field(description="Training mask written by moby-dick download-mask: its grid and ocean points")
  meshes: list[MeshLevelConfig] = Field(
    min_length=1, description="The model's meshes, finest first (one, or a hierarchy)"
  )
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
    default_factory=SquashedGraphConfig, description="The squashed grid-to-grid graph"
  )

  _mask_is_absolute = field_validator("mask")(_absolute)
