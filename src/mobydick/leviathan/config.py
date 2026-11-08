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
"""Validated settings of the Leviathan model (the `[model]` section of a training config)."""

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, NonNegativeInt, PositiveInt


class LeviathanConfig(BaseModel):
  model_config = ConfigDict(extra="forbid")

  architecture: Literal["leviathan"] = Field(
    default="leviathan", description="The model to train, a key of mobydick.models.MODELS."
  )
  latent_size: PositiveInt
  gnn_msg_steps: PositiveInt
  hidden_layers: NonNegativeInt
  radius_query_fraction_edge_length: Annotated[float, Field(gt=0)]
  per_variable_weights: dict[str, float] = Field(default_factory=dict)
  mesh2grid_edge_normalization_factor: Annotated[float, Field(gt=0)] | None = None
  connected_grid_nodes: bool = Field(
    default=False,
    description="Grid nodes only for the grid points connected by an edge (encoder senders, decoder receivers): "
    "the others only cost memory.",
  )
  level_weights_coord: str = Field(
    default="depth",
    description="Coordinate along the level dimension from which the loss weights of the levels are computed: linear "
    "in it, summing to 1 (e.g. depth, or level for the positions of the selected levels).",
  )
  level_weights_decreasing: bool = Field(
    default=True, description="Whether the level weights decrease as the coordinate grows (deeper levels weigh less)."
  )
