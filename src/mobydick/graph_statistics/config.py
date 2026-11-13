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
"""Settings of moby-dick graph-statistics: the inputs (mask and meshes, absolute paths as in the training configs) and
the settings of the statistics (`GraphStatisticsConfig`), and the ocean regions of the basin connectivity
(`RegionsConfig`, a file of its own: regions may be derived from data under other licenses)."""

import pathlib
from typing import Annotated, Self

from pydantic import (
  BaseModel,
  ConfigDict,
  Field,
  NonNegativeInt,
  PositiveFloat,
  PositiveInt,
  field_validator,
  model_validator,
)


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
  max_steps_to_connect: NonNegativeInt = Field(
    default=128,
    description="Largest number of message passing steps (on every mesh) tried when searching the fewest steps that "
    "connect a basin and its ocean",
  )
  batch_size: PositiveInt = Field(
    default=64,
    description="Sources propagated together (columns of the dense blocks: memory grows as batch_size × grid nodes)",
  )


class RegionConfig(BaseModel):
  """A named region of the ocean: the grid nodes inside any of its polygons of (longitude, latitude) vertices (several
  for archipelagic seas or regions in parts)."""

  model_config = ConfigDict(extra="forbid")

  name: str = Field(min_length=1, description="Name of the region, unique")
  polygons: list[Annotated[list[tuple[float, float]], Field(min_length=3)]] = Field(
    min_length=1,
    description="Polygons of vertices (longitude, latitude) in degrees, longitudes in [-180, 180]; each closes by "
    "itself and must not cross the antimeridian",
  )

  @field_validator("polygons")
  @classmethod
  def _within_bounds(cls, polygons: list[list[tuple[float, float]]]) -> list[list[tuple[float, float]]]:
    for lon, lat in (vertex for polygon in polygons for vertex in polygon):
      if not (-180.0 <= lon <= 180.0 and -90.0 <= lat <= 90.0):
        raise ValueError(f"vertex ({lon}, {lat}) is not a (longitude, latitude) in [-180, 180] x [-90, 90]")
    return polygons


class RegionsConfig(BaseModel):
  """The regions file given to moby-dick graph-statistics: [[regions]] tables only, with unique names."""

  model_config = ConfigDict(extra="forbid")

  regions: list[RegionConfig] = Field(default=[], description="Regions referred to by basin_connections")

  @model_validator(mode="after")
  def _names_are_unique(self) -> Self:
    names = [region.name for region in self.regions]
    if len(set(names)) != len(names):
      raise ValueError(f"region names must be unique: {names}")
    return self

  def polygons(self) -> dict[str, list[list[tuple[float, float]]]]:
    """The polygons of every region, by name."""
    return {region.name: region.polygons for region in self.regions}


def undefined_regions(config: "GraphStatisticsConfig", regions: RegionsConfig) -> list[str]:
  """Names used by `config.basin_connections` that `regions` does not define, in order of first use."""
  defined = {region.name for region in regions.regions}
  used = (name for connection in config.basin_connections for name in (connection.basin, connection.ocean))
  return list(dict.fromkeys(name for name in used if name not in defined))


class BasinConnectionConfig(BaseModel):
  """A semi-enclosed basin or inner sea and the nearby ocean it opens to, both names of regions of the regions file."""

  model_config = ConfigDict(extra="forbid")

  basin: str = Field(description="Region of the basin, e.g. the Black Sea")
  ocean: str = Field(description="Region of the nearby ocean, e.g. the Aegean Sea")


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
  basin_connections: list[BasinConnectionConfig] = Field(
    default=[],
    description="Pairs of a basin and its nearby ocean whose connectivity in the squashed graph is measured, by the "
    "names of regions of the regions file",
  )

  _mask_is_absolute = field_validator("mask")(_absolute)
