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
"""The node sets and graphs of a model on a grid and a hierarchy of meshes.

The edges come from the `mesh_connectivity` functions the model uses, so the statistics describe the graphs it trains
on (`tests/test_graph_statistics.py` checks it for one mesh):
- grid -> mesh 0: radius query from the unmasked grid points to the mesh vertices (radius: a fraction of the mesh
  size at each vertex);
- mesh i-1 -> mesh i: the same radius query from the vertices of the finer mesh;
- processors: the mesh's edges (`MeshGraph.edges`: every face edge in both directions, or the multi-mesh's edges);
- mesh i -> mesh i-1 and mesh 0 -> grid: from the 3 vertices of the coarser face containing each receiver.
Hierarchical models do not exist yet: when they define their mesh-to-mesh graphs, `model_graphs` should call the same
functions.
"""

import dataclasses
from collections.abc import Sequence
from typing import Literal

import numpy as np

from mobydick import mesh_connectivity
from mobydick.dataloader_utils import Grid
from mobydick.gis_utils import cartesian_to_lat_lon
from mobydick.mesh_graph import MeshData, MeshGraph

GraphKind = Literal["encoder", "processor", "decoder"]


@dataclasses.dataclass(frozen=True)
class NodeSet:
  """Nodes of a grid (its unmasked points) or of a mesh (its vertices), with their positions in degrees."""

  name: str  # "grid" or "mesh_<level>"
  lat: np.ndarray
  lon: np.ndarray

  @property
  def size(self) -> int:
    return len(self.lat)


@dataclasses.dataclass(frozen=True)
class Graph:
  """Directed edges from `sender_nodes` to `receiver_nodes`: edge k goes from sender node `senders[k]` to receiver node
  `receivers[k]`. Processors have the same sender and receiver nodes, and their mesh (for the triangle quality)."""

  name: str  # e.g. "grid_to_mesh_0", "mesh_0", "mesh_0_to_grid"
  kind: GraphKind
  senders: np.ndarray
  receivers: np.ndarray
  sender_nodes: NodeSet
  receiver_nodes: NodeSet
  mesh: MeshGraph | None = None


def model_graphs(meshes: Sequence[MeshData], grid: Grid, radius_query_fraction: float) -> list[Graph]:
  """The graphs of a model on `grid` and `meshes` (finest first): the encoders (finest first), the processors (finest
  first) and the decoders (coarsest first), i.e. in the order data flows through them.

  Grid nodes are the unmasked grid points, numbered row-major over the flattened grid as with
  `model.connected_grid_nodes`; mesh nodes are the mesh vertices.
  """
  mask = np.asarray(grid.mask, dtype=bool)
  flat_mask = mask.reshape(-1)
  # The mesh_connectivity functions index the flattened grid: map those indices to the unmasked points.
  point_to_node = np.cumsum(flat_mask) - 1
  grid_lon, grid_lat = np.meshgrid(grid.lon, grid.lat)
  # `lower` is the finer level the next mesh connects to: the grid, then each mesh in turn.
  lower = NodeSet("grid", grid_lat.reshape(-1)[flat_mask], grid_lon.reshape(-1)[flat_mask])
  lower_mesh: MeshGraph | None = None
  encoders, processors, decoders = [], [], []
  for level, mesh_data in enumerate(meshes):
    mesh = mesh_data.mesh_graph
    if mesh_data.mesh_size is None:
      raise ValueError(f"mesh {level} has no mesh size: the encoder radius is a fraction of it")
    radius = radius_query_fraction * mesh_data.mesh_size
    nodes = NodeSet(f"mesh_{level}", *cartesian_to_lat_lon(np.asarray(mesh.vertices)))
    if lower_mesh is None:
      # The grid level: the model's own encoder and decoder functions.
      grid_senders, encoder_receivers = mesh_connectivity.radius_query_indices(
        grid_latitude=grid.lat, grid_longitude=grid.lon, mesh=mesh, radius=radius, mask=mask
      )
      encoder_senders = point_to_node[grid_senders]
      decoder_senders, grid_receivers = mesh_connectivity.get_mesh_to_grid_edges(
        grid_latitude=grid.lat, grid_longitude=grid.lon, mesh=mesh, mask=mask
      )
      decoder_receivers = point_to_node[grid_receivers]
    else:
      # A coarser mesh: the same rules, with the finer mesh's vertices in place of the grid points.
      encoder_senders, encoder_receivers = mesh_connectivity.get_points_to_mesh_edges(
        senders_position=lower_mesh.vertices, receivers_mesh=mesh, radius=radius
      )
      decoder_senders, decoder_receivers = mesh_connectivity.get_mesh_to_points_edges(
        senders_mesh=mesh, receivers_position=lower_mesh.vertices
      )
    encoders.append(Graph(f"{lower.name}_to_{nodes.name}", "encoder", encoder_senders, encoder_receivers, lower, nodes))
    senders, receivers = (np.asarray(e) for e in mesh.edges)
    processors.append(Graph(nodes.name, "processor", senders, receivers, nodes, nodes, mesh))
    decoders.append(Graph(f"{nodes.name}_to_{lower.name}", "decoder", decoder_senders, decoder_receivers, nodes, lower))
    lower, lower_mesh = nodes, mesh
  return encoders + processors + decoders[::-1]
