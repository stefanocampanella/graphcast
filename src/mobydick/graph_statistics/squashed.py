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
"""The squashed grid-to-grid graph of a model: an edge i -> j when the input at grid node i can reach the output at
grid node j through the encoders, the message passing steps and the decoders.

Reach as boolean linear algebra
-------------------------------
Each graph of the model is a boolean adjacency matrix M (receivers × senders, True for an edge). For each level (grid,
mesh 0, mesh 1, ...) a boolean vector x marks the nodes whose state holds information from a set of sources. Every
node update of the model keeps the node's own state (residual updates in the encoders, processors and decoders), so a
graph sending from level s to level t does

  x_t <- x_t OR (M x_s)

with the boolean product (OR of ANDs). One message passing step on a mesh is x <- x OR (P x) = (I + P) x, "+" being
OR. The identity matters: P^L alone holds the walks of exactly L edges, but information also arrives over shorter
paths and stays. By induction, (I + P)^L = I + P + P^2 + ... + P^L, i.e. (I + P)^L x marks the nodes within at most
L hops: true for L = 1, and (I + P)^(L+1) x = (I + P)^L x OR P (I + P)^L x adds the neighbours of the nodes within L
hops, which are the nodes within L + 1 hops. This is the OR of the powers of P that the comment of `graph_pruning.py`
sums, computed one step at a time.

The schedule (`schedule`) is a V-cycle, as in the planned hierarchical processor (mesh 0 the finest, L_i steps on
mesh i):
  grid -> mesh 0, L_0 steps, mesh 0 -> mesh 1, L_1 steps, ..., mesh K-1 -> mesh K-2, ..., mesh 1 -> mesh 0,
  mesh 0 -> grid.
On the way up, each decoder adds the coarser mesh's result to the finer mesh's state computed on the way down (the
skip connections of a U-Net): x_{i-1} <- x_{i-1} OR (D_i x_i), with x_{i-1} the state after its L_{i-1} steps. The
finer state is kept, so the paths that never leave mesh i-1 still count; replacing it by D_i x_i would lose them.
There are no message passing steps on the way up. For one mesh with L steps, and for two meshes:

  S = I + D_0 (I + P_0)^L_0 E_0
  S = I + D_0 [(I + P_0)^L_0 + D_1 (I + P_1)^L_1 E_1 (I + P_0)^L_0] E_0

(I: every grid node keeps its own input, so the squashed graph has a self-loop at every node; the statistics leave
them out.)

Computing it
------------
S has up to (grid nodes)² entries, 4.8e11 at 0.25°: it is never formed. `reach` applies the operations to boolean
indicator vectors of a batch of sampled grid nodes (one column per node), i.e. products of the sparse boolean matrices
(scipy computes them with OR and AND) with dense boolean blocks. The cost is (encoder + steps × processor + decoder
edges) × sampled nodes.

Forward or backward
-------------------
- Forward, from sampled senders, gives columns of S: the grid nodes the input at each sender reaches, i.e. its
  domain of influence (out-degree; how far a perturbation of one input spreads in one model step).
- Backward, from sampled receivers, gives rows of S: the grid nodes whose input reaches each receiver, i.e. the
  receptive field of one prediction (in-degree; e.g. the halo a domain decomposition would need). It applies the
  transposed operations in reverse order (the transpose of a product is the product of the transposes in reverse
  order): y_s <- y_s OR (M^T y_t).
The two differ per node: encoders (radius query) and decoders (containing triangle) are not transposes of each
other, and a grid node without encoder edges reaches no other node but is reached through the decoder. The mean
out-degree and in-degree are equal (both are edges / nodes), so their sampled means estimate the same number. A node
is on a path from the grid to the grid (the mask of `graph_pruning.py`) only if it is reached forward from some grid
node and backward from some grid node: pruning needs both directions, statistics of one end need one.

The horizon is the geodesic length of the squashed graph's edges, estimated from the sampled senders, each with at
most `max_targets` of its receivers (sampled when it reaches more) weighted by its out-degree / the receivers
measured, so each edge counts once. The mean distance per sampled sender and per sampled receiver describe single
grid nodes (domain of influence and receptive field).

Basins
------
`basin_connectivity` measures how a semi-enclosed basin or inner sea (the Black Sea, the Gulf of California) is
connected to the nearby ocean, two regions of grid nodes (`regions`). A column marking every node of a region
propagates the union of their reaches, so the values are exact with one column per region and direction: the shares
of basin nodes whose input reaches the ocean, of ocean nodes depending on the basin, and the reverse. The fewest steps
that connect them (`steps_to_connect`, the same steps on every mesh) come from a bisection over the steps: reach only
grows with them.
"""

import dataclasses
from collections.abc import Sequence
from typing import Any

import numpy as np
import scipy.sparse

from mobydick.graph_statistics.geometry import geodesic_lengths
from mobydick.graph_statistics.graphs import Graph, NodeSet
from mobydick.graph_statistics.quantities import described, fraction, summarize, weighted_summarize


@dataclasses.dataclass(frozen=True)
class Operation:
  """x[target] <- x[target] OR (matrix x[source]), for one graph of the model (one message passing step for a
  processor, where target = source)."""

  name: str
  source: str
  target: str
  matrix: scipy.sparse.csr_array  # Boolean, receivers × senders.
  transpose: scipy.sparse.csr_array  # Boolean, senders × receivers: the backward operation.


def adjacency(graph: Graph) -> scipy.sparse.csr_array:
  """The boolean adjacency matrix of `graph`: True at (receiver, sender) for every edge."""
  edges = np.ones(len(graph.senders), dtype=bool)
  shape = (graph.receiver_nodes.size, graph.sender_nodes.size)
  # Duplicate edges are summed when converting to CSR, which for booleans is OR.
  return scipy.sparse.csr_array((edges, (graph.receivers, graph.senders)), shape=shape)


def _operation(graph: Graph) -> Operation:
  matrix = adjacency(graph)
  return Operation(graph.name, graph.sender_nodes.name, graph.receiver_nodes.name, matrix, matrix.T.tocsr())


def graph_operations(graphs: Sequence[Graph]) -> dict[str, Operation]:
  """The operation of every graph, by name: build them once when several schedules use them."""
  return {graph.name: _operation(graph) for graph in graphs}


def schedule(
  graphs: Sequence[Graph], steps: Sequence[int], operations: dict[str, Operation] | None = None
) -> list[Operation]:
  """The operations of a forward pass in V-cycle order: each encoder (finest first) followed by `steps[i]` message
  passing steps on its mesh i, then the decoders (coarsest first); `graphs` are those of `graphs.model_graphs`, and
  `operations` those of `graph_operations` (built here if not given)."""
  operations = operations or graph_operations(graphs)
  encoders = [operations[g.name] for g in graphs if g.kind == "encoder"]
  processors = [operations[g.name] for g in graphs if g.kind == "processor"]
  decoders = [operations[g.name] for g in graphs if g.kind == "decoder"]
  forward = []
  for encoder, processor, mesh_steps in zip(encoders, processors, steps, strict=True):
    # The same matrix for every step: the list holds references, not copies.
    forward += [encoder, *[processor] * mesh_steps]
  return forward + decoders


def node_sizes(graphs: Sequence[Graph]) -> dict[str, int]:
  """The number of nodes of every level (grid and meshes), by name."""
  return {nodes.name: nodes.size for g in graphs for nodes in (g.sender_nodes, g.receiver_nodes)}


def reach(
  operations: Sequence[Operation], sizes: dict[str, int], level: str, nodes: np.ndarray, backward: bool = False
) -> np.ndarray:
  """Boolean (len(nodes), sizes[level]) array: row k marks the nodes of `level` that node `nodes[k]` of `level` reaches
  through `operations` (forward), or that reach it (backward). Every node reaches itself."""
  # One column per source node: sparse boolean matrix × dense boolean block.
  start = np.zeros((sizes[level], len(nodes)), dtype=bool)
  start[nodes, np.arange(len(nodes))] = True
  return np.ascontiguousarray(propagate(operations, sizes, level, start, backward).T)


def propagate(
  operations: Sequence[Operation], sizes: dict[str, int], level: str, start: np.ndarray, backward: bool = False
) -> np.ndarray:
  """Boolean (sizes[level], k) block: column c marks the nodes of `level` reached from (forward), or reaching
  (backward), any node marked in column c of `start`, a (sizes[level], k) block. A column marking a set of nodes gives
  the union of their reaches: the reach of a whole region in one product per operation."""
  state = {name: np.zeros((size, start.shape[1]), dtype=bool) for name, size in sizes.items()}
  state[level] = start.copy()
  for operation in reversed(operations) if backward else operations:
    if backward:
      source, target, matrix = operation.target, operation.source, operation.transpose
    else:
      source, target, matrix = operation.source, operation.target, operation.matrix
    # The product is computed before the OR, so a processor step (source = target) uses the state before the step.
    state[target] |= matrix @ state[source]
  return state[level]


def _distances_km(nodes: NodeSet, node: int, others: np.ndarray) -> np.ndarray:
  """Geodesic distances (km) from `node` to `others`, nodes of the same set."""
  lat, lon = np.full(len(others), nodes.lat[node]), np.full(len(others), nodes.lon[node])
  return geodesic_lengths(lat, lon, nodes.lat[others], nodes.lon[others]) / 1000


def squashed_statistics(
  graphs: Sequence[Graph],
  steps: Sequence[int],
  nodes: np.ndarray,
  max_targets: int,
  batch_size: int,
  rng: np.random.Generator,
) -> dict[str, Any]:
  """Statistics of the squashed grid-to-grid graph of `graphs` with `steps[i]` message passing steps on mesh i, from
  the sampled grid nodes `nodes`, as senders (forward: out-degrees, horizon, domains of influence) and as receivers
  (backward: in-degrees, receptive fields). Self-loops are left out."""
  grid = next(g.sender_nodes for g in graphs if g.kind == "encoder")
  operations = schedule(graphs, steps)
  sizes = node_sizes(graphs)
  out_degree, in_degree = [], []
  distances, weights, sender_means, receiver_means = [], [], [], []
  for start in range(0, len(nodes), batch_size):
    batch = nodes[start : start + batch_size]
    for sender, reached in zip(batch, reach(operations, sizes, grid.name, batch), strict=True):
      receivers = np.flatnonzero(reached)
      receivers = receivers[receivers != sender]
      out_degree.append(len(receivers))
      if not len(receivers):
        continue
      if len(receivers) > max_targets:
        receivers = rng.choice(receivers, max_targets, replace=False)
      lengths = _distances_km(grid, sender, receivers)
      distances.append(lengths)
      # Each measured edge stands for out-degree / measured edges of this sender.
      weights.append(np.full(len(lengths), out_degree[-1] / len(lengths)))
      sender_means.append(lengths.mean())
    for receiver, reaching in zip(batch, reach(operations, sizes, grid.name, batch, backward=True), strict=True):
      senders = np.flatnonzero(reaching)
      senders = senders[senders != receiver]
      in_degree.append(len(senders))
      if not len(senders):
        continue
      if len(senders) > max_targets:
        senders = rng.choice(senders, max_targets, replace=False)
      receiver_means.append(_distances_km(grid, receiver, senders).mean())
  out_degree_array, in_degree_array = np.asarray(out_degree), np.asarray(in_degree)
  num_nodes = grid.size
  pairs = np.concatenate(distances) if distances else np.zeros(0)
  pair_weights = np.concatenate(weights) if weights else np.zeros(0)
  return {
    "message_passing_steps": described("Message passing steps on each mesh, finest first (V-cycle)", list(steps)),
    "nodes": described("Grid nodes (unmasked points)", num_nodes),
    "sampled_nodes": described("Grid nodes sampled as senders (forward) and as receivers (backward)", len(nodes)),
    "edges": described(
      "Estimated directed edges without self-loops: mean out-degree of the sampled senders × grid nodes",
      float(out_degree_array.mean() * num_nodes),
    ),
    "density": described(
      "Estimated edges / (nodes × (nodes − 1)): the mean fraction of the other grid nodes a node reaches",
      float(out_degree_array.mean() / (num_nodes - 1)) if num_nodes > 1 else 0.0,
    ),
    "out_degree": described(
      "Other grid nodes each sampled sender reaches: the size of its domain of influence", summarize(out_degree_array)
    ),
    "in_degree": described(
      "Other grid nodes reaching each sampled receiver: the size of its receptive field", summarize(in_degree_array)
    ),
    "senders_without_edges": described(
      "Sampled senders reaching no other grid node (no encoder edge: their input reaches only their own output)",
      fraction(out_degree_array == 0),
    ),
    "receivers_without_edges": described(
      "Sampled receivers reached by no other grid node", fraction(in_degree_array == 0)
    ),
    "measured_edges": described(
      "Edges whose geodesic length was computed (at most max_targets per sampled sender)", len(pairs)
    ),
    "horizon_km": described(
      "Geodesic lengths of the edges (km) on the WGS84 ellipsoid, each sender's measured edges weighted by its "
      "out-degree / their number: the mean is the mean distance between two grid nodes linked by the squashed graph",
      weighted_summarize(pairs, pair_weights),
    ),
    "sender_mean_distance_km": described(
      "Mean geodesic distance from each sampled sender to the grid nodes it reaches (km): domain of influence",
      summarize(np.asarray(sender_means)),
    ),
    "receiver_mean_distance_km": described(
      "Mean geodesic distance from each sampled receiver to the grid nodes reaching it (km): receptive field",
      summarize(np.asarray(receiver_means)),
    ),
  }


def steps_to_connect(
  graphs: Sequence[Graph],
  operations: dict[str, Operation],
  sources: np.ndarray,
  targets: np.ndarray,
  max_steps: int,
) -> int | None:
  """The fewest message passing steps (the same on every mesh) with which the input at some grid node of `sources`
  reaches the output at some grid node of `targets` (boolean masks over the grid nodes), or None if `max_steps` are not
  enough. Reach grows with the steps (each step ORs into the state), so a bisection finds it."""
  grid = next(g.sender_nodes for g in graphs if g.kind == "encoder")
  sizes = node_sizes(graphs)
  num_meshes = sum(g.kind == "processor" for g in graphs)

  def connected(steps: int) -> bool:
    forward = schedule(graphs, [steps] * num_meshes, operations)
    return bool(np.any(propagate(forward, sizes, grid.name, sources[:, None])[:, 0] & targets))

  if not (sources.any() and targets.any() and connected(max_steps)):
    return None
  # Invariant: not connected with `low` steps (-1: none tried), connected with `high`.
  low, high = -1, max_steps
  while high - low > 1:
    middle = (low + high) // 2
    low, high = (low, middle) if connected(middle) else (middle, high)
  return high


def basin_connectivity(
  graphs: Sequence[Graph],
  steps: Sequence[int],
  basin_name: str,
  basin: np.ndarray,
  ocean_name: str,
  ocean: np.ndarray,
  max_steps: int,
  operations: dict[str, Operation] | None = None,
) -> dict[str, Any]:
  """Connectivity of a basin and its nearby ocean (boolean masks over the grid nodes) in the squashed graph with
  `steps[i]` message passing steps on mesh i: exact values, from the union of each region (one column each, forward
  and backward), and the fewest steps that connect them in each direction."""
  operations = operations or graph_operations(graphs)
  grid = next(g.sender_nodes for g in graphs if g.kind == "encoder")
  sizes = node_sizes(graphs)
  forward = schedule(graphs, steps, operations)
  regions = np.column_stack([basin, ocean])
  # Columns: from the basin, from the ocean; forward: the nodes they reach, backward: the nodes reaching them.
  reached = propagate(forward, sizes, grid.name, regions)
  reaching = propagate(forward, sizes, grid.name, regions, backward=True)

  def share(selected: np.ndarray, region: np.ndarray) -> dict[str, float]:
    return fraction(selected[region]) if region.any() else {"count": 0, "percent": float("nan")}

  return {
    "basin": basin_name,
    "ocean": ocean_name,
    "basin_nodes": described("Grid nodes in the basin region", int(basin.sum())),
    "ocean_nodes": described("Grid nodes in the ocean region", int(ocean.sum())),
    "basin_nodes_reaching_ocean": described(
      "Basin nodes whose input reaches the output at some ocean node", share(reaching[:, 1], basin)
    ),
    "ocean_nodes_reached_from_basin": described(
      "Ocean nodes whose output depends on the input at some basin node", share(reached[:, 0], ocean)
    ),
    "ocean_nodes_reaching_basin": described(
      "Ocean nodes whose input reaches the output at some basin node", share(reaching[:, 0], ocean)
    ),
    "basin_nodes_reached_from_ocean": described(
      "Basin nodes whose output depends on the input at some ocean node", share(reached[:, 1], basin)
    ),
    "steps_basin_to_ocean": described(
      f"Fewest message passing steps on every mesh for an input in the basin to reach an output in the ocean (null: "
      f"more than {max_steps})",
      steps_to_connect(graphs, operations, basin, ocean, max_steps),
    ),
    "steps_ocean_to_basin": described(
      f"Fewest message passing steps on every mesh for an input in the ocean to reach an output in the basin (null: "
      f"more than {max_steps})",
      steps_to_connect(graphs, operations, ocean, basin, max_steps),
    ),
  }
