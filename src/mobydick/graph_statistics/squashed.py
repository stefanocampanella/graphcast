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

Each graph of the model is a boolean adjacency matrix M (receivers × senders). Every node update of the model keeps
the node's own state (residual updates in the encoders, processors and decoders), so the information a node holds
after an update is that it held before OR that of its senders. With x a boolean vector per level (grid, mesh 0,
mesh 1, ...) marking the nodes that hold information from a set of sources, a graph sending from level s to level t
does x_t <- x_t OR (M x_s). One message passing step on a mesh is x <- x OR (P x). The squashed graph's adjacency is
the product of these operations, in the order of the data flow (`schedule`); for one mesh with L steps:

  S = I + D (I + P)^L E

(I: every grid node keeps its own input, so the squashed graph has a self-loop at every node; the statistics leave
the self-loops out). S has up to (grid nodes)² entries, 4.8e11 at 0.25°: it is never formed. Instead, `reach` applies
the operations to the indicator vectors of a batch of sampled grid nodes, i.e. sparse matrix × dense block products
(one column per source), and thresholds after each operation, so the values stay booleans:
- forward, from sampled senders: the columns of S, i.e. the receivers each sender reaches (out-degrees);
- backward, from sampled receivers, with the transposed operations in reverse order (the transpose of a product is
  the product of the transposes in reverse order): the rows of S, i.e. the senders reaching each receiver
  (in-degrees).
The cost is that of a few sparse matrix-vector products per operation and source: (encoder + L processor + decoder
edges) × sources.

Hierarchical models do not exist yet. For a hierarchy (mesh 0 the finest) `schedule` assumes a down sweep with the
same L steps on every mesh, then the decoders back up to the grid, each one a residual update of the finer level
(as a U-Net's skip connections):
grid -> mesh 0, L steps, mesh 0 -> mesh 1, L steps, ..., mesh K-1 -> mesh K-2, ..., mesh 0 -> grid.

The horizon of a grid node is the geodesic distance to the nodes it reaches. Over the edges of the squashed graph
(pairs of distinct nodes), it is estimated from the sampled senders, each with at most `max_targets` of its receivers
(sampled when it reaches more) weighted by its out-degree / the receivers measured, so each edge counts once.
"""

import dataclasses
from collections.abc import Sequence
from typing import Any

import numpy as np
import scipy.sparse

from mobydick.graph_statistics.geometry import geodesic_lengths
from mobydick.graph_statistics.graphs import Graph
from mobydick.graph_statistics.quantities import described, fraction, summarize, weighted_summarize


@dataclasses.dataclass(frozen=True)
class Operation:
  """x[target] <- x[target] OR (matrix @ x[source]), for one graph of the model (one message passing step for a
  processor, where target = source)."""

  name: str
  source: str
  target: str
  matrix: scipy.sparse.csr_array  # receivers × senders
  transpose: scipy.sparse.csr_array  # senders × receivers: the backward operation


def adjacency(graph: Graph) -> scipy.sparse.csr_array:
  """The adjacency matrix of `graph`: a 1 at (receiver, sender) for every edge (duplicate edges add up, which the
  thresholding in `reach` ignores)."""
  ones = np.ones(len(graph.senders), dtype=np.float32)
  shape = (graph.receiver_nodes.size, graph.sender_nodes.size)
  return scipy.sparse.csr_array((ones, (graph.receivers, graph.senders)), shape=shape)


def _operation(graph: Graph) -> Operation:
  matrix = adjacency(graph)
  return Operation(graph.name, graph.sender_nodes.name, graph.receiver_nodes.name, matrix, matrix.T.tocsr())


def schedule(graphs: Sequence[Graph], steps: int) -> list[Operation]:
  """The operations of a forward pass, in order: each encoder (finest first) followed by `steps` message passing steps
  on its mesh, then the decoders (coarsest first); `graphs` are those of `graphs.model_graphs`."""
  encoders = [g for g in graphs if g.kind == "encoder"]
  processors = [g for g in graphs if g.kind == "processor"]
  decoders = [g for g in graphs if g.kind == "decoder"]
  operations = []
  for encoder, processor in zip(encoders, processors, strict=True):
    operations.append(_operation(encoder))
    operations += [_operation(processor)] * steps
  return operations + [_operation(decoder) for decoder in decoders]


def node_sizes(graphs: Sequence[Graph]) -> dict[str, int]:
  """The number of nodes of every level (grid and meshes), by name."""
  return {nodes.name: nodes.size for g in graphs for nodes in (g.sender_nodes, g.receiver_nodes)}


def reach(
  operations: Sequence[Operation], sizes: dict[str, int], level: str, nodes: np.ndarray, backward: bool = False
) -> np.ndarray:
  """Boolean (len(nodes), sizes[level]) array: row k marks the nodes of `level` that node `nodes[k]` of `level` reaches
  through `operations` (forward), or that reach it (backward). Every node reaches itself."""
  # One column per source: sparse matrix × dense block products.
  state = {name: np.zeros((size, len(nodes)), dtype=np.float32) for name, size in sizes.items()}
  state[level][nodes, np.arange(len(nodes))] = 1.0
  for operation in reversed(operations) if backward else operations:
    if backward:
      source, target, matrix = operation.target, operation.source, operation.transpose
    else:
      source, target, matrix = operation.source, operation.target, operation.matrix
    # Thresholding keeps 0/1 values: path counts would grow as degree^steps.
    state[target] = ((state[target] + matrix @ state[source]) > 0).astype(np.float32)
  return np.ascontiguousarray(state[level].T > 0)


def squashed_statistics(
  graphs: Sequence[Graph],
  steps: int,
  nodes: np.ndarray,
  max_targets: int,
  batch_size: int,
  rng: np.random.Generator,
) -> dict[str, Any]:
  """Statistics of the squashed grid-to-grid graph of `graphs` with `steps` message passing steps per mesh, from the
  sampled grid nodes `nodes`, as senders (out-degrees, horizon) and as receivers (in-degrees). Self-loops are left
  out."""
  grid = next(g.sender_nodes for g in graphs if g.kind == "encoder")
  operations = schedule(graphs, steps)
  sizes = node_sizes(graphs)
  out_degree, in_degree = [], []
  distances, weights, sender_means = [], [], []
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
      lengths = (
        geodesic_lengths(
          np.full(len(receivers), grid.lat[sender]),
          np.full(len(receivers), grid.lon[sender]),
          grid.lat[receivers],
          grid.lon[receivers],
        )
        / 1000
      )
      distances.append(lengths)
      # Each measured receiver stands for out-degree / measured edges of this sender.
      weights.append(np.full(len(lengths), out_degree[-1] / len(lengths)))
      sender_means.append(lengths.mean())
    for receiver, reached in zip(batch, reach(operations, sizes, grid.name, batch, backward=True), strict=True):
      in_degree.append(int(reached.sum()) - int(reached[receiver]))
  out_degree_array, in_degree_array = np.asarray(out_degree), np.asarray(in_degree)
  num_nodes = grid.size
  pairs = np.concatenate(distances) if distances else np.zeros(0)
  pair_weights = np.concatenate(weights) if weights else np.zeros(0)
  return {
    "message_passing_steps": described("Message passing steps on every mesh", steps),
    "nodes": described("Grid nodes (unmasked points)", num_nodes),
    "sampled_nodes": described("Grid nodes sampled as senders and as receivers", len(nodes)),
    "edges": described(
      "Estimated directed edges without self-loops: mean out-degree of the sampled senders × grid nodes",
      float(out_degree_array.mean() * num_nodes),
    ),
    "density": described(
      "Estimated edges / (nodes × (nodes − 1)): the mean fraction of the other grid nodes a node reaches",
      float(out_degree_array.mean() / (num_nodes - 1)) if num_nodes > 1 else 0.0,
    ),
    "out_degree": described(
      "Other grid nodes each sampled sender reaches (its input influences their output)", summarize(out_degree_array)
    ),
    "in_degree": described(
      "Other grid nodes reaching each sampled receiver (their input influences its output)",
      summarize(in_degree_array),
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
      "Mean geodesic length of each sampled sender's edges (km): the horizon of single grid nodes",
      summarize(np.asarray(sender_means)),
    ),
  }
