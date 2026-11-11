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
"""Structure of the graphs: node and edge counts and degrees of every graph, the decoder check, connected components
and shortest paths of the processors, coverage of each pair of levels.

Only the processors' components and shortest paths use networkx, through functions its cugraph backend implements
(`connected_components`, `single_source_shortest_path_length`). Degrees and coverage are numpy counts.

Average clustering, betweenness and closeness were dropped (review of PR #7). The last two add nothing to the shortest
path lengths: on a connected graph of N nodes with mean shortest path length L (over ordered pairs of distinct
nodes), each pair's shortest paths pass through d − 1 intermediate nodes, so the average normalized betweenness is
exactly (L − 1) / (N − 2), and the average closeness is the mean of 1 / (mean distance from each node), about 1 / L
(exact networkx values on icosahedral meshes and open patches of them, Nov 2026: betweenness equal to 6 digits,
closeness within 3 % of 1 / L). On the uniform ocean meshes L grows as N^0.5 (L · edge length ≈ 11,500 km), so both
scale as N^-0.5.
"""

from typing import Any

import networkx as nx
import numpy as np

from mobydick.graph_statistics.graphs import Graph
from mobydick.graph_statistics.quantities import degree_summary, described, fraction, summarize


def node_counts(graph: Graph) -> dict[str, Any]:
  """Sizes of the sender and receiver node sets and the number of edges."""
  return {
    "sender_nodes": described(f"Nodes of the sender set ({graph.sender_nodes.name})", graph.sender_nodes.size),
    "receiver_nodes": described(f"Nodes of the receiver set ({graph.receiver_nodes.name})", graph.receiver_nodes.size),
    "edges": described("Directed edges", len(graph.senders)),
  }


def encoder_statistics(graph: Graph) -> dict[str, Any]:
  """Node and edge counts and the degrees of the senders and receivers.

  Degrees count every node of a set, also those without edges (degree 0): for the grid -> mesh encoder these are the
  unmasked points out of reach of every mesh vertex, and the mesh vertices over land.
  """
  out_degree = np.bincount(graph.senders, minlength=graph.sender_nodes.size)
  in_degree = np.bincount(graph.receivers, minlength=graph.receiver_nodes.size)
  return {
    **node_counts(graph),
    "sender_out_degree": described("Edges per sender node, also nodes without edges", degree_summary(out_degree)),
    "receiver_in_degree": described("Edges per receiver node, also nodes without edges", degree_summary(in_degree)),
  }


def decoder_statistics(graph: Graph) -> dict[str, Any]:
  """Node and edge counts, and whether every receiver has the 3 edges from the vertices of its face (by construction:
  a failure means the edges are not built as expected)."""
  in_degree = np.bincount(graph.receivers, minlength=graph.receiver_nodes.size)
  return {
    **node_counts(graph),
    "expected_edges": described(
      "3 per receiver node, from the vertices of the face containing it", 3 * graph.receiver_nodes.size
    ),
    "three_edges_per_receiver": described("Every receiver node has exactly 3 edges", bool(np.all(in_degree == 3))),
  }


def path_statistics(graph: nx.Graph, nodes: list[int], num_sources: int, rng: np.random.Generator) -> dict[str, Any]:
  """Diameter and shortest path lengths of the connected component `nodes` of the undirected `graph`.

  One breadth-first search runs from every source: all nodes of the component (exact), or `num_sources` sampled ones.
  A search from a node of the component reaches exactly the component, so no subgraph is needed. When sampling:
  - the shortest path lengths are those from the sources to every other node of the component;
  - the diameter is a lower bound: the largest eccentricity (distance to the farthest node) of the sources, improved
    by one more search from the farthest node found (a 2-sweep), which on mesh-like graphs is close to the diameter.
  """
  exact = num_sources == 0 or num_sources >= len(nodes)
  sources = nodes if exact else rng.choice(nodes, num_sources, replace=False).tolist()
  lengths, diameter, farthest = [], 0, sources[0]
  for source in sources:
    # {node: hops from the source} for every node of the component, the source included (0 hops).
    distances = nx.single_source_shortest_path_length(graph, source)
    values = np.fromiter(distances.values(), dtype=np.int64, count=len(distances))
    lengths.append(values[values > 0])
    if values.max() > diameter:
      diameter, farthest = int(values.max()), list(distances)[int(values.argmax())]
  if not exact:
    diameter = max(diameter, max(nx.single_source_shortest_path_length(graph, farthest).values()))
  return {
    "nodes": described("Nodes of the component", len(nodes)),
    "sources": described("Nodes from which shortest paths are computed (all nodes: exact values)", len(sources)),
    "diameter": described(
      "Longest shortest path (hops); when sampling, a lower bound: the largest eccentricity of the sources and of the "
      "node farthest from them",
      diameter,
    ),
    "diameter_exact": described("The diameter is exact (paths from all nodes)", exact),
    "shortest_path_length": described(
      "Hops of the shortest paths from the sources to the other nodes of the component; with all nodes as sources, "
      "the mean is networkx's average_shortest_path_length",
      summarize(np.concatenate(lengths)),
    ),
  }


def processor_statistics(graph: Graph, num_sources: int, rng: np.random.Generator) -> dict[str, Any]:
  """Structure of a mesh graph: degrees, density, one-way edges, connected components and their path metrics.

  Paths are computed on the undirected graph: the processor's edges go both ways (`mesh_graph.faces_to_edges`), which
  `edges_without_reverse` = 0 confirms.
  """
  num_nodes = graph.sender_nodes.size
  num_edges = len(graph.senders)
  degree = np.bincount(graph.senders, minlength=num_nodes)
  # One integer per directed edge, to find the edges whose reverse is missing.
  forward = graph.senders.astype(np.int64) * num_nodes + graph.receivers
  backward = graph.receivers.astype(np.int64) * num_nodes + graph.senders
  undirected = nx.Graph()
  undirected.add_nodes_from(range(num_nodes))
  undirected.add_edges_from(zip(graph.senders.tolist(), graph.receivers.tolist(), strict=True))
  # Sorted node lists, largest component first, so that sampling is reproducible.
  components = sorted((sorted(c) for c in nx.connected_components(undirected)), key=len, reverse=True)
  return {
    "nodes": described("Mesh vertices", num_nodes),
    "edges": described("Directed edges", num_edges),
    "density": described(
      "Directed edges / (nodes × (nodes − 1))", num_edges / (num_nodes * (num_nodes - 1)) if num_nodes > 1 else 0.0
    ),
    "edges_without_reverse": described(
      "Directed edges whose reverse is missing (0 since mesh_graph.faces_to_edges adds the reverses of boundary "
      "edges, Nov 2026)",
      int((~np.isin(forward, backward)).sum()),
    ),
    "out_degree": described("Edges sent per node", degree_summary(degree)),
    "connected_components": described("Connected components of the undirected graph", len(components)),
    "isolated_nodes": described("Nodes without edges", sum(len(c) == 1 for c in components)),
    "components": described(
      "Components of more than one node, largest first",
      [path_statistics(undirected, c, num_sources, rng) for c in components if len(c) > 1],
    ),
  }


def coverage(encoder: Graph, decoder: Graph) -> dict[str, Any]:
  """Nodes of a level and of the next coarser one connected to each other by the encoder, the decoder or both.

  `lower` is the finer level (the grid's unmasked points, or a mesh), `upper` the coarser mesh. A lower node is useful
  in both directions only if it is in both graphs: e.g. an ocean point without encoder edges is predicted but never
  read. An upper node in neither graph exchanges nothing with the lower level (e.g. a vertex over land).
  """
  lower, upper = encoder.sender_nodes, encoder.receiver_nodes
  lower_encoder, lower_decoder = np.zeros(lower.size, bool), np.zeros(lower.size, bool)
  upper_encoder, upper_decoder = np.zeros(upper.size, bool), np.zeros(upper.size, bool)
  lower_encoder[encoder.senders] = True
  lower_decoder[decoder.receivers] = True
  upper_encoder[encoder.receivers] = True
  upper_decoder[decoder.senders] = True
  return {
    "lower": lower.name,
    "upper": upper.name,
    "lower_in_encoder": described(
      f"Nodes of {lower.name} sending encoder edges to {upper.name}", fraction(lower_encoder)
    ),
    "lower_in_decoder": described(
      f"Nodes of {lower.name} receiving decoder edges from {upper.name}", fraction(lower_decoder)
    ),
    "lower_in_both": described(
      f"Nodes of {lower.name} in the encoder and the decoder", fraction(lower_encoder & lower_decoder)
    ),
    "upper_in_encoder": described(
      f"Nodes of {upper.name} receiving encoder edges from {lower.name}", fraction(upper_encoder)
    ),
    "upper_in_decoder": described(
      f"Nodes of {upper.name} sending decoder edges to {lower.name}", fraction(upper_decoder)
    ),
    "upper_in_either": described(
      f"Nodes of {upper.name} in the encoder or the decoder", fraction(upper_encoder | upper_decoder)
    ),
  }
