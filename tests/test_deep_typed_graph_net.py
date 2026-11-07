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
"""Tests of the configurable graph network (deep_typed_graph_net.py) on a small hand-made graph."""

import haiku as hk
import jax
import numpy as np
import pytest

from graphcast import deep_typed_graph_net, typed_graph

NUM_NODES = 4
SENDERS, RECEIVERS = np.array([0, 1, 2, 3, 0]), np.array([1, 2, 3, 0, 2])


def _graph(context=None) -> typed_graph.TypedGraph:
  rng = np.random.default_rng(0)
  return typed_graph.TypedGraph(
    context=typed_graph.Context(n_graph=np.array([1]), features=() if context is None else context),
    nodes={"n": typed_graph.NodeSet(n_node=np.array([NUM_NODES]), features=rng.normal(size=(NUM_NODES, 3)))},
    edges={
      typed_graph.EdgeSetKey("e", ("n", "n")): typed_graph.EdgeSet(
        n_edge=np.array([len(SENDERS)]),
        indices=typed_graph.EdgesIndices(senders=SENDERS, receivers=RECEIVERS),
        features=rng.normal(size=(len(SENDERS), 2)),
      )
    },
  )


def _network(**options):
  settings = {
    "node_latent_size": {"n": 8},
    "edge_latent_size": {"e": 8},
    "mlp_hidden_size": 8,
    "mlp_num_hidden_layers": 1,
    "num_message_passing_steps": 2,
    "node_output_size": {"n": 5},
    "remat": False,
    **options,
  }
  return deep_typed_graph_net.DeepTypedGraphNet(**settings)


def _run(graph=None, **options):
  transformed = hk.transform(lambda g: _network(**options)(g))
  graph = _graph() if graph is None else graph
  params = transformed.init(jax.random.key(0), graph)
  return transformed.apply(params, None, graph), params


@pytest.mark.parametrize(
  "options",
  [
    {},
    {"use_concat_trick": False},
    {"use_layer_norm": False},
    {"include_sent_messages_in_node_update": True},
    {"activation": "swish"},
    {"activation": "identity"},
    {"activation": "tanh"},
    {"aggregate_edges_for_nodes_fn": "segment_mean"},
    {"aggregate_normalization": 2.0},
    {"embed_edges": False, "edge_latent_size": {"e": 2}},
    {"num_processor_repetitions": 2},
    {"scan": True},
    {"edge_output_size": {"e": 3}},
  ],
  ids=lambda options: "-".join(f"{k}={v}" for k, v in options.items()) or "defaults",
)
def test_options(options):
  """Every option builds a network whose outputs have the requested sizes and are finite."""
  output, _ = _run(**options)
  nodes = output.nodes["n"].features
  assert nodes.shape == (NUM_NODES, 5) and np.isfinite(nodes).all()
  if "edge_output_size" in options:
    assert output.edges[typed_graph.EdgeSetKey("e", ("n", "n"))].features.shape == (len(SENDERS), 3)


def test_repetitions_reuse_the_processor():
  """Repeating the processor adds no parameters."""
  _, once = _run()
  _, twice = _run(num_processor_repetitions=2)
  assert jax.tree.structure(once) == jax.tree.structure(twice)


def test_context_features_are_given_to_every_node():
  """Graph-level (context) features are concatenated to each node's features before the embedding."""
  _, params = _run(_graph(context=np.ones((1, 4))), use_concat_trick=False)
  _, without = _run(use_concat_trick=False)
  embedder = next(name for name in params if "encoder_nodes_n" in name and name.endswith("linear_0"))
  assert params[embedder]["w"].shape[0] == without[embedder]["w"].shape[0] + 4


@pytest.mark.parametrize(
  ("options", "message"),
  [
    ({"activation": "no_such_function"}, "Unknown activation"),
    ({"aggregate_edges_for_nodes_fn": "no_such_function"}, "Unknown aggregate"),
  ],
)
def test_invalid_names(options, message):
  """Activations and aggregations are looked up by name: unknown names are errors."""
  with pytest.raises(ValueError, match=message):
    hk.transform(lambda: _network(**options)).init(None)


def test_aggregate_normalization_requires_a_sum():
  """Normalizing the aggregated messages only makes sense for sums."""
  with pytest.raises(AssertionError):
    hk.transform(lambda: _network(aggregate_edges_for_nodes_fn="segment_mean", aggregate_normalization=2.0)).init(None)
