"""Tests of the typed graph networks (typed_graph_net.py) on a hand-made graph with results computed by hand."""

import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from graphcast import typed_graph, typed_graph_net

# Three nodes a, b, c (features 1, 2, 4) and edges a->b, b->c, c->a, a->c (features 10, 20, 30, 40).
NODES = np.array([[1.0], [2.0], [4.0]])
SENDERS, RECEIVERS = np.array([0, 1, 2, 0]), np.array([1, 2, 0, 2])
EDGES = np.array([[10.0], [20.0], [30.0], [40.0]])
KEY = typed_graph.EdgeSetKey("e", ("n", "n"))


def _graph(nodes=NODES, edges=EDGES, senders=SENDERS, receivers=RECEIVERS, context=np.array([[100.0]])):
  return typed_graph.TypedGraph(
    context=typed_graph.Context(n_graph=np.array([1]), features=context),
    nodes={"n": typed_graph.NodeSet(n_node=np.array([len(nodes)]), features=nodes)},
    edges={
      typed_graph.EdgeSetKey("e", ("n", "n")): typed_graph.EdgeSet(
        n_edge=np.array([len(senders)]),
        indices=typed_graph.EdgesIndices(senders=senders, receivers=receivers),
        features=edges,
      )
    },
  )


def _edge_fn(edges, senders, receivers, globals_):
  return edges + senders + 2 * receivers + globals_


def _node_fn(nodes, sent, received, globals_):
  return nodes + sent["e"] + 10 * received["e"] + globals_


def test_graph_network_by_hand():
  """Edges see their features, both ends and the globals; nodes see the sums of the edges they send and receive; the
  globals see the sums of all nodes and edges."""
  network = typed_graph_net.GraphNetwork(
    update_edge_fn={"e": _edge_fn},
    update_node_fn={"n": _node_fn},
    update_global_fn=lambda nodes, edges, globals_: globals_ + nodes["n"] + edges["e"],
  )
  updated = network(_graph())

  edges = EDGES + NODES[SENDERS] + 2 * NODES[RECEIVERS] + 100.0
  np.testing.assert_allclose(updated.edges[KEY].features, edges)
  sent = np.array([edges[0] + edges[3], edges[1], edges[2]])
  received = np.array([edges[2], edges[0], edges[1] + edges[3]])
  nodes = NODES + sent + 10 * received + 100.0
  np.testing.assert_allclose(updated.nodes["n"].features, nodes)
  np.testing.assert_allclose(updated.context.features, [[100.0 + nodes.sum() + edges.sum()]])


@pytest.mark.parametrize("include_sent", [False, True])
def test_interaction_network(include_sent):
  """An interaction network has no globals; the node update sees the received (and optionally the sent) edges."""
  network = typed_graph_net.InteractionNetwork(
    update_edge_fn={"e": lambda e, s, r: e + s - r},
    update_node_fn={"n": (lambda n, s, r: n + s["e"] + r["e"]) if include_sent else (lambda n, r: n + r["e"])},
    include_sent_messages_in_node_update=include_sent,
  )
  updated = network(_graph())

  edges = EDGES + NODES[SENDERS] - NODES[RECEIVERS]
  received = np.array([edges[2], edges[0], edges[1] + edges[3]])
  sent = np.array([edges[0] + edges[3], edges[1], edges[2]])
  np.testing.assert_allclose(updated.nodes["n"].features, NODES + received + (sent if include_sent else 0.0))
  np.testing.assert_allclose(updated.context.features, [[100.0]])


def test_graph_map_features():
  """Nodes, edges and globals are embedded independently."""
  updated = typed_graph_net.GraphMapFeatures(
    embed_edge_fn={"e": lambda e: 2 * e}, embed_node_fn={"n": lambda n: n + 1}, embed_global_fn=lambda g: -g
  )(_graph())
  np.testing.assert_allclose(updated.edges[KEY].features, 2 * EDGES)
  np.testing.assert_allclose(updated.nodes["n"].features, NODES + 1)
  np.testing.assert_allclose(updated.context.features, [[-100.0]])


@pytest.mark.parametrize("chunk_size", [1, 3, 4, 100])
def test_edge_chunks_equal_the_whole_update(chunk_size):
  """Updating the edges in chunks (the last one padded with edges from node 0 to node 0, dropped afterwards) gives
  the same edges, nodes and gradients as updating them at once, also with a learned edge function."""
  rng = np.random.default_rng(0)
  num_nodes, num_edges = 5, 11
  nodes = rng.normal(size=(num_nodes, 3)).astype(np.float32)
  edges = rng.normal(size=(num_edges, 2)).astype(np.float32)
  senders, receivers = rng.integers(num_nodes, size=num_edges), rng.integers(num_nodes, size=num_edges)

  def run(chunk_size, nodes):
    def edge_fn(e, s, r, g):
      del g
      return hk.Linear(4, name="edge")(jnp.concatenate([e, s, r], axis=-1))

    def node_fn(n, s, r, g):
      del s, g
      return hk.Linear(3, name="node")(jnp.concatenate([n, r["e"]], axis=-1))

    network = typed_graph_net.GraphNetwork({"e": edge_fn}, {"n": node_fn}, edge_chunk_size=chunk_size)
    # Node features are JAX arrays in the model (chunks index them with traced indices).
    graph = network(_graph(jnp.asarray(nodes), edges, senders, receivers, context=np.zeros((1, 0), np.float32)))
    return graph.edges[KEY].features, graph.nodes["n"].features

  whole = hk.transform(lambda nodes: run(0, nodes))
  chunked = hk.transform(lambda nodes: run(chunk_size, nodes))
  params = whole.init(jax.random.key(0), nodes)

  expected, actual = whole.apply(params, None, nodes), chunked.apply(params, None, nodes)
  jax.tree.map(lambda a, b: np.testing.assert_allclose(a, b, rtol=1e-5, atol=1e-6), actual, expected)
  grads = [jax.grad(lambda p, f=f: sum(x.sum() for x in f.apply(p, None, nodes)))(params) for f in (whole, chunked)]
  jax.tree.map(lambda a, b: np.testing.assert_allclose(a, b, rtol=1e-4, atol=1e-5), grads[1], grads[0])
