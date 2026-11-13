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
"""Tests of graph_statistics: the model's graphs (one mesh or a hierarchy), the statistics of each kind of graph,
coverage, edge lengths, triangle quality, the CLI and its output."""

import dataclasses
import io
import json
import pathlib
from typing import Any

import haiku as hk
import jax
import jax.numpy as jnp
import networkx as nx
import numpy as np
import pytest
import xarray as xr
from click.testing import CliRunner
from rich.console import Console

from mobydick import icosahedral_mesh, xarray_jax
from mobydick.dataloader_utils import Grid
from mobydick.graph_statistics import cli, config, geometry, graphs, quantities, report, squashed, structure
from mobydick.leviathan.model import Leviathan, ModelConfig, TaskConfig
from mobydick.mesh_graph import longest_incident_edges


def _grid(ocean_below: float = 70.0) -> Grid:
  lat, lon = np.arange(-75.0, 90.0, 30.0), np.arange(0.0, 360.0, 45.0)
  return Grid(lat=lat, lon=lon, mask=np.broadcast_to(lat[:, None] < ocean_below, (lat.size, lon.size)).copy())


def _graph(senders, receivers, num_senders, num_receivers, kind="encoder"):
  # Nodes 1° apart along a meridian: distinct positions, so the processors' distances are defined.
  senders_nodes = graphs.NodeSet("a", np.arange(num_senders, dtype=float), np.zeros(num_senders))
  receiver_nodes = senders_nodes if kind == "processor" else graphs.NodeSet("b", *np.zeros((2, num_receivers)))
  return graphs.Graph(
    "g", kind, np.array(senders, dtype=int), np.array(receivers, dtype=int), senders_nodes, receiver_nodes
  )


def _quantities(entry: dict, path: str = "") -> list[str]:
  """Paths of the quantities of a graph's or coverage entry's statistics that have no description."""
  missing = []
  for key, value in entry.items():
    if key in ("kind", "senders", "receivers", "lower", "upper"):
      continue
    if key == "triangles":
      missing += _quantities(value, f"{path}{key}.")
    elif not isinstance(value, dict) or "description" not in value:
      missing.append(f"{path}{key}")
    elif key == "components":
      for i, component in enumerate(value["value"]):
        missing += _quantities(component, f"{path}{key}[{i}].")
  return missing


def _reach_matrix(built: list[graphs.Graph], steps: list[int]) -> np.ndarray:
  """The squashed graph's dense boolean adjacency (receivers × senders) for one mesh or two, from the formula of the
  V-cycle written out by hand (numpy's product of boolean matrices is OR of ANDs):
  I + D0 [(I + P0)^L0 + D1 (I + P1)^L1 E1 (I + P0)^L0] E0."""
  dense = {g.name: squashed.adjacency(g).toarray() for g in built}

  def steps_on(name, count):
    identity = np.eye(dense[name].shape[0], dtype=bool)
    power = identity
    for _ in range(count):
      power = power @ (identity | dense[name])
    return power

  e0, d0 = dense["grid_to_mesh_0"], dense["mesh_0_to_grid"]
  inner = steps_on("mesh_0", steps[0])
  if "mesh_1" in dense:
    e1, d1 = dense["mesh_0_to_mesh_1"], dense["mesh_1_to_mesh_0"]
    inner = inner | d1 @ steps_on("mesh_1", steps[1]) @ e1 @ steps_on("mesh_0", steps[0])
  return np.eye(e0.shape[1], dtype=bool) | d0 @ inner @ e0


def _config(steps: list[int], **settings) -> config.GraphStatisticsConfig:
  """Settings for `cli.compute` on meshes already read (the paths are not opened)."""
  meshes = [{"path": f"/mesh_{level}.msh", "message_passing_steps": s} for level, s in enumerate(steps)]
  return config.GraphStatisticsConfig.model_validate({"mask": "/mask", "meshes": meshes, **settings})


def test_graphs_equal_the_models(tiny_model):
  """The statistics describe the graphs the model trains on: the encoder and decoder edges equal those Leviathan builds
  with connected grid nodes (grid nodes are unmasked points), the processor edges are the mesh's."""
  grid = _grid()
  model_config = dataclasses.replace(tiny_model.model_config, connected_grid_nodes=True)
  captured = {}

  def build():
    model = Leviathan(
      _model_config=model_config,
      _task_config=tiny_model.task,
      _grid_lat=grid.lat,
      _grid_lon=grid.lon,
      _grid_mask=grid.mask,
      _mesh_data=tiny_model.mesh_data,
    )
    captured["grid_to_mesh_0"] = model._get_grid2mesh_edges()
    captured["mesh_0"] = model._get_mesh_edges()
    captured["mesh_0_to_grid"] = model._get_mesh2grid_edges()

  hk.transform(build).init(None)
  built = graphs.model_graphs([tiny_model.mesh_data], grid, model_config.radius_query_fraction_edge_length)
  assert [(g.name, g.kind) for g in built] == [
    ("grid_to_mesh_0", "encoder"),
    ("mesh_0", "processor"),
    ("mesh_0_to_grid", "decoder"),
  ]
  assert built[0].sender_nodes.size == grid.mask.sum()
  for graph in built:
    senders, receivers = captured[graph.name]
    np.testing.assert_array_equal(graph.senders, senders, err_msg=graph.name)
    np.testing.assert_array_equal(graph.receivers, receivers, err_msg=graph.name)


def test_hierarchy_has_mesh_to_mesh_encoders_and_decoders():
  """With N meshes there are N encoders, N processors and N decoders, in the order of the data flow; the mesh-to-mesh
  encoder connects the finer vertices within the radius of each coarser vertex (0.6 × its longest triangle edge, brute
  force) and the decoder sends 3 edges to every finer vertex."""
  meshes = [icosahedral_mesh.get_multi_mesh_data(splits) for splits in (2, 1)]
  built = graphs.model_graphs(meshes, _grid(), 0.6)
  assert [g.name for g in built] == [
    "grid_to_mesh_0",
    "mesh_0_to_mesh_1",
    "mesh_0",
    "mesh_1",
    "mesh_1_to_mesh_0",
    "mesh_0_to_grid",
  ]
  encoder, decoder = built[1], built[4]
  fine, coarse = (m.mesh_graph.vertices for m in meshes)
  coarse_mesh = meshes[1].mesh_graph
  radius = 0.6 * longest_incident_edges(coarse_mesh.vertices, coarse_mesh.faces)
  within = np.linalg.norm(fine[:, None] - coarse[None], axis=-1) <= radius[None]
  assert set(zip(encoder.senders.tolist(), encoder.receivers.tolist(), strict=True)) == set(zip(*np.nonzero(within)))
  assert decoder.sender_nodes.name == "mesh_1" and decoder.receiver_nodes.size == len(fine)
  np.testing.assert_array_equal(np.bincount(decoder.receivers), 3)
  results = cli.compute(meshes, _grid(), _config([2, 1], num_sources=4, squashed={"num_sources": 8}))
  assert [(c["lower"], c["upper"]) for c in results["coverage"]] == [("grid", "mesh_0"), ("mesh_0", "mesh_1")]
  assert results["coverage"][1]["lower_in_both"]["percent"] == 100.0


def test_summarize_gives_mean_std_and_quartiles():
  """Summaries have the mean, standard deviation, minimum, quartiles and maximum; no values give NaN."""
  summary = quantities.summarize(np.arange(9.0))
  assert list(summary) == ["mean", "std", "min", "p25", "p50", "p75", "max"]
  assert [summary[k] for k in ("min", "p25", "p50", "p75", "max")] == [0, 2, 4, 6, 8]
  assert summary["std"] == pytest.approx(np.sqrt(60 / 9))
  assert np.isnan(quantities.summarize(np.array([]))["mean"])


def test_exact_path_statistics_match_networkx():
  """With every node as a source the values are exact: on a cycle of 6 nodes the diameter is 3 and the mean shortest
  path length equals networkx's average_shortest_path_length (1.8)."""
  graph = nx.cycle_graph(6)
  nodes = list(range(6))
  stats = structure.path_statistics(graph, nodes, structure.sample_sources(nodes, 0, np.random.default_rng(0)))
  assert stats["diameter"]["value"] == 3 and stats["diameter_exact"]["value"] and stats["sources"]["value"] == 6
  path = stats["shortest_path_length"]
  assert path["mean"] == pytest.approx(nx.average_shortest_path_length(graph)) == pytest.approx(1.8)
  assert path["min"] == 1 and path["max"] == 3


def test_sampled_diameter_is_a_two_sweep_lower_bound():
  """From one source at the end of a path graph's middle, the farthest node is an end, whose eccentricity is the
  diameter: the 2-sweep finds it although the source's own eccentricity is smaller."""
  graph = nx.path_graph(11)
  stats = structure.path_statistics(graph, list(range(11)), [5])
  assert stats["diameter"]["value"] == 10 and not stats["diameter_exact"]["value"]
  assert stats["shortest_path_length"]["max"] == 5


def test_processor_statistics_per_component():
  """Every component of more than one node gets its own path metrics (largest first); nodes without edges count as
  isolated; one-way edges (the boundary of an open mesh) are counted."""
  # A triangle (0-1-2) and a path (3-4-5-6), edges in both directions, and an isolated node 7.
  undirected = [(0, 1), (1, 2), (2, 0), (3, 4), (4, 5), (5, 6)]
  senders, receivers = zip(*(undirected + [(b, a) for a, b in undirected]), strict=True)
  graph = _graph(senders, receivers, 8, 8, kind="processor")
  stats = structure.processor_statistics(graph, 0, np.random.default_rng(0))
  assert stats["connected_components"]["value"] == 3 and stats["isolated_nodes"]["value"] == 1
  assert stats["edges_without_reverse"]["value"] == 0 and stats["density"]["value"] == pytest.approx(12 / 56)
  components = stats["components"]["value"]
  assert [c["nodes"]["value"] for c in components] == [4, 3]
  assert [c["diameter"]["value"] for c in components] == [3, 1]
  # Degree 0: node 7; 1: the path's ends; 2: the triangle and the path's middle.
  assert stats["out_degree"]["nodes_by_degree"] == [1, 2, 5]
  assert stats["duplicate_edges"]["value"] == 0
  duplicated = _graph((*senders, 0), (*receivers, 1), 8, 8, kind="processor")
  assert graph_statistics_duplicates(duplicated) == 1
  one_way = _graph(senders[:-1], receivers[:-1], 8, 8, kind="processor")
  assert structure.processor_statistics(one_way, 0, np.random.default_rng(0))["edges_without_reverse"]["value"] == 1


def graph_statistics_duplicates(graph) -> int:
  return structure.processor_statistics(graph, 0, np.random.default_rng(0))["duplicate_edges"]["value"]


def test_summary_tables_handle_graphs_without_paths_or_receivers():
  """A processor whose nodes are all isolated (no component to measure paths on) and an encoder with no receiver
  nodes print "-" instead of failing after the statistics were computed."""
  processor = _graph([], [], 3, 3, kind="processor")
  encoder = _graph([], [], 4, 0)
  processor_stats = structure.processor_statistics(processor, 0, np.random.default_rng(0))
  assert processor_stats["components"]["value"] == []
  stats: dict[str, Any] = {
    "g": {"kind": "processor", **processor_stats},
    "e": {"kind": "encoder", **structure.encoder_statistics(encoder)},
  }
  buffer = io.StringIO()
  console = Console(file=buffer, width=200)
  console.print(report.graphs_table(stats))
  console.print(report.processors_table(stats))
  assert "-" in buffer.getvalue()


def test_json_output_has_no_nan_or_infinity():
  """JSON has no NaN or Infinity: they are written as null (e.g. the summary of no values, the aspect ratio of a
  degenerate triangle), so strict parsers read the output."""
  value = {"a": float("nan"), "b": [1.0, float("inf"), {"c": -float("inf")}], "d": 2}
  assert cli.finite_or_null(value) == {"a": None, "b": [1.0, None, {"c": None}], "d": 2}


def test_volume_length_histogram_counts_inverted_faces():
  """The volume-length histogram spans [-1, 1], so inverted faces (negative values) are counted: its counts add up to
  the faces."""
  mesh = icosahedral_mesh.get_multi_mesh_data(1).mesh_graph
  faces = np.asarray(mesh.faces).copy()
  faces[:3] = faces[:3, ::-1]  # Three inverted faces.
  flipped = dataclasses.replace(mesh, faces=faces)
  triangles = geometry.triangle_statistics(flipped, 20)
  assert triangles["inverted"]["value"] == 3
  assert sum(triangles["volume_length"]["histogram"]["counts"]) == len(faces)


def test_encoder_degrees_count_nodes_without_edges():
  """Degrees include nodes without edges (zeros), so the mean is edges / nodes; the histogram counts nodes by degree."""
  stats = structure.encoder_statistics(_graph([0, 0, 1], [2, 2, 2], 4, 3))
  out_degree, in_degree = stats["sender_out_degree"], stats["receiver_in_degree"]
  assert out_degree["mean"] == pytest.approx(3 / 4) and out_degree["min"] == 0
  assert out_degree["nodes_by_degree"] == [2, 1, 1]
  assert in_degree["max"] == 3 and in_degree["p50"] == 0


def test_decoder_checks_three_edges_per_receiver():
  """A decoder sends exactly 3 edges to each receiver (the vertices of its face); a receiver with fewer is flagged."""
  stats = structure.decoder_statistics(_graph([0, 1, 2, 1, 2, 3], [0, 0, 0, 1, 1, 1], 4, 2, kind="decoder"))
  assert stats["expected_edges"]["value"] == stats["edges"]["value"] == 6
  assert stats["three_edges_per_receiver"]["value"]
  stats = structure.decoder_statistics(_graph([0, 1, 2, 1, 2], [0, 0, 0, 1, 1], 4, 2, kind="decoder"))
  assert not stats["three_edges_per_receiver"]["value"]


def test_coverage_counts_nodes_of_both_levels():
  """A coarser node counts as covered when it receives an encoder edge or sends a decoder edge; a finer node by encoder
  and decoder separately, and in both."""
  encoder = _graph([0, 1], [0, 0], 3, 4)  # lower (3 nodes) -> upper (4 nodes)
  decoder = graphs.Graph(
    "d", "decoder", np.array([1, 1, 1]), np.array([0, 1, 2]), encoder.receiver_nodes, encoder.sender_nodes
  )
  result = structure.coverage(encoder, decoder)
  assert result["upper_in_encoder"]["count"] == 1 and result["upper_in_encoder"]["percent"] == 25.0
  assert result["upper_in_decoder"]["count"] == 1 and result["upper_in_either"]["count"] == 2
  assert result["lower_in_encoder"]["count"] == 2 and result["lower_in_decoder"]["percent"] == 100.0
  assert result["lower_in_both"]["count"] == 2


def test_geodesic_lengths_on_the_ellipsoid():
  """One degree of longitude on the equator is 111.32 km on WGS84, one degree of latitude there 110.57 km."""
  lengths = geometry.geodesic_lengths(
    np.array([0.0, 0.0]), np.array([0.0, 0.0]), np.array([0.0, 1.0]), np.array([1.0, 0.0])
  )
  np.testing.assert_allclose(lengths, [111_319.5, 110_574.4], rtol=1e-5)


def test_triangle_quality_of_equilateral_and_degenerate_triangles():
  """An equilateral triangle scores 1 in every metric and has 60° angles; a sliver scores close to 0 (radius ratio,
  volume-length) or large (aspect ratio); reversing the orientation (normal towards the Earth's centre) makes the
  volume-length metric negative. `angle` has the three angles of every face."""
  r = 6.4e6
  vertices = np.array([[r, 0, 0], [r, 1e3, 0], [r, 5e2, 5e2 * np.sqrt(3)], [r, 5e2, 1.0]])
  faces = np.array([[0, 1, 2], [0, 1, 3], [0, 2, 1]])
  quality = geometry.triangle_quality(vertices, faces)
  assert set(quality) == set(geometry.TRIANGLE_QUALITY)
  for name in ("aspect_ratio", "radius_ratio", "volume_length", "edge_ratio"):
    assert quality[name][0] == pytest.approx(1.0), name
  assert quality["min_angle"][0] == pytest.approx(60.0) and quality["max_angle"][0] == pytest.approx(60.0)
  assert quality["angle"].shape == (9,)
  np.testing.assert_allclose(quality["angle"].reshape(3, 3).sum(1), 180.0)
  assert quality["area_km2"][0] == pytest.approx(np.sqrt(3) / 4)
  assert quality["radius_ratio"][1] < 0.01 and quality["volume_length"][1] < 0.01 and quality["aspect_ratio"][1] > 100
  assert quality["volume_length"][2] == pytest.approx(-1.0)


def test_icosahedral_triangles_are_nearly_equilateral():
  """The finest faces of the refined icosahedron on the ellipsoid are close to equilateral and none is inverted (all
  faces of a closed mesh are oriented outwards)."""
  mesh = icosahedral_mesh.get_multi_mesh_data(3).mesh_graph
  quality = geometry.triangle_quality(np.asarray(mesh.vertices, np.float64), np.asarray(mesh.faces))
  assert quality["volume_length"].min() > 0.9
  assert quality["radius_ratio"].min() > 0.9
  assert quality["min_angle"].min() > 45


def test_cli_prints_tables_and_writes_described_statistics(mask_path, tmp_path, write_config):
  """moby-dick graph-statistics reads the mask and a mesh file, prints the summary tables to stdout and writes every
  quantity with a description; the icosahedral mesh covers every unmasked point of the synthetic grid in the encoder
  and the decoder. An existing output is kept."""
  from mobydick.geospatial_mesh_utils import write_multi_mesh

  mesh_path = tmp_path / "mesh.msh"
  write_multi_mesh(icosahedral_mesh.get_multi_mesh_data(2), mesh_path)
  config_path = write_config(
    {
      "mask": mask_path,
      "meshes": [{"path": str(mesh_path), "message_passing_steps": 1}],
      "num_sources": 8,
      "length_bin_width_km": 100.0,
      "squashed": {"num_sources": 16},
    }
  )
  output = tmp_path / "stats.json"
  arguments = [str(config_path), str(output), "--log-level", "warning"]
  result = CliRunner().invoke(cli.graph_statistics_command, arguments)
  assert result.exit_code == 0, result.output
  titles = ["Graphs", "Edge lengths", "Degrees", "Processors", "Distances", "Coverage", "Squashed"]
  for title in [*titles, "Triangle quality of mesh_0"]:
    assert title in result.output

  def reject(constant):
    raise ValueError(f"{constant} is not JSON")

  stats = json.loads(output.read_text(), parse_constant=reject)
  assert list(stats["graphs"]) == ["grid_to_mesh_0", "mesh_0", "mesh_0_to_grid"]
  entries = [*stats["graphs"].items(), *(("coverage", c) for c in stats["coverage"])]
  for name, entry in [*entries, ("squashed", stats["squashed"])]:
    assert _quantities(entry) == [], name
  processor = stats["graphs"]["mesh_0"]
  assert processor["nodes"]["value"] == 162 and processor["connected_components"]["value"] == 1
  assert stats["graphs"]["mesh_0_to_grid"]["three_edges_per_receiver"]["value"]
  assert stats["coverage"][0]["lower_in_encoder"]["percent"] == 100.0
  assert stats["coverage"][0]["lower_in_decoder"]["percent"] == 100.0
  assert sum(processor["edge_length_km"]["histogram"]["counts"]) == processor["edges"]["value"]
  assert processor["triangles"]["inverted"]["value"] == 0
  assert stats["squashed"]["message_passing_steps"]["value"] == [1] and stats["squashed"]["out_degree"]["max"] > 0
  assert stats["settings"]["meshes"][0]["path"] == str(mesh_path)
  result = CliRunner().invoke(cli.graph_statistics_command, arguments)
  assert result.exit_code != 0 and "exists" in result.output


@pytest.mark.parametrize(
  ("splits", "steps"), [((1,), [0]), ((1,), [1]), ((1,), [2]), ((2, 1), [1, 2]), ((2, 1), [2, 0]), ((2, 1), [0, 1])]
)
def test_squashed_reach_equals_the_matrix_product(splits, steps):
  """Propagating boolean indicator vectors through the operations gives the columns (forward) and rows (backward) of
  the squashed graph's adjacency, computed independently as a product of dense matrices, for one mesh and a V-cycle
  on two meshes with different steps: every grid node reaches itself, and the backward pass is the transpose of the
  forward one."""
  built = graphs.model_graphs([icosahedral_mesh.get_multi_mesh_data(s) for s in splits], _grid(), 0.6)
  expected = _reach_matrix(built, steps)
  operations, sizes = squashed.schedule(built, steps), squashed.node_sizes(built)
  nodes = np.arange(expected.shape[0])
  np.testing.assert_array_equal(squashed.reach(operations, sizes, "grid", nodes), expected.T)
  np.testing.assert_array_equal(squashed.reach(operations, sizes, "grid", nodes, backward=True), expected)


def test_squashed_statistics_are_exact_with_every_node_sampled():
  """With every grid node sampled and every edge measured, the degrees are the squashed adjacency's row and column
  sums without the self-loops, the edge estimate is exact, and the horizon is the mean geodesic distance over the
  pairs it links; batches of any size give the same values."""
  grid = _grid()
  built = graphs.model_graphs([icosahedral_mesh.get_multi_mesh_data(1)], grid, 0.6)
  adjacency = _reach_matrix(built, [1]) & ~np.eye(built[0].sender_nodes.size, dtype=bool)
  nodes = np.arange(adjacency.shape[0])
  stats = squashed.squashed_statistics(built, [1], nodes, 10_000, 5, np.random.default_rng(0))
  assert stats["edges"]["value"] == pytest.approx(adjacency.sum())
  assert stats["out_degree"]["mean"] == pytest.approx(adjacency.sum(axis=0).mean())
  assert stats["in_degree"]["max"] == adjacency.sum(axis=1).max()
  receivers, senders = np.nonzero(adjacency)
  lat, lon = built[0].sender_nodes.lat, built[0].sender_nodes.lon
  lengths = geometry.geodesic_lengths(lat[senders], lon[senders], lat[receivers], lon[receivers]) / 1000
  assert stats["horizon_km"]["mean"] == pytest.approx(lengths.mean())
  assert stats["horizon_km"]["max"] == pytest.approx(lengths.max())
  other_batch = squashed.squashed_statistics(built, [1], nodes, 10_000, 64, np.random.default_rng(0))
  assert other_batch["horizon_km"] == stats["horizon_km"]
  # Per receiver: the mean distance to its senders (a column of the adjacency).
  receiver_means = [
    geometry.geodesic_lengths(np.full(len(s), lat[r]), np.full(len(s), lon[r]), lat[s], lon[s]).mean() / 1000
    for r, s in ((r, np.flatnonzero(adjacency[r])) for r in nodes)
    if len(s)
  ]
  assert stats["receiver_mean_distance_km"]["mean"] == pytest.approx(np.mean(receiver_means))


def test_message_passing_steps_reach_every_node_within_that_many_hops():
  """L steps reach the nodes within at most L hops, not only those exactly L hops away: each step ORs the
  neighbours into the state, so (I + P)^L = I + P + ... + P^L. On a path graph 0-1-2-3-4, two steps from node 0 reach
  nodes 0, 1 and 2, and backward from node 4 nodes 2, 3 and 4."""
  undirected = [(0, 1), (1, 2), (2, 3), (3, 4)]
  senders, receivers = zip(*(undirected + [(b, a) for a, b in undirected]), strict=True)
  path = _graph(senders, receivers, 5, 5, kind="processor")
  operations = [squashed._operation(path)] * 2
  reached = squashed.reach(operations, {"a": 5}, "a", np.array([0, 4]))
  np.testing.assert_array_equal(reached, [[1, 1, 1, 0, 0], [0, 0, 1, 1, 1]])
  np.testing.assert_array_equal(
    squashed.reach(operations, {"a": 5}, "a", np.array([4]), backward=True), [[0, 0, 1, 1, 1]]
  )


@pytest.mark.parametrize("steps", [1, 2])
def test_reach_matches_the_models_dependencies(steps):
  """`reach` describes the dependencies of a Leviathan model built with moby-dick's code (random weights, 5° grid
  with land, icosahedral mesh with 3 refinements, so that a grid point reaches about 2–30 % of the others):
  - perturbing the input at a grid point changes the output only inside its domain of influence (forward reach);
  - the forward-mode derivative of the outputs with respect to that input is nonzero exactly on that domain (a finite
    perturbation leaves some points of the domain unchanged: their derivative, below 1e-6, is lost in float32
    rounding of outputs of order 1);
  - the gradient of the output at a grid point with respect to the inputs is nonzero exactly on its receptive field
    (backward reach).
  Both directions are thus exact: `reach` misses no dependency and adds none."""
  lat, lon = np.arange(-87.5, 90.0, 5.0), np.arange(0.0, 360.0, 5.0)
  mask = np.broadcast_to(lat[:, None] < 70.0, (lat.size, lon.size)).copy()
  mask[10:14, 20:30] = False  # A continent: grid points without nodes, and coasts.
  mesh = icosahedral_mesh.get_multi_mesh_data(3)
  model_config = ModelConfig(
    latent_size=4,
    gnn_msg_steps=steps,
    hidden_layers=1,
    radius_query_fraction_edge_length=0.6,
    connected_grid_nodes=True,
  )
  task = TaskConfig(
    input_variables=("zos",), target_variables=("zos",), forcing_variables=("2t",), levels=(0,), input_duration="1D"
  )
  rng = np.random.default_rng(0)

  def field(day: int) -> xr.DataArray:
    return xr.DataArray(
      rng.normal(size=(1, 1, lat.size, lon.size)).astype(np.float32),
      dims=("batch", "time", "lat", "lon"),
      coords={"time": np.array([day], dtype="timedelta64[D]"), "lat": lat, "lon": lon},
    )

  inputs, targets, forcings = xr.Dataset({"zos": field(0)}), xr.Dataset({"zos": field(1)}), xr.Dataset({"2t": field(1)})

  def model(i, t, f):
    return Leviathan(
      _model_config=model_config,
      _task_config=task,
      _grid_lat=lat,
      _grid_lon=lon,
      _grid_mask=mask,
      _mesh_data=mesh,
    )(i, t, f)

  transformed = hk.transform(model)
  params = transformed.init(jax.random.key(0), inputs, targets, forcings)

  @jax.jit
  def predict(zos: jax.Array) -> jax.Array:
    """The predicted (lat, lon) field from the input field (batch and time of size 1)."""
    data = xarray_jax.Dataset({"zos": xarray_jax.DataArray(zos, dims=inputs.zos.dims, coords=inputs.zos.coords)})
    return xarray_jax.unwrap_data(transformed.apply(params, None, data, targets, forcings)["zos"])[0, 0]

  def point_output(zos: jax.Array, weights: jax.Array) -> jax.Array:
    return jnp.sum(predict(zos) * weights)

  gradient = jax.jit(jax.grad(point_output))
  zos = jnp.asarray(inputs.zos.values)
  unperturbed = np.asarray(predict(zos))

  built = graphs.model_graphs([mesh], Grid(lat=lat, lon=lon, mask=mask), model_config.radius_query_fraction_edge_length)
  operations, sizes = squashed.schedule(built, [steps]), squashed.node_sizes(built)
  points = np.flatnonzero(mask.reshape(-1))  # Grid node -> flattened grid point.
  nodes = np.array([0, len(points) // 3, len(points) // 2, len(points) - 1])
  domains = squashed.reach(operations, sizes, "grid", nodes)
  receptive_fields = squashed.reach(operations, sizes, "grid", nodes, backward=True)

  def on_grid(reached: np.ndarray) -> np.ndarray:
    grid = np.zeros(mask.shape, dtype=bool)
    grid.reshape(-1)[points[reached]] = True
    return grid

  for node, domain, receptive_field in zip(nodes, map(on_grid, domains), map(on_grid, receptive_fields), strict=True):
    i, j = np.unravel_index(points[node], mask.shape)
    # Non-trivial sets: more than the point itself, far from the whole ocean.
    assert 1 < domain.sum() < mask.sum() // 2 and 1 < receptive_field.sum() < mask.sum() // 2
    changed = np.asarray(predict(zos.at[0, 0, i, j].add(1.0))) != unperturbed
    assert changed[i, j] and not np.any(changed & ~domain), node
    _, derivative = jax.jvp(predict, (zos,), (jnp.zeros_like(zos).at[0, 0, i, j].set(1.0),))
    np.testing.assert_array_equal(np.asarray(derivative) != 0, domain, err_msg=f"node {node}")
    weights = jnp.zeros(mask.shape, jnp.float32).at[i, j].set(1.0)
    dependencies = np.asarray(gradient(zos, weights))[0, 0] != 0
    np.testing.assert_array_equal(dependencies, receptive_field, err_msg=f"node {node}")


def test_hierarchy_keeps_the_paths_of_the_finer_mesh():
  """On the way up the V-cycle each decoder ORs the coarser mesh's result into the finer mesh's state (skip
  connections, as in a U-Net), so the squashed graph of a hierarchy contains that of its finest mesh alone with the
  same steps: the paths that never leave mesh 0 still count."""
  grid = _grid()
  meshes = [icosahedral_mesh.get_multi_mesh_data(s) for s in (2, 1)]
  hierarchy = graphs.model_graphs(meshes, grid, 0.6)
  single = graphs.model_graphs(meshes[:1], grid, 0.6)
  nodes = np.arange(hierarchy[0].sender_nodes.size)
  with_coarse = squashed.reach(squashed.schedule(hierarchy, [1, 1]), squashed.node_sizes(hierarchy), "grid", nodes)
  alone = squashed.reach(squashed.schedule(single, [1]), squashed.node_sizes(single), "grid", nodes)
  assert np.all(with_coarse | ~alone) and with_coarse.sum() > alone.sum()


def test_sampled_horizon_weights_senders_by_their_out_degree():
  """When only a few edges per sender are measured, each counts for out-degree / measured edges, so the horizon stays
  an estimate of the mean over all edges (not over senders): close to the exact value on a fine icosahedral mesh."""
  built = graphs.model_graphs([icosahedral_mesh.get_multi_mesh_data(2)], _grid(), 0.6)
  nodes = np.arange(built[0].sender_nodes.size)
  exact = squashed.squashed_statistics(built, [2], nodes, 10_000, 64, np.random.default_rng(0))
  sampled = squashed.squashed_statistics(built, [2], nodes, 8, 64, np.random.default_rng(0))
  assert sampled["measured_edges"]["value"] < exact["measured_edges"]["value"]
  assert sampled["horizon_km"]["mean"] == pytest.approx(exact["horizon_km"]["mean"], rel=0.05)


def test_weighted_summary_counts_each_value_by_its_weight():
  """A value of weight 3 counts as three copies: the weighted summary equals the summary of the repeated values."""
  weighted = quantities.weighted_summarize(np.array([1.0, 2.0, 5.0]), np.array([1.0, 3.0, 1.0]))
  repeated = quantities.summarize(np.array([1.0, 2.0, 2.0, 2.0, 5.0]))
  for key in ("mean", "std", "min", "p50", "max"):
    assert weighted[key] == pytest.approx(repeated[key])
  assert np.isnan(quantities.weighted_summarize(np.array([]), np.array([]))["mean"])


def test_distances_compare_paths_along_the_mesh_with_geodesics():
  """Along a chain of nodes on the equator (a geodesic) the path along the mesh equals the geodesic distance (detour
  1); a node moved off the equator lengthens the path through it, so the detour exceeds 1."""

  def chain(lat):
    nodes = graphs.NodeSet("mesh_0", np.array(lat, dtype=float), np.array([0.0, 10.0, 20.0]))
    return graphs.Graph("mesh_0", "processor", np.array([0, 1, 1, 2]), np.array([1, 0, 2, 1]), nodes, nodes)

  straight = structure.distance_statistics(chain([0, 0, 0]), [0, 1, 2], [0, 1, 2])
  assert straight["detour"]["max"] == pytest.approx(1.0)
  assert straight["geodesic_distance_km"]["max"] == pytest.approx(2 * 1113.19, rel=1e-4)
  bent = structure.distance_statistics(chain([0, 10, 0]), [0, 1, 2], [0])
  assert bent["detour"]["max"] > 1.1
  assert bent["path_length_km"]["max"] > bent["geodesic_distance_km"]["max"]


def test_unknown_settings_are_rejected():
  """A misspelt setting fails instead of being ignored."""
  with pytest.raises(ValueError, match="num_source"):
    _config([1], num_source=3)


def test_input_paths_must_be_absolute():
  """The mask and mesh paths are absolute (container paths, as in the training configs): a relative path would depend
  on the directory the command runs from."""
  with pytest.raises(ValueError, match="not an absolute path"):
    config.GraphStatisticsConfig.model_validate(
      {"mask": "mask", "meshes": [{"path": "/m.msh", "message_passing_steps": 1}]}
    )
  with pytest.raises(ValueError, match="not an absolute path"):
    config.GraphStatisticsConfig.model_validate(
      {"mask": "/mask", "meshes": [{"path": "m.msh", "message_passing_steps": 1}]}
    )


def test_shipped_settings_are_valid():
  """configs/analysis/graph_statistics.toml validates: its top-level keys come before the [[meshes]] tables (keys after
  a table header belong to the table)."""
  from mobydick import cli_utils

  path = pathlib.Path(__file__).parents[1] / "configs" / "analysis" / "graph_statistics.toml"
  settings = cli_utils.read_configs(path, schema=config.GraphStatisticsConfig)
  assert settings.meshes[0].message_passing_steps == 16 and settings.radius_query_fraction_edge_length == 0.6
