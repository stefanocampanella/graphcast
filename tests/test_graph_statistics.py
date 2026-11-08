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
"""Tests of graph_statistics: the model's graphs, networkx metrics, coverage, edge lengths, triangle quality, CLI."""

import dataclasses
import json

import haiku as hk
import networkx as nx
import numpy as np
import pytest
from click.testing import CliRunner

from mobydick import graph_statistics, icosahedral_mesh
from mobydick.dataloader_utils import Grid
from mobydick.leviathan.model import Leviathan


def _grid(ocean_below: float = 70.0) -> Grid:
  lat, lon = np.arange(-75.0, 90.0, 30.0), np.arange(0.0, 360.0, 45.0)
  return Grid(lat=lat, lon=lon, mask=np.broadcast_to(lat[:, None] < ocean_below, (lat.size, lon.size)).copy())


def _bipartite(senders, receivers, num_senders, num_receivers, same_nodes=False):
  zeros_s, zeros_r = np.zeros(num_senders), np.zeros(num_receivers)
  return graph_statistics.BipartiteGraph(
    np.array(senders), np.array(receivers), num_senders, num_receivers, zeros_s, zeros_s, zeros_r, zeros_r, same_nodes
  )


def test_graphs_equal_the_models(tiny_model):
  """The statistics describe the graphs the model trains on: the encoder and decoder edges equal those Leviathan builds
  with connected grid nodes (grid nodes are unmasked points), the processor edges are the mesh's."""
  grid = _grid()
  config = dataclasses.replace(tiny_model.model_config, connected_grid_nodes=True)
  captured = {}

  def build():
    model = Leviathan(
      _model_config=config,
      _task_config=tiny_model.task,
      _grid_lat=grid.lat,
      _grid_lon=grid.lon,
      _grid_mask=grid.mask,
      _mesh_data=tiny_model.mesh_data,
    )
    captured["encoder"] = model._get_grid2mesh_edges()
    captured["processor"] = model._get_mesh_edges()
    captured["decoder"] = model._get_mesh2grid_edges()

  hk.transform(build).init(None)
  graphs, num_points = graph_statistics.model_graphs(
    tiny_model.mesh_data, grid, config.radius_query_fraction_edge_length
  )
  assert num_points == grid.mask.sum()
  for name, (senders, receivers) in captured.items():
    np.testing.assert_array_equal(graphs[name].senders, senders, err_msg=name)
    np.testing.assert_array_equal(graphs[name].receivers, receivers, err_msg=name)


def test_networkx_statistics_of_a_cycle_are_exact():
  """On a directed cycle of 6 nodes in both directions every value is known: density 2·6/(6·5), degree 4 (in + out),
  diameter 3, average shortest path (1+1+2+2+3)/5 = 1.8, no triangles, betweenness 0.2·... equal for all nodes, closeness
  1/1.8."""
  edges = [(i, (i + 1) % 6) for i in range(6)]
  graph = nx.DiGraph(edges + [(b, a) for a, b in edges])
  stats = graph_statistics.networkx_statistics(graph, num_sources=0)
  assert stats["nodes"] == 6 and stats["edges"] == 12 and stats["directed"]
  assert stats["density"] == pytest.approx(12 / 30)
  assert stats["degree"]["mean"] == 4 and stats["degree"]["std"] == 0
  assert stats["degree_histogram"][4] == 6
  assert stats["connected"] and stats["strongly_connected"] and stats["sampled_nodes"] is None
  assert stats["diameter"] == 3
  assert stats["average_shortest_path"] == pytest.approx(1.8)
  assert stats["average_clustering"] == 0
  assert stats["average_closeness_centrality"] == pytest.approx(1 / 1.8)
  assert stats["average_betweenness_centrality"] == pytest.approx(nx.betweenness_centrality(nx.cycle_graph(6))[0])


def test_sampled_networkx_statistics_use_the_largest_component():
  """With isolated nodes (mesh nodes over land without edges) the graph is not connected: path metrics come from the
  largest component, and sampling gives the exact values on a vertex-transitive graph (a cycle)."""
  graph = nx.DiGraph(nx.cycle_graph(10))
  graph.add_nodes_from([10, 11])
  stats = graph_statistics.networkx_statistics(graph, num_sources=4)
  assert not stats["connected"] and stats["connected_components"] == 3 and stats["largest_component_nodes"] == 10
  assert stats["sampled_nodes"] == 4
  assert stats["diameter"] == 5
  exact = nx.average_shortest_path_length(nx.cycle_graph(10))
  assert stats["average_shortest_path"] == pytest.approx(exact)


def test_to_networkx_numbers_receivers_after_senders():
  """Two node sets become one graph without collisions: receiver i is node num_senders + i; one node set keeps its
  numbering."""
  graph = graph_statistics.to_networkx(_bipartite([0, 1, 1], [0, 0, 2], 2, 3))
  assert graph.number_of_nodes() == 5 and set(graph.edges) == {(0, 2), (1, 2), (1, 4)}
  same = graph_statistics.to_networkx(_bipartite([0, 1], [1, 0], 2, 2, same_nodes=True))
  assert same.number_of_nodes() == 2 and set(same.edges) == {(0, 1), (1, 0)}


def test_degree_statistics_count_nodes_without_edges():
  """Degrees include nodes without edges (zeros), so the mean is edges / nodes."""
  stats = graph_statistics.degree_statistics(_bipartite([0, 0, 1], [2, 2, 2], 4, 3))
  assert stats["sender_out_degree"]["mean"] == pytest.approx(3 / 4) and stats["sender_out_degree"]["min"] == 0
  assert stats["receiver_in_degree"]["max"] == 3 and stats["receiver_in_degree"]["median"] == 0


def test_coverage_counts_mesh_nodes_and_grid_points():
  """A mesh node counts as covered when it receives an encoder edge or sends a decoder edge; a grid point by encoder
  and decoder separately."""
  graphs = {
    "encoder": _bipartite([0, 1], [0, 0], 3, 4),  # grid -> mesh
    "decoder": _bipartite([1, 1, 1], [0, 1, 2], 4, 3),  # mesh -> grid
  }
  result = graph_statistics.coverage(graphs, num_mesh=4, num_points=3)
  assert result["mesh_nodes_in_encoder"] == {"count": 1, "percent": 25.0}
  assert result["mesh_nodes_in_decoder"]["count"] == 1 and result["mesh_nodes_in_either"]["count"] == 2
  assert result["grid_points_in_encoder"]["count"] == 2 and result["grid_points_in_decoder"]["percent"] == 100.0
  assert result["grid_points_in_both"]["count"] == 2


def test_geodesic_lengths_on_the_ellipsoid():
  """One degree of longitude on the equator is 111.32 km on WGS84, one degree of latitude there 110.57 km."""
  lengths = graph_statistics.geodesic_lengths(
    np.array([0.0, 0.0]), np.array([0.0, 0.0]), np.array([0.0, 1.0]), np.array([1.0, 0.0])
  )
  np.testing.assert_allclose(lengths, [111_319.5, 110_574.4], rtol=1e-5)


def test_triangle_quality_of_equilateral_and_degenerate_triangles():
  """An equilateral triangle scores 1 in every metric and has 60° angles; a sliver scores close to 0 (radius ratio,
  volume-length) or large (aspect ratio); reversing the orientation (normal towards the Earth's centre) makes the
  volume-length metric negative."""
  r = 6.4e6
  vertices = np.array([[r, 0, 0], [r, 1e3, 0], [r, 5e2, 5e2 * np.sqrt(3)], [r, 5e2, 1.0]])
  faces = np.array([[0, 1, 2], [0, 1, 3], [0, 2, 1]])
  quality = graph_statistics.triangle_quality(vertices, faces)
  for name in ("aspect_ratio", "radius_ratio", "volume_length", "edge_ratio"):
    assert quality[name][0] == pytest.approx(1.0), name
  assert quality["min_angle"][0] == pytest.approx(60.0) and quality["max_angle"][0] == pytest.approx(60.0)
  assert quality["area_km2"][0] == pytest.approx(np.sqrt(3) / 4)
  assert quality["radius_ratio"][1] < 0.01 and quality["volume_length"][1] < 0.01 and quality["aspect_ratio"][1] > 100
  assert quality["volume_length"][2] == pytest.approx(-1.0)


def test_icosahedral_triangles_are_nearly_equilateral():
  """The finest faces of the refined icosahedron on the ellipsoid are close to equilateral and none is inverted (all
  faces of a closed mesh are oriented outwards)."""
  mesh = icosahedral_mesh.get_multi_mesh_data(3).mesh_graph
  quality = graph_statistics.triangle_quality(np.asarray(mesh.vertices, np.float64), np.asarray(mesh.faces))
  assert quality["volume_length"].min() > 0.9
  assert quality["radius_ratio"].min() > 0.9
  assert quality["min_angle"].min() > 45


def test_cli_writes_the_statistics(mask_path, tmp_path, write_config):
  """moby-dick graph-statistics reads the mask and a mesh file, and writes all sections; the icosahedral mesh covers
  every unmasked point of the synthetic grid in the encoder and the decoder."""
  from mobydick.geospatial_mesh_utils import write_multi_mesh

  mesh_path = tmp_path / "mesh.msh"
  write_multi_mesh(icosahedral_mesh.get_multi_mesh_data(2), mesh_path)
  config = write_config({"num_sources": 8, "length_bin_width_km": 100.0})
  output = tmp_path / "stats.json"
  arguments = [str(config), str(mesh_path), mask_path, str(output), "--log-level", "warning"]
  result = CliRunner().invoke(graph_statistics.graph_statistics_command, arguments)
  assert result.exit_code == 0, result.output
  stats = json.loads(output.read_text())
  assert set(stats) >= {"settings", "encoder", "processor", "decoder", "coverage", "triangles"}
  assert stats["processor"]["networkx"]["nodes"] == 162 and stats["processor"]["networkx"]["connected"]
  assert stats["coverage"]["grid_points_in_encoder"]["percent"] == 100.0
  assert stats["coverage"]["grid_points_in_decoder"]["percent"] == 100.0
  assert sum(stats["processor"]["edge_length_km"]["histogram"]["counts"]) == stats["processor"]["edges"]
  assert stats["triangles"]["volume_length"]["inverted"] == 0
  # An existing output is kept.
  result = CliRunner().invoke(graph_statistics.graph_statistics_command, arguments)
  assert result.exit_code != 0 and "exists" in result.output


def test_unknown_settings_are_rejected():
  """A misspelt setting fails instead of being ignored."""
  with pytest.raises(ValueError, match="num_source"):
    graph_statistics.GraphStatisticsConfig.model_validate({"num_source": 3})


def test_compute_without_networkx_logs_a_summary(tiny_model, caplog):
  """The training log path (networkx off) computes every other section and logs one line per graph."""
  config = graph_statistics.GraphStatisticsConfig(networkx=False)
  results = graph_statistics.compute(tiny_model.mesh_data, _grid(), config)
  assert "networkx" not in results["processor"]
  with caplog.at_level("INFO", logger="mobydick.graph_statistics"):
    graph_statistics.log_summary(results)
  messages = [record.getMessage() for record in caplog.records]
  assert [m.split(":")[0] for m in messages] == ["encoder", "processor", "decoder", "coverage", "triangles"]
