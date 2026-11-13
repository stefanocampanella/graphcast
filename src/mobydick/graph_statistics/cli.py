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
"""`compute` (all statistics of a model's graphs) and the command moby-dick graph-statistics."""

import contextlib
import json
import logging
import math
import pathlib
import time
from collections.abc import Generator, Sequence
from typing import Any

import click
import networkx as nx
import numpy as np
from rich.console import Console

from mobydick import cli_utils
from mobydick.dataloader_utils import Grid, get_grid
from mobydick.graph_statistics import geometry, regions, squashed, structure
from mobydick.graph_statistics.config import GraphStatisticsConfig, RegionsConfig, undefined_regions
from mobydick.graph_statistics.graphs import Graph, model_graphs
from mobydick.graph_statistics.quantities import described, histogram, summarize
from mobydick.graph_statistics.report import print_summary
from mobydick.mesh_graph import MeshData

logger = logging.getLogger(__name__)


def finite_or_null(value: Any) -> Any:
  """`value` with NaN and infinite floats replaced by None (null in JSON, which has no NaN or Infinity): e.g. the
  summary of no values, or the aspect ratio of a degenerate triangle."""
  if isinstance(value, dict):
    return {key: finite_or_null(item) for key, item in value.items()}
  if isinstance(value, list | tuple):
    return [finite_or_null(item) for item in value]
  if isinstance(value, float) and not math.isfinite(value):
    return None
  return value


@contextlib.contextmanager
def _timed(message: str) -> Generator[None]:
  """Logs `message` with the time the block took: the log has sizes and timings only, the results go to stdout and
  the JSON."""
  start = time.perf_counter()
  yield
  logger.info(f"{message} in {time.perf_counter() - start:.1f} s")


def read_settings(config_path: pathlib.Path, regions_path: pathlib.Path) -> tuple[GraphStatisticsConfig, RegionsConfig]:
  """The settings and the regions, checking that every basin connection names regions of the regions file."""
  config = cli_utils.read_configs(config_path, schema=GraphStatisticsConfig)
  regions = cli_utils.read_configs(regions_path, schema=RegionsConfig)
  if missing := undefined_regions(config, regions):
    raise click.ClickException(f"Basin connections of {config_path} refer to regions not in {regions_path}: {missing}")
  return config, regions


def graph_statistics(graph: Graph, config: GraphStatisticsConfig, rng: np.random.Generator) -> dict[str, Any]:
  """The statistics of one graph: those of its kind (`structure`, plus the triangle quality of a processor's mesh),
  and the geodesic edge lengths with their histogram."""
  stats: dict[str, Any] = {
    "kind": graph.kind,
    "senders": graph.sender_nodes.name,
    "receivers": graph.receiver_nodes.name,
  }
  if graph.kind == "encoder":
    stats.update(structure.encoder_statistics(graph))
  elif graph.kind == "decoder":
    stats.update(structure.decoder_statistics(graph))
  else:
    assert graph.mesh is not None
    stats.update(structure.processor_statistics(graph, config.num_sources, rng))
    stats["triangles"] = geometry.triangle_statistics(graph.mesh, config.quality_bins)
  lengths = geometry.edge_lengths_km(graph)
  # Bins of `length_bin_width_km` from 0 to past the longest edge.
  width = config.length_bin_width_km
  bins = np.arange(0.0, (np.floor(lengths.max(initial=0.0) / width) + 1) * width + width / 2, width)
  stats["edge_length_km"] = described(
    "Geodesic edge lengths on the WGS84 ellipsoid (km)", {**summarize(lengths), "histogram": histogram(lengths, bins)}
  )
  return stats


def compute(
  meshes: Sequence[MeshData], grid: Grid, config: GraphStatisticsConfig, regions_config: RegionsConfig | None = None
) -> dict[str, Any]:
  """The statistics of every graph (by name, in data-flow order), the coverage of every pair of levels and the
  squashed grid-to-grid graph, and the connectivity of the basins of `config.basin_connections` (regions from
  `regions_config`); `meshes` are those of `config.meshes`, read."""
  if len(meshes) != len(config.meshes):
    raise ValueError(f"{len(meshes)} meshes for {len(config.meshes)} entries of the settings")
  regions_config = regions_config or RegionsConfig()
  if missing := undefined_regions(config, regions_config):
    raise ValueError(f"Basin connections refer to undefined regions: {missing}")
  with _timed(f"Built the graphs of {len(meshes)} mesh(es)"):
    graphs = model_graphs(meshes, grid, config.radius_query_fraction_edge_length)
  # One generator for all components of all processors: the sampled sources depend only on the seed.
  rng = np.random.default_rng(config.seed)
  results: dict[str, Any] = {"graphs": {}, "coverage": []}
  for graph in graphs:
    with _timed(f"{graph.name}: statistics of the {graph.kind} ({len(graph.senders):,} edges)"):
      results["graphs"][graph.name] = graph_statistics(graph, config, rng)
  # Each encoder lower -> upper has a decoder upper -> lower.
  decoders = {(g.receiver_nodes.name, g.sender_nodes.name): g for g in graphs if g.kind == "decoder"}
  for encoder in (g for g in graphs if g.kind == "encoder"):
    decoder = decoders[(encoder.sender_nodes.name, encoder.receiver_nodes.name)]
    results["coverage"].append(structure.coverage(encoder, decoder))
  # A generator of its own: the sampled grid nodes do not depend on the processors' samples.
  squashed_rng = np.random.default_rng([config.seed, 1])
  grid_nodes = graphs[0].sender_nodes.size
  sampled = squashed_rng.choice(grid_nodes, min(config.squashed.num_sources, grid_nodes), replace=False)
  steps = [mesh.message_passing_steps for mesh in config.meshes]
  # The sparse matrices of every graph, built once for the squashed graph and the basins.
  operations = squashed.graph_operations(graphs)
  with _timed(f"Squashed grid-to-grid graph with {steps} steps, from {len(sampled):,} grid nodes"):
    results["squashed"] = squashed.squashed_statistics(
      graphs, steps, sampled, config.squashed.max_targets, config.squashed.batch_size, squashed_rng, operations
    )
  grid_nodes = graphs[0].sender_nodes
  polygons = regions_config.polygons()
  results["basin_connections"] = []
  for connection in config.basin_connections:
    basin, ocean = (regions.in_region(grid_nodes, polygons[name]) for name in (connection.basin, connection.ocean))
    with _timed(
      f"Connectivity of {connection.basin} and {connection.ocean} ({basin.sum():,} and {ocean.sum():,} nodes)"
    ):
      results["basin_connections"].append(
        squashed.basin_connectivity(
          graphs,
          steps,
          connection.basin,
          basin,
          connection.ocean,
          ocean,
          config.squashed.max_steps_to_connect,
          operations,
        )
      )
  return results


@click.command("graph-statistics")
@click.argument("config_path", type=click.Path(path_type=pathlib.Path, exists=True, dir_okay=False))
@click.argument("regions_path", type=click.Path(path_type=pathlib.Path, exists=True, dir_okay=False))
@click.argument("output_path", type=click.Path(path_type=pathlib.Path, dir_okay=False, writable=True))
@click.option("--overwrite/--no-overwrite", default=False, help="Whether to overwrite existing outputs", is_flag=True)
@cli_utils.LOG_LEVEL_OPTION
def graph_statistics_command(
  config_path: pathlib.Path,
  regions_path: pathlib.Path,
  output_path: pathlib.Path,
  overwrite: bool = False,
  log_level: str = "info",
):
  """Statistics of the graphs of the model described by CONFIG_PATH: its mask (written by moby-dick download-mask)
  and meshes (one, or a hierarchy, finest first) with their message passing steps, and the settings of the
  statistics. REGIONS_PATH holds the ocean regions that the basin connections of CONFIG_PATH name. Prints a summary
  and writes every value, with its description, to OUTPUT_PATH (JSON). The environment variable
  NETWORKX_BACKEND_PRIORITY selects a networkx backend for the path metrics (e.g. cugraph, with the graph-gpu
  extra)."""
  from mobydick.geospatial_mesh_utils import read_mesh_data

  cli_utils.configure_logging(log_level)
  config, regions_config = read_settings(config_path, regions_path)
  if output_path.exists() and not overwrite:
    raise click.ClickException(f"{output_path} exists: pass --overwrite to replace it")
  logger.info(f"Using settings: {config!r}, networkx backend priority {nx.config.backend_priority}")
  with _timed(f"Read {len(config.meshes)} mesh(es) and the mask"):
    meshes = [read_mesh_data(mesh.path) for mesh in config.meshes]
    grid = get_grid(config.mask)
  results = compute(meshes, grid, config, regions_config)
  # The JSON first: the results are kept even if printing fails.
  output_path.parent.mkdir(parents=True, exist_ok=True)
  output = {"settings": config.model_dump(mode="json"), "regions_file": str(regions_path)}
  output_path.write_text(json.dumps(finite_or_null({**output, **results}), indent=1, allow_nan=False))
  logger.info(f"Wrote {output_path}")
  print_summary(results, Console())
