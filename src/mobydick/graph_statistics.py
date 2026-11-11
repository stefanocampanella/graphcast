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
"""Statistics of the encoder (grid2mesh), processor (mesh) and decoder (mesh2grid) graphs of a mesh and a grid.

- Structure (networkx): nodes, edges, density, degrees, connectivity; diameter, average shortest path, clustering,
  betweenness and closeness centrality on the undirected graph (sampled on large graphs).
- Geometry: geodesic edge lengths on the WGS84 ellipsoid; quality of the mesh triangles (aspect ratio, radius ratio,
  volume-length metric, angles; see Shewchuk, "What is a good linear finite element?", 2002, and
  https://scicomp.stackexchange.com/a/27095).
- Coverage: mesh nodes connected to at least one unmasked grid point, unmasked grid points connected to at least one
  mesh node.

The edges are built as `leviathan.model.Leviathan` builds them (`mesh_connectivity`): radius query from the unmasked
grid points to the mesh vertices (radius a fraction of the mesh size) and containing triangles from the mesh vertices
to the unmasked grid points.
"""

import dataclasses
import json
import logging
import pathlib
from typing import Any

import click
import networkx as nx
import numpy as np
import pyproj
from pydantic import BaseModel, ConfigDict, Field, NonNegativeInt, PositiveFloat, PositiveInt

from mobydick import cli_utils, mesh_connectivity
from mobydick.dataloader_utils import Grid, get_grid
from mobydick.gis_utils import cartesian_crs, equirectangular_crs, get_transform
from mobydick.mesh_graph import MeshData

logger = logging.getLogger(__name__)


class GraphStatisticsConfig(BaseModel):
  """Settings of moby-dick graph-statistics."""

  model_config = ConfigDict(extra="forbid")

  radius_query_fraction_edge_length: PositiveFloat = Field(
    default=0.6, description="Encoder radius as a fraction of the mesh size, as model.radius_query_fraction_edge_length"
  )
  networkx: bool = Field(default=True, description="Compute the path, clustering and centrality metrics (slow)")
  num_sources: NonNegativeInt = Field(
    default=32, description="Sampled nodes for path, clustering and centrality metrics (0: all nodes, exact)"
  )
  seed: int = 0
  length_bin_width_km: PositiveFloat = Field(default=5.0, description="Width of the edge length histogram bins")
  quality_bins: PositiveInt = Field(default=20, description="Bins of the triangle quality histograms over [0, 1]")
  kdtree_workers: int = Field(default=1, description="Threads of the encoder's radius query (-1: all)")


@dataclasses.dataclass(frozen=True)
class BipartiteGraph:
  """Directed edges between two node sets (`senders` and `receivers` index into each set; a set may be both)."""

  senders: np.ndarray
  receivers: np.ndarray
  num_senders: int
  num_receivers: int
  sender_lat: np.ndarray
  sender_lon: np.ndarray
  receiver_lat: np.ndarray
  receiver_lon: np.ndarray
  same_nodes: bool = False


def summarize(values: np.ndarray) -> dict[str, float]:
  """Mean, median, min, max and standard deviation (NaN for no values)."""
  values = np.asarray(values, dtype=np.float64)
  if values.size == 0:
    return dict.fromkeys(("mean", "median", "min", "max", "std"), float("nan"))
  return {
    "mean": float(values.mean()),
    "median": float(np.median(values)),
    "min": float(values.min()),
    "max": float(values.max()),
    "std": float(values.std()),
  }


def histogram(values: np.ndarray, bins: np.ndarray) -> dict[str, list]:
  counts, edges = np.histogram(values, bins=bins)
  return {"bin_edges": edges.tolist(), "counts": counts.tolist()}


def geodesic_lengths(lat1: np.ndarray, lon1: np.ndarray, lat2: np.ndarray, lon2: np.ndarray) -> np.ndarray:
  """Geodesic distances on the WGS84 ellipsoid (metres) between points given in degrees."""
  _, _, distances = pyproj.Geod(ellps="WGS84").inv(lon1, lat1, lon2, lat2)
  return np.asarray(distances)


def model_graphs(mesh_data: MeshData, grid: Grid, radius_query_fraction: float, kdtree_workers: int = 1):
  """The encoder, processor and decoder edges of the model, with node positions in degrees.

  Grid node indices are the unmasked grid points (row-major over the flattened grid), as with
  `model.connected_grid_nodes`; returns (graphs by name, number of unmasked grid points).
  """
  mesh = mesh_data.mesh_graph
  if mesh_data.mesh_size is None:
    raise ValueError("the mesh has no mesh size: the encoder radius is a fraction of it")
  mask = np.asarray(grid.mask, dtype=bool)
  grid_senders, mesh_receivers = mesh_connectivity.radius_query_indices(
    grid_latitude=grid.lat,
    grid_longitude=grid.lon,
    mesh=mesh,
    radius=radius_query_fraction * mesh_data.mesh_size,
    mask=mask,
    workers=kdtree_workers,
  )
  mesh_senders, grid_receivers = mesh_connectivity.get_mesh_to_grid_edges(
    grid_latitude=grid.lat, grid_longitude=grid.lon, mesh=mesh, mask=mask
  )
  # Flattened grid point -> index among the unmasked points.
  flat_mask = mask.reshape(-1)
  point_to_node = np.cumsum(flat_mask) - 1
  num_points = int(flat_mask.sum())
  grid_lon, grid_lat = np.meshgrid(grid.lon, grid.lat)
  grid_lat, grid_lon = grid_lat.reshape(-1)[flat_mask], grid_lon.reshape(-1)[flat_mask]
  mesh_lon, mesh_lat = get_transform(cartesian_crs, equirectangular_crs)(mesh.vertices).T
  num_mesh = len(mesh.vertices)
  graphs = {
    "encoder": BipartiteGraph(
      point_to_node[grid_senders], mesh_receivers, num_points, num_mesh, grid_lat, grid_lon, mesh_lat, mesh_lon
    ),
    "processor": BipartiteGraph(
      np.asarray(mesh.edges[0]),
      np.asarray(mesh.edges[1]),
      num_mesh,
      num_mesh,
      mesh_lat,
      mesh_lon,
      mesh_lat,
      mesh_lon,
      same_nodes=True,
    ),
    "decoder": BipartiteGraph(
      mesh_senders, point_to_node[grid_receivers], num_mesh, num_points, mesh_lat, mesh_lon, grid_lat, grid_lon
    ),
  }
  return graphs, num_points


def to_networkx(graph: BipartiteGraph) -> nx.DiGraph:
  """A directed networkx graph; for two node sets, receivers are numbered after the senders."""
  result = nx.DiGraph()
  offset = 0 if graph.same_nodes else graph.num_senders
  result.add_nodes_from(range(graph.num_senders if graph.same_nodes else graph.num_senders + graph.num_receivers))
  result.add_edges_from(zip(graph.senders.tolist(), (graph.receivers + offset).tolist(), strict=True))
  return result


def degree_statistics(graph: BipartiteGraph) -> dict[str, Any]:
  """Out-degrees of the senders and in-degrees of the receivers (all nodes of their set, also those without edges)."""
  out_degree = np.bincount(graph.senders, minlength=graph.num_senders)
  in_degree = np.bincount(graph.receivers, minlength=graph.num_receivers)
  return {"sender_out_degree": summarize(out_degree), "receiver_in_degree": summarize(in_degree)}


def networkx_statistics(graph: nx.DiGraph, num_sources: int = 32, seed: int = 0) -> dict[str, Any]:
  """Structure of a directed graph; paths, clustering and centralities on its undirected version.

  Path metrics and centralities are computed on the largest connected component. With `num_sources` > 0, they use that
  many sampled nodes: the diameter is a lower bound (networkx's 2-sweep approximation), the average shortest path, the
  closeness and the clustering are averages over the sampled nodes, the betweenness is networkx's sampled estimate.
  """
  rng = np.random.default_rng(seed)
  undirected = graph.to_undirected(as_view=True)
  components = list(nx.connected_components(undirected))
  largest = undirected.subgraph(max(components, key=len)) if components else undirected
  num_nodes = largest.number_of_nodes()
  degrees = np.array([d for _, d in graph.degree()])
  exact = num_sources == 0 or num_sources >= num_nodes
  sources = list(largest.nodes) if exact else rng.choice(list(largest.nodes), num_sources, replace=False).tolist()
  if num_nodes > 1:
    distances = [nx.single_source_shortest_path_length(largest, s) for s in sources]
    lengths = np.concatenate([np.fromiter(d.values(), dtype=np.int64) for d in distances])
    average_path = float(lengths.sum() / (len(sources) * (num_nodes - 1)))
    closeness = [(num_nodes - 1) / sum(d.values()) for d in distances]
    diameter = nx.diameter(largest) if exact else nx.approximation.diameter(largest, seed=seed)
    betweenness = nx.betweenness_centrality(largest, k=None if exact else len(sources), seed=seed)
    average_betweenness = float(np.mean(list(betweenness.values())))
  else:
    average_path, closeness, diameter, average_betweenness = 0.0, [0.0], 0, 0.0
  return {
    "nodes": graph.number_of_nodes(),
    "edges": graph.number_of_edges(),
    "directed": graph.is_directed(),
    "density": nx.density(graph),
    "degree": summarize(degrees),
    "degree_histogram": nx.degree_histogram(graph),
    "connected": len(components) == 1,
    "strongly_connected": nx.is_strongly_connected(graph),
    "connected_components": len(components),
    "largest_component_nodes": num_nodes,
    "sampled_nodes": None if exact else len(sources),
    "diameter": int(diameter),
    "average_shortest_path": average_path,
    "average_clustering": float(nx.average_clustering(undirected, nodes=None if exact else sources)),
    "average_betweenness_centrality": average_betweenness,
    "average_closeness_centrality": float(np.mean(closeness)),
  }


def coverage(graphs: dict[str, BipartiteGraph], num_mesh: int, num_points: int) -> dict[str, Any]:
  """Mesh nodes connected to at least one unmasked grid point and unmasked grid points connected to a mesh node."""
  encoder, decoder = graphs["encoder"], graphs["decoder"]
  mesh_encoder = np.zeros(num_mesh, bool)
  mesh_encoder[encoder.receivers] = True
  mesh_decoder = np.zeros(num_mesh, bool)
  mesh_decoder[decoder.senders] = True
  grid_encoder = np.zeros(num_points, bool)
  grid_encoder[encoder.senders] = True
  grid_decoder = np.zeros(num_points, bool)
  grid_decoder[decoder.receivers] = True

  def count(selected: np.ndarray) -> dict[str, float]:
    return {"count": int(selected.sum()), "percent": float(100 * selected.mean()) if selected.size else float("nan")}

  return {
    "mesh_nodes": num_mesh,
    "grid_points": num_points,
    "mesh_nodes_in_encoder": count(mesh_encoder),
    "mesh_nodes_in_decoder": count(mesh_decoder),
    "mesh_nodes_in_either": count(mesh_encoder | mesh_decoder),
    "grid_points_in_encoder": count(grid_encoder),
    "grid_points_in_decoder": count(grid_decoder),
    "grid_points_in_both": count(grid_encoder & grid_decoder),
  }


def triangle_quality(vertices: np.ndarray, faces: np.ndarray) -> dict[str, np.ndarray]:
  """Quality of the flat triangles through the mesh vertices (Earth-centred coordinates), per face.

  - `aspect_ratio`: longest edge / shortest altitude, scaled to 1 for an equilateral triangle (>= 1).
  - `radius_ratio`: 2 × inradius / circumradius (1 equilateral, 0 degenerate).
  - `volume_length`: 4/√3 × area / (root-mean-square edge length)² (1 equilateral, 0 degenerate), negative for a
    triangle whose orientation is inverted, i.e. whose normal points to the centre of the Earth.
  - `min_angle`, `max_angle` in degrees, `edge_ratio` longest / shortest edge.
  """
  a, b, c = (vertices[faces[:, i]] for i in range(3))
  # Edge opposite each vertex.
  lengths = np.stack([np.linalg.norm(b - c, axis=1), np.linalg.norm(c - a, axis=1), np.linalg.norm(a - b, axis=1)], 1)
  normal = np.cross(b - a, c - a)
  area = 0.5 * np.linalg.norm(normal, axis=1)
  orientation = np.sign(np.einsum("ij,ij->i", normal, a + b + c))
  longest, shortest = lengths.max(1), lengths.min(1)
  perimeter = lengths.sum(1)
  inradius = 2 * area / perimeter
  circumradius = lengths.prod(1) / (4 * area)
  la, lb, lc = lengths.T
  cosines = np.stack(
    [
      (lb**2 + lc**2 - la**2) / (2 * lb * lc),
      (lc**2 + la**2 - lb**2) / (2 * lc * la),
      (la**2 + lb**2 - lc**2) / (2 * la * lb),
    ],
    1,
  )
  angles = np.degrees(np.arccos(np.clip(cosines, -1, 1)))
  return {
    "area_km2": area / 1e6,
    "aspect_ratio": np.sqrt(3) / 2 * longest / (2 * area / longest),
    "radius_ratio": 2 * inradius / circumradius,
    "volume_length": orientation * 4 / np.sqrt(3) * area / np.mean(lengths**2, axis=1),
    "min_angle": angles.min(1),
    "max_angle": angles.max(1),
    "edge_ratio": longest / shortest,
  }


def compute(mesh_data: MeshData, grid: Grid, config: GraphStatisticsConfig) -> dict[str, Any]:
  """All statistics of the three graphs, the coverage and the triangle quality."""
  graphs, num_points = model_graphs(mesh_data, grid, config.radius_query_fraction_edge_length, config.kdtree_workers)
  results: dict[str, Any] = {}
  for name, graph in graphs.items():
    lengths = (
      geodesic_lengths(
        graph.sender_lat[graph.senders],
        graph.sender_lon[graph.senders],
        graph.receiver_lat[graph.receivers],
        graph.receiver_lon[graph.receivers],
      )
      / 1e3
    )
    width = config.length_bin_width_km
    bins = np.arange(0.0, (np.floor(lengths.max(initial=0.0) / width) + 1) * width + width / 2, width)
    stats: dict[str, Any] = {"edges": len(graph.senders), **degree_statistics(graph)}
    stats["edge_length_km"] = {**summarize(lengths), "histogram": histogram(lengths, bins)}
    if config.networkx:
      logger.info(f"Computing networkx statistics of the {name} graph")
      stats["networkx"] = networkx_statistics(to_networkx(graph), config.num_sources, config.seed)
    results[name] = stats
  results["coverage"] = coverage(graphs, len(mesh_data.mesh_graph.vertices), num_points)
  quality = triangle_quality(
    np.asarray(mesh_data.mesh_graph.vertices, np.float64), np.asarray(mesh_data.mesh_graph.faces)
  )
  unit_bins = np.linspace(0.0, 1.0, config.quality_bins + 1)
  triangles: dict[str, Any] = {"faces": len(mesh_data.mesh_graph.faces)}
  results["triangles"] = triangles
  for name, values in quality.items():
    entry: dict[str, Any] = summarize(values)
    if name in ("radius_ratio", "volume_length"):
      entry["histogram"] = histogram(values, unit_bins)
    triangles[name] = entry
  triangles["volume_length"]["inverted"] = int((quality["volume_length"] < 0).sum())
  return results


def log_summary(results: dict[str, Any]) -> None:
  """The most relevant statistics, one line per graph."""
  for name in ("encoder", "processor", "decoder"):
    stats = results[name]
    length = stats["edge_length_km"]
    line = (
      f"{name}: {stats['edges']} edges, sender out-degree {stats['sender_out_degree']['mean']:.2f} "
      f"[{stats['sender_out_degree']['min']:.0f}, {stats['sender_out_degree']['max']:.0f}], receiver in-degree "
      f"{stats['receiver_in_degree']['mean']:.2f} [{stats['receiver_in_degree']['min']:.0f}, "
      f"{stats['receiver_in_degree']['max']:.0f}], edge length median {length['median']:.1f} km "
      f"[{length['min']:.1f}, {length['max']:.1f}]"
    )
    if "networkx" in stats:
      graph = stats["networkx"]
      line += (
        f", density {graph['density']:.3g}, {graph['connected_components']} components, diameter {graph['diameter']}, "
        f"average shortest path {graph['average_shortest_path']:.2f}, clustering {graph['average_clustering']:.3f}"
      )
    logger.info(line)
  c = results["coverage"]
  logger.info(
    f"coverage: mesh nodes connected to unmasked grid points {c['mesh_nodes_in_either']['percent']:.1f} % "
    f"(encoder {c['mesh_nodes_in_encoder']['percent']:.1f} %, decoder {c['mesh_nodes_in_decoder']['percent']:.1f} %), "
    f"unmasked grid points in the encoder {c['grid_points_in_encoder']['percent']:.2f} % "
    f"({c['grid_points'] - c['grid_points_in_encoder']['count']} missing), "
    f"in the decoder {c['grid_points_in_decoder']['percent']:.2f} %"
  )
  t = results["triangles"]
  logger.info(
    f"triangles: {t['faces']} faces, aspect ratio median {t['aspect_ratio']['median']:.3f} (max "
    f"{t['aspect_ratio']['max']:.2f}), radius ratio median {t['radius_ratio']['median']:.3f} (min "
    f"{t['radius_ratio']['min']:.3f}), volume-length median {t['volume_length']['median']:.3f} (min "
    f"{t['volume_length']['min']:.3f}, {t['volume_length']['inverted']} inverted), min angle "
    f"{t['min_angle']['min']:.1f}°"
  )


@click.command("graph-statistics")
@click.argument("config_path", type=click.Path(path_type=pathlib.Path, exists=True, dir_okay=False))
@click.argument("mesh_path", type=click.Path(path_type=pathlib.Path, exists=True, dir_okay=False))
@click.argument("mask_path", type=click.Path(path_type=pathlib.Path, exists=True))
@click.argument("output_path", type=click.Path(path_type=pathlib.Path, dir_okay=False, writable=True))
@click.option("--overwrite/--no-overwrite", default=False, help="Whether to overwrite existing outputs", is_flag=True)
@cli_utils.LOG_LEVEL_OPTION
def graph_statistics_command(
  config_path: pathlib.Path,
  mesh_path: pathlib.Path,
  mask_path: pathlib.Path,
  output_path: pathlib.Path,
  overwrite: bool = False,
  log_level: str = "info",
):
  """Statistics of the encoder, processor and decoder graphs of the mesh at MESH_PATH and the grid of the mask at
  MASK_PATH (written by moby-dick download-mask), with the settings of CONFIG_PATH, saved to OUTPUT_PATH (JSON)."""
  from mobydick.geospatial_mesh_utils import read_mesh_data

  cli_utils.configure_logging(log_level)
  config = cli_utils.read_configs(config_path, schema=GraphStatisticsConfig)
  if output_path.exists() and not overwrite:
    raise click.ClickException(f"{output_path} exists: pass --overwrite to replace it")
  logger.info(f"Using settings: {config!r}")
  results = compute(read_mesh_data(mesh_path), get_grid(mask_path), config)
  log_summary(results)
  output_path.parent.mkdir(parents=True, exist_ok=True)
  output_path.write_text(json.dumps({"mesh": str(mesh_path), "settings": config.model_dump(), **results}, indent=1))
