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
"""Statistics of the graphs of a model on a grid and a hierarchy of meshes (mesh 0 the finest).

The graphs, built with `mesh_connectivity` as `leviathan.model.Leviathan` builds them:
- encoders: grid → mesh 0 from the unmasked grid points, mesh i−1 → mesh i from the vertices of mesh i−1, to the
  vertices of the receiving mesh within a radius (a fraction of its mesh size);
- processors: the edges of each mesh;
- decoders: mesh i → mesh i−1 and mesh 0 → grid, from the vertices of the face containing each receiver.
Models use one mesh today; the mesh-to-mesh graphs follow the grid's rules until hierarchical models define theirs.

Statistics:
- every graph: nodes, edges, geodesic edge lengths on the WGS84 ellipsoid;
- encoders: degrees of senders and receivers; decoders: whether every receiver has 3 edges;
- processors: degrees, density, one-way edges, connected components and, per component, the diameter and the distribution
  of shortest path lengths (networkx, whose backends, e.g. cugraph, can compute them); triangle quality (aspect ratio,
  radius ratio, volume-length, angles, edge ratio: Shewchuk, "What is a good linear finite element?", 2002, and
  https://scicomp.stackexchange.com/a/27095);
- coverage of each pair of levels: nodes of either level connected to the other by the encoder, the decoder or both.
"""

import contextlib
import dataclasses
import json
import logging
import pathlib
import time
from collections.abc import Generator, Sequence
from typing import Any, Literal

import click
import networkx as nx
import numpy as np
import pyproj
from pydantic import BaseModel, ConfigDict, Field, NonNegativeInt, PositiveFloat, PositiveInt
from rich.console import Console
from rich.table import Table

from mobydick import cli_utils, mesh_connectivity
from mobydick.dataloader_utils import Grid, get_grid
from mobydick.gis_utils import cartesian_crs, equirectangular_crs, get_transform
from mobydick.mesh_graph import MeshData, MeshGraph

logger = logging.getLogger(__name__)

SUMMARY_KEYS = ("mean", "std", "min", "p25", "p50", "p75", "max")
GraphKind = Literal["encoder", "processor", "decoder"]


class GraphStatisticsConfig(BaseModel):
  """Settings of moby-dick graph-statistics."""

  model_config = ConfigDict(extra="forbid")

  radius_query_fraction_edge_length: PositiveFloat = Field(
    default=0.6, description="Encoder radius as a fraction of the mesh size, as model.radius_query_fraction_edge_length"
  )
  num_sources: NonNegativeInt = Field(
    default=32,
    description="Nodes per connected component from which the shortest paths are computed (0: all nodes, exact, "
    "one breadth-first search per node)",
  )
  seed: int = Field(default=0, description="Seed of the sampled nodes")
  length_bin_width_km: PositiveFloat = Field(default=5.0, description="Width of the edge length histogram bins")
  quality_bins: PositiveInt = Field(
    default=20, description="Bins of the triangle quality histograms (over [0, 1], angles over [0°, 180°])"
  )


@dataclasses.dataclass(frozen=True)
class NodeSet:
  """Nodes of a grid (its unmasked points) or of a mesh, with their positions in degrees."""

  name: str
  lat: np.ndarray
  lon: np.ndarray

  @property
  def size(self) -> int:
    return len(self.lat)


@dataclasses.dataclass(frozen=True)
class Graph:
  """Directed edges from `sender_nodes` to `receiver_nodes` (`senders` and `receivers` index into them); processors
  have the same sender and receiver nodes and their mesh."""

  name: str
  kind: GraphKind
  senders: np.ndarray
  receivers: np.ndarray
  sender_nodes: NodeSet
  receiver_nodes: NodeSet
  mesh: MeshGraph | None = None


def described(description: str, value: Any) -> dict[str, Any]:
  """A quantity of the output, with a description: the fields of a dict, or any other value under "value"."""
  if isinstance(value, dict):
    return {"description": description, **value}
  return {"description": description, "value": value}


def summarize(values: np.ndarray) -> dict[str, float]:
  """Mean, standard deviation, minimum, quartiles (25th, 50th and 75th percentiles) and maximum (NaN for no values)."""
  values = np.asarray(values, dtype=np.float64)
  if values.size == 0:
    return dict.fromkeys(SUMMARY_KEYS, float("nan"))
  p25, p50, p75 = np.percentile(values, [25, 50, 75])
  return {
    "mean": float(values.mean()),
    "std": float(values.std()),
    "min": float(values.min()),
    "p25": float(p25),
    "p50": float(p50),
    "p75": float(p75),
    "max": float(values.max()),
  }


def histogram(values: np.ndarray, bins: np.ndarray) -> dict[str, list]:
  counts, edges = np.histogram(values, bins=bins)
  return {"bin_edges": edges.tolist(), "counts": counts.tolist()}


def degree_summary(degrees: np.ndarray) -> dict[str, Any]:
  """Summary of integer degrees, with the number of nodes of each degree (index = degree)."""
  return {**summarize(degrees), "nodes_by_degree": np.bincount(degrees).tolist()}


def fraction(selected: np.ndarray) -> dict[str, float]:
  """Number and percentage of the selected nodes."""
  return {"count": int(selected.sum()), "percent": float(100 * selected.mean()) if selected.size else float("nan")}


def geodesic_lengths(lat1: np.ndarray, lon1: np.ndarray, lat2: np.ndarray, lon2: np.ndarray) -> np.ndarray:
  """Geodesic distances on the WGS84 ellipsoid (metres) between points given in degrees."""
  _, _, distances = pyproj.Geod(ellps="WGS84").inv(lon1, lat1, lon2, lat2)
  return np.asarray(distances)


def model_graphs(meshes: Sequence[MeshData], grid: Grid, radius_query_fraction: float) -> list[Graph]:
  """The graphs of a model on `grid` and `meshes` (finest first): encoders (finest first), processors (finest first)
  and decoders (coarsest first).

  Grid nodes are the unmasked grid points (row-major over the flattened grid), as with `model.connected_grid_nodes`.
  """
  mask = np.asarray(grid.mask, dtype=bool)
  flat_mask = mask.reshape(-1)
  point_to_node = np.cumsum(flat_mask) - 1
  grid_lon, grid_lat = np.meshgrid(grid.lon, grid.lat)
  lower = NodeSet("grid", grid_lat.reshape(-1)[flat_mask], grid_lon.reshape(-1)[flat_mask])
  lower_mesh: MeshGraph | None = None
  to_lat_lon = get_transform(cartesian_crs, equirectangular_crs)
  encoders, processors, decoders = [], [], []
  for level, mesh_data in enumerate(meshes):
    mesh = mesh_data.mesh_graph
    if mesh_data.mesh_size is None:
      raise ValueError(f"mesh {level} has no mesh size: the encoder radius is a fraction of it")
    radius = radius_query_fraction * mesh_data.mesh_size
    lon, lat = to_lat_lon(np.asarray(mesh.vertices)).T
    nodes = NodeSet(f"mesh_{level}", lat, lon)
    if lower_mesh is None:
      grid_senders, encoder_receivers = mesh_connectivity.radius_query_indices(
        grid_latitude=grid.lat, grid_longitude=grid.lon, mesh=mesh, radius=radius, mask=mask
      )
      encoder_senders = point_to_node[grid_senders]
      decoder_senders, grid_receivers = mesh_connectivity.get_mesh_to_grid_edges(
        grid_latitude=grid.lat, grid_longitude=grid.lon, mesh=mesh, mask=mask
      )
      decoder_receivers = point_to_node[grid_receivers]
    else:
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


def edge_lengths_km(graph: Graph) -> np.ndarray:
  senders, receivers = graph.sender_nodes, graph.receiver_nodes
  return (
    geodesic_lengths(
      senders.lat[graph.senders],
      senders.lon[graph.senders],
      receivers.lat[graph.receivers],
      receivers.lon[graph.receivers],
    )
    / 1e3
  )


def node_counts(graph: Graph) -> dict[str, Any]:
  return {
    "sender_nodes": described(f"Nodes of the sender set ({graph.sender_nodes.name})", graph.sender_nodes.size),
    "receiver_nodes": described(f"Nodes of the receiver set ({graph.receiver_nodes.name})", graph.receiver_nodes.size),
    "edges": described("Directed edges", len(graph.senders)),
  }


def encoder_statistics(graph: Graph) -> dict[str, Any]:
  """Node and edge counts and the degrees of the senders and receivers."""
  out_degree = np.bincount(graph.senders, minlength=graph.sender_nodes.size)
  in_degree = np.bincount(graph.receivers, minlength=graph.receiver_nodes.size)
  return {
    **node_counts(graph),
    "sender_out_degree": described("Edges per sender node, also nodes without edges", degree_summary(out_degree)),
    "receiver_in_degree": described("Edges per receiver node, also nodes without edges", degree_summary(in_degree)),
  }


def decoder_statistics(graph: Graph) -> dict[str, Any]:
  """Node and edge counts, and whether every receiver has the 3 edges from the vertices of its face."""
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

  Breadth-first searches run from all nodes (exact) or from `num_sources` sampled ones. When sampling, the diameter is a
  lower bound: the largest eccentricity of the sources and of the node farthest from them (a 2-sweep), and the
  shortest path lengths are those from the sources.
  """
  exact = num_sources == 0 or num_sources >= len(nodes)
  sources = nodes if exact else rng.choice(nodes, num_sources, replace=False).tolist()
  lengths, diameter, farthest = [], 0, sources[0]
  for source in sources:
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
  """Structure of a mesh graph: degrees, density, one-way edges, connected components and their path metrics."""
  num_nodes = graph.sender_nodes.size
  num_edges = len(graph.senders)
  degree = np.bincount(graph.senders, minlength=num_nodes)
  forward = graph.senders.astype(np.int64) * num_nodes + graph.receivers
  backward = graph.receivers.astype(np.int64) * num_nodes + graph.senders
  undirected = nx.Graph()
  undirected.add_nodes_from(range(num_nodes))
  undirected.add_edges_from(zip(graph.senders.tolist(), graph.receivers.tolist(), strict=True))
  components = sorted((sorted(c) for c in nx.connected_components(undirected)), key=len, reverse=True)
  return {
    "nodes": described("Mesh vertices", num_nodes),
    "edges": described("Directed edges", num_edges),
    "density": described(
      "Directed edges / (nodes × (nodes − 1))", num_edges / (num_nodes * (num_nodes - 1)) if num_nodes > 1 else 0.0
    ),
    "edges_without_reverse": described(
      "Directed edges whose reverse is missing: the boundary edges of an open mesh, whose faces give each edge in one "
      "direction (mesh_graph.faces_to_edges); paths are those of the undirected graph",
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


TRIANGLE_QUALITY = {
  "area_km2": "Area of the flat triangle through the vertices (km²)",
  "aspect_ratio": "Longest edge / smallest height (the height on the longest edge), times √3/2: 1 for an equilateral "
  "triangle, larger for worse ones",
  "radius_ratio": "2 × inradius / circumradius: 1 for an equilateral triangle, 0 for a degenerate one",
  "volume_length": "4/√3 × area / mean squared edge length: 1 for an equilateral triangle, 0 for a degenerate one, "
  "negative when the normal points to the Earth's centre (inverted)",
  "angle": "Interior angles of all faces (degrees)",
  "min_angle": "Smallest angle of each face (degrees)",
  "max_angle": "Largest angle of each face (degrees)",
  "edge_ratio": "Longest / shortest edge of each face",
}


def triangle_quality(vertices: np.ndarray, faces: np.ndarray) -> dict[str, np.ndarray]:
  """The quantities of `TRIANGLE_QUALITY` for the flat triangles through the mesh vertices (Earth-centred
  coordinates): one value per face, `angle` three per face."""
  a, b, c = (vertices[faces[:, i]] for i in range(3))
  # Edge opposite each vertex.
  lengths = np.stack([np.linalg.norm(b - c, axis=1), np.linalg.norm(c - a, axis=1), np.linalg.norm(a - b, axis=1)], 1)
  normal = np.cross(b - a, c - a)
  area = 0.5 * np.linalg.norm(normal, axis=1)
  orientation = np.sign(np.einsum("ij,ij->i", normal, a + b + c))
  longest, shortest = lengths.max(1), lengths.min(1)
  inradius = 2 * area / lengths.sum(1)
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
    "angle": angles.reshape(-1),
    "min_angle": angles.min(1),
    "max_angle": angles.max(1),
    "edge_ratio": longest / shortest,
  }


def triangle_statistics(mesh: MeshGraph, bins: int) -> dict[str, Any]:
  """Summaries of the triangle quality, with histograms of the bounded quantities."""
  quality = triangle_quality(np.asarray(mesh.vertices, np.float64), np.asarray(mesh.faces))
  histogram_bins = {
    "radius_ratio": np.linspace(0.0, 1.0, bins + 1),
    "volume_length": np.linspace(0.0, 1.0, bins + 1),
    "angle": np.linspace(0.0, 180.0, bins + 1),
  }
  result = {
    "faces": described("Triangles of the mesh", len(mesh.faces)),
    "inverted": described("Faces whose normal points to the Earth's centre", int((quality["volume_length"] < 0).sum())),
  }
  for name, values in quality.items():
    entry: dict[str, Any] = summarize(values)
    if name in histogram_bins:
      entry["histogram"] = histogram(values, histogram_bins[name])
    result[name] = described(TRIANGLE_QUALITY[name], entry)
  return result


def coverage(encoder: Graph, decoder: Graph) -> dict[str, Any]:
  """Nodes of a level and of the next coarser one connected to each other by the encoder, the decoder or both."""
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


@contextlib.contextmanager
def _timed(message: str) -> Generator[None]:
  start = time.perf_counter()
  yield
  logger.info(f"{message} in {time.perf_counter() - start:.1f} s")


def compute(meshes: Sequence[MeshData], grid: Grid, config: GraphStatisticsConfig) -> dict[str, Any]:
  """The statistics of every graph and the coverage of every pair of levels."""
  with _timed(f"Built the graphs of {len(meshes)} mesh(es)"):
    graphs = model_graphs(meshes, grid, config.radius_query_fraction_edge_length)
  rng = np.random.default_rng(config.seed)
  results: dict[str, Any] = {"graphs": {}, "coverage": []}
  for graph in graphs:
    with _timed(f"{graph.name}: statistics of the {graph.kind} ({len(graph.senders):,} edges)"):
      stats: dict[str, Any] = {"kind": graph.kind, "senders": graph.sender_nodes.name}
      stats["receivers"] = graph.receiver_nodes.name
      if graph.kind == "encoder":
        stats.update(encoder_statistics(graph))
      elif graph.kind == "decoder":
        stats.update(decoder_statistics(graph))
      else:
        assert graph.mesh is not None
        stats.update(processor_statistics(graph, config.num_sources, rng))
        stats["triangles"] = triangle_statistics(graph.mesh, config.quality_bins)
      lengths = edge_lengths_km(graph)
      width = config.length_bin_width_km
      bins = np.arange(0.0, (np.floor(lengths.max(initial=0.0) / width) + 1) * width + width / 2, width)
      stats["edge_length_km"] = described(
        "Geodesic edge lengths on the WGS84 ellipsoid (km)",
        {**summarize(lengths), "histogram": histogram(lengths, bins)},
      )
      results["graphs"][graph.name] = stats
  encoders = [g for g in graphs if g.kind == "encoder"]
  decoders = {(g.receiver_nodes.name, g.sender_nodes.name): g for g in graphs if g.kind == "decoder"}
  for encoder in encoders:
    results["coverage"].append(coverage(encoder, decoders[(encoder.sender_nodes.name, encoder.receiver_nodes.name)]))
  return results


def _format(value: Any) -> str:
  if isinstance(value, bool):
    return "yes" if value else "no"
  if isinstance(value, int):
    return f"{value:,}"
  if isinstance(value, float):
    return f"{value:.4g}"
  return str(value)


def _summary_table(title: str, rows: list[tuple[str, dict[str, Any]]]) -> Table:
  table = Table(title=title, title_justify="left")
  table.add_column("")
  for key in SUMMARY_KEYS:
    table.add_column(key, justify="right")
  for label, summary in rows:
    table.add_row(label, *(_format(summary[key]) for key in SUMMARY_KEYS))
  return table


def print_summary(results: dict[str, Any], console: Console) -> None:
  """Tables of the main statistics."""
  graphs = results["graphs"]
  table = Table(title="Graphs", title_justify="left")
  for column in ("graph", "kind", "senders", "receivers", "edges", "edges per receiver"):
    table.add_column(column, justify="left" if column in ("graph", "kind") else "right")
  for name, stats in graphs.items():
    nodes = (stats["nodes"]["value"],) * 2 if stats["kind"] == "processor" else None
    senders, receivers = nodes or (stats["sender_nodes"]["value"], stats["receiver_nodes"]["value"])
    edges = stats["edges"]["value"]
    table.add_row(name, stats["kind"], _format(senders), _format(receivers), _format(edges), f"{edges / receivers:.2f}")
  console.print(table)
  console.print(
    _summary_table("Edge lengths (km)", [(name, stats["edge_length_km"]) for name, stats in graphs.items()])
  )
  degrees = []
  for name, stats in graphs.items():
    for key in ("sender_out_degree", "receiver_in_degree", "out_degree"):
      if key in stats:
        degrees.append((f"{name} {key.replace('_', ' ')}", stats[key]))
  console.print(_summary_table("Degrees", degrees))
  table = Table(title="Processors (largest component)", title_justify="left")
  columns = ("graph", "density", "one-way edges", "components", "isolated", "nodes", "diameter", "sources")
  for column in (*columns, "path mean", "path p50", "path max"):
    table.add_column(column, justify="left" if column == "graph" else "right")
  for name, stats in graphs.items():
    if stats["kind"] != "processor":
      continue
    largest = stats["components"]["value"][0]
    path = largest["shortest_path_length"]
    diameter = f"{largest['diameter']['value']}{'' if largest['diameter_exact']['value'] else ' (≥)'}"
    table.add_row(
      name,
      *(
        _format(stats[key]["value"])
        for key in ("density", "edges_without_reverse", "connected_components", "isolated_nodes")
      ),
      _format(largest["nodes"]["value"]),
      diameter,
      _format(largest["sources"]["value"]),
      *(_format(path[key]) for key in ("mean", "p50", "max")),
    )
  console.print(table)
  table = Table(title="Coverage (nodes connected to the other level; without: their number)", title_justify="left")
  keys = ("lower_in_encoder", "lower_in_decoder", "lower_in_both", "upper_in_encoder", "upper_in_decoder")
  for column in ("lower", "upper", *keys, "upper_in_either"):
    table.add_column(column.replace("_", " "), justify="left" if column in ("lower", "upper") else "right")
  for entry in results["coverage"]:
    sizes = {"lower": graphs[f"{entry['lower']}_to_{entry['upper']}"]["sender_nodes"]["value"]}
    sizes["upper"] = graphs[f"{entry['lower']}_to_{entry['upper']}"]["receiver_nodes"]["value"]
    cells = []
    for key in (*keys, "upper_in_either"):
      size = sizes[key.split("_")[0]]
      cells.append(f"{entry[key]['percent']:.2f} % ({size - entry[key]['count']:,} without)")
    table.add_row(entry["lower"], entry["upper"], *cells)
  console.print(table)
  for name, stats in graphs.items():
    if stats["kind"] == "processor":
      triangles = stats["triangles"]
      title = (
        f"Triangle quality of {name} ({_format(triangles['faces']['value'])} faces, "
        f"{_format(triangles['inverted']['value'])} inverted)"
      )
      console.print(_summary_table(title, [(key, triangles[key]) for key in TRIANGLE_QUALITY]))


@click.command("graph-statistics")
@click.argument("config_path", type=click.Path(path_type=pathlib.Path, exists=True, dir_okay=False))
@click.argument("mask_path", type=click.Path(path_type=pathlib.Path, exists=True))
@click.argument("output_path", type=click.Path(path_type=pathlib.Path, dir_okay=False, writable=True))
@click.argument(
  "mesh_paths", nargs=-1, required=True, type=click.Path(path_type=pathlib.Path, exists=True, dir_okay=False)
)
@click.option("--overwrite/--no-overwrite", default=False, help="Whether to overwrite existing outputs", is_flag=True)
@click.option(
  "--networkx-backend",
  default=None,
  help="networkx backend for the path metrics, e.g. cugraph (the nx-cugraph package); algorithms it lacks run in "
  "networkx. Default: networkx's own setting (environment variable NETWORKX_BACKEND_PRIORITY), else networkx.",
)
@cli_utils.LOG_LEVEL_OPTION
def graph_statistics_command(
  config_path: pathlib.Path,
  mask_path: pathlib.Path,
  output_path: pathlib.Path,
  mesh_paths: tuple[pathlib.Path, ...],
  overwrite: bool = False,
  networkx_backend: str | None = None,
  log_level: str = "info",
):
  """Statistics of the graphs of a model on the grid of MASK_PATH (written by moby-dick download-mask) and the meshes
  MESH_PATHS (one, or a hierarchy, finest first), with the settings of CONFIG_PATH. Prints a summary and writes every
  value, with its description, to OUTPUT_PATH (JSON)."""
  from mobydick.geospatial_mesh_utils import read_mesh_data

  cli_utils.configure_logging(log_level)
  config = cli_utils.read_configs(config_path, schema=GraphStatisticsConfig)
  if output_path.exists() and not overwrite:
    raise click.ClickException(f"{output_path} exists: pass --overwrite to replace it")
  if networkx_backend is not None:
    try:
      nx.config.backend_priority = [networkx_backend]
    except ValueError as error:
      raise click.ClickException(f"networkx backend {networkx_backend!r}: {error}") from error
  logger.info(f"Using settings: {config!r}, networkx backend priority {nx.config.backend_priority}")
  with _timed(f"Read {len(mesh_paths)} mesh(es) and the mask"):
    meshes = [read_mesh_data(path) for path in mesh_paths]
    grid = get_grid(mask_path)
  results = compute(meshes, grid, config)
  print_summary(results, Console())
  output_path.parent.mkdir(parents=True, exist_ok=True)
  output = {
    "meshes": [str(path) for path in mesh_paths],
    "mask": str(mask_path),
    "settings": config.model_dump(),
    **results,
  }
  output_path.write_text(json.dumps(output, indent=1))
  logger.info(f"Wrote {output_path}")
