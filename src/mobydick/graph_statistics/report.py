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
"""Summary tables of the statistics, printed to stdout (rich).

Each function reads the results of `cli.compute` (the JSON content) and builds one table; the JSON holds every value.
"""

from typing import Any

from rich.console import Console
from rich.table import Table

from mobydick.graph_statistics.geometry import TRIANGLE_QUALITY
from mobydick.graph_statistics.quantities import SUMMARY_KEYS


def _format(value: Any) -> str:
  """Integers with thousands separators, floats with 4 significant digits, missing values as "-"."""
  if value is None:
    return "-"
  if isinstance(value, bool):
    return "yes" if value else "no"
  if isinstance(value, int):
    return f"{value:,}"
  if isinstance(value, float):
    return f"{value:.4g}"
  return str(value)


def _table(title: str, columns: list[str], left: int = 1) -> Table:
  """A table whose first `left` columns are labels (left-aligned) and the others numbers (right-aligned)."""
  table = Table(title=title, title_justify="left")
  for i, column in enumerate(columns):
    table.add_column(column, justify="left" if i < left else "right")
  return table


def summary_table(title: str, rows: list[tuple[str, dict[str, Any]]]) -> Table:
  """One row per summary (`quantities.summarize`): label, mean, std, min, quartiles, max."""
  table = _table(title, ["", *SUMMARY_KEYS])
  for label, summary in rows:
    table.add_row(label, *(_format(summary[key]) for key in SUMMARY_KEYS))
  return table


def graphs_table(graphs: dict[str, Any]) -> Table:
  """Node and edge counts of every graph; edges per receiver is 3 for decoders and the mean in-degree otherwise."""
  table = _table("Graphs", ["graph", "kind", "senders", "receivers", "edges", "edges per receiver"], left=2)
  for name, stats in graphs.items():
    if stats["kind"] == "processor":
      senders = receivers = stats["nodes"]["value"]
    else:
      senders, receivers = stats["sender_nodes"]["value"], stats["receiver_nodes"]["value"]
    edges = stats["edges"]["value"]
    per_receiver = f"{edges / receivers:.2f}" if receivers else "-"
    table.add_row(name, stats["kind"], _format(senders), _format(receivers), _format(edges), per_receiver)
  return table


def degrees_table(graphs: dict[str, Any]) -> Table:
  """Degree summaries of the encoders (senders and receivers) and processors."""
  rows = []
  for name, stats in graphs.items():
    for key in ("sender_out_degree", "receiver_in_degree", "out_degree"):
      if key in stats:
        rows.append((f"{name} {key.replace('_', ' ')}", stats[key]))
  return summary_table("Degrees", rows)


def processors_table(graphs: dict[str, Any]) -> Table:
  """Structure of each processor, with the path metrics of its largest component ("≥": a lower bound)."""
  columns = ["graph", "density", "one-way edges", "components", "isolated", "nodes", "diameter", "sources"]
  table = _table("Processors (largest component)", [*columns, "path mean", "path p50", "path max"])
  for name, stats in graphs.items():
    if stats["kind"] != "processor":
      continue
    structure = [
      _format(stats[key]["value"])
      for key in ("density", "edges_without_reverse", "connected_components", "isolated_nodes")
    ]
    if not stats["components"]["value"]:  # No component of more than one node: no paths.
      table.add_row(name, *structure, *("-",) * 6)
      continue
    largest = stats["components"]["value"][0]
    path = largest["shortest_path_length"]
    bound = "" if largest["diameter_exact"]["value"] else " (≥)"
    table.add_row(
      name,
      *structure,
      _format(largest["nodes"]["value"]),
      f"{largest['diameter']['value']}{bound}",
      _format(largest["sources"]["value"]),
      *(_format(path[key]) for key in ("mean", "p50", "max")),
    )
  return table


def coverage_table(coverage: list[dict[str, Any]], graphs: dict[str, Any]) -> Table:
  """Percentages of the nodes of each level connected to the other level, with the number of nodes left out."""
  keys = ["lower_in_encoder", "lower_in_decoder", "lower_in_both", "upper_in_encoder", "upper_in_decoder"]
  keys.append("upper_in_either")
  title = "Coverage (nodes connected to the other level; without: their number)"
  table = _table(title, ["lower", "upper", *(key.replace("_", " ") for key in keys)], left=2)
  for entry in coverage:
    # The sizes of the two levels are those of their encoder's node sets.
    encoder = graphs[f"{entry['lower']}_to_{entry['upper']}"]
    sizes = {"lower": encoder["sender_nodes"]["value"], "upper": encoder["receiver_nodes"]["value"]}
    cells = []
    for key in keys:
      size = sizes[key.split("_")[0]]
      cells.append(f"{entry[key]['percent']:.2f} % ({size - entry[key]['count']:,} without)")
    table.add_row(entry["lower"], entry["upper"], *cells)
  return table


def print_summary(results: dict[str, Any], console: Console) -> None:
  """All tables: graphs, edge lengths, degrees, processors, coverage, then the triangle quality of each processor."""
  graphs = results["graphs"]
  console.print(graphs_table(graphs))
  console.print(summary_table("Edge lengths (km)", [(name, stats["edge_length_km"]) for name, stats in graphs.items()]))
  console.print(degrees_table(graphs))
  console.print(processors_table(graphs))
  console.print(coverage_table(results["coverage"], graphs))
  for name, stats in graphs.items():
    if stats["kind"] == "processor":
      triangles = stats["triangles"]
      faces, inverted = _format(triangles["faces"]["value"]), _format(triangles["inverted"]["value"])
      title = f"Triangle quality of {name} ({faces} faces, {inverted} inverted)"
      console.print(summary_table(title, [(key, triangles[key]) for key in TRIANGLE_QUALITY]))
