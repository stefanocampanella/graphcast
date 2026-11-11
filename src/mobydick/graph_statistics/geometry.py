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
"""Geometry of the graphs: geodesic edge lengths, and the quality of the mesh triangles.

Triangle quality follows Shewchuk, "What is a good linear finite element?" (2002), and
https://scicomp.stackexchange.com/a/27095. The metrics are computed on the flat triangle through the three vertices in
Earth-centred coordinates; for ocean meshes (edges up to ~200 km) it differs from the spherical triangle by far less
than the metrics' variation.
"""

from typing import Any

import numpy as np
import pyproj

from mobydick.graph_statistics.graphs import Graph
from mobydick.graph_statistics.quantities import described, histogram, summarize
from mobydick.mesh_graph import MeshGraph


def geodesic_lengths(lat1: np.ndarray, lon1: np.ndarray, lat2: np.ndarray, lon2: np.ndarray) -> np.ndarray:
  """Geodesic distances on the WGS84 ellipsoid (metres) between points given in degrees."""
  _, _, distances = pyproj.Geod(ellps="WGS84").inv(lon1, lat1, lon2, lat2)
  return np.asarray(distances)


def edge_lengths_km(graph: Graph) -> np.ndarray:
  """Geodesic length of every edge of the graph (km)."""
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


# Triangle quality metrics and their descriptions (also the JSON descriptions).
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
  """The metrics of `TRIANGLE_QUALITY` for each face (`angle`: three values per face).

  Args:
    vertices: Earth-centred coordinates [num_vertices, 3] (metres).
    faces: Vertex indices [num_faces, 3].
  """
  a, b, c = (vertices[faces[:, i]] for i in range(3))
  # Edge lengths [num_faces, 3], each opposite a vertex: |bc|, |ca|, |ab|.
  lengths = np.stack([np.linalg.norm(b - c, axis=1), np.linalg.norm(c - a, axis=1), np.linalg.norm(a - b, axis=1)], 1)
  normal = np.cross(b - a, c - a)
  area = 0.5 * np.linalg.norm(normal, axis=1)
  # +1 when the normal points away from the Earth's centre (the centroid is along a + b + c), -1 for inverted faces.
  orientation = np.sign(np.einsum("ij,ij->i", normal, a + b + c))
  longest, shortest = lengths.max(1), lengths.min(1)
  inradius = 2 * area / lengths.sum(1)
  circumradius = lengths.prod(1) / (4 * area)
  # Interior angles from the law of cosines, each at the vertex opposite the corresponding edge.
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
    # The smallest height is the one on the longest edge: 2 * area / longest.
    "aspect_ratio": np.sqrt(3) / 2 * longest / (2 * area / longest),
    "radius_ratio": 2 * inradius / circumradius,
    "volume_length": orientation * 4 / np.sqrt(3) * area / np.mean(lengths**2, axis=1),
    "angle": angles.reshape(-1),
    "min_angle": angles.min(1),
    "max_angle": angles.max(1),
    "edge_ratio": longest / shortest,
  }


def triangle_statistics(mesh: MeshGraph, bins: int) -> dict[str, Any]:
  """Summaries of the triangle quality of a mesh, with histograms of the bounded metrics."""
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
