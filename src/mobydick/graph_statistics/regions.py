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
"""Regions of the ocean (`RegionConfig`): the nodes of a node set inside any of a list of polygons of (longitude,
latitude) vertices.

The polygon is tested in plate carrée coordinates (longitude and latitude as plane coordinates), with longitudes
brought to [-180, 180): straight polygon edges are lines of constant slope in degrees, not geodesics, which is what a
region drawn on a map means. Polygons must not cross the antimeridian.
"""

import matplotlib.path
import numpy as np

from mobydick.graph_statistics.graphs import NodeSet


def in_region(nodes: NodeSet, polygons: list[list[tuple[float, float]]]) -> np.ndarray:
  """Boolean mask over `nodes`: True for the nodes inside any of `polygons` ((longitude, latitude) vertices in
  degrees)."""
  lon = (np.asarray(nodes.lon, dtype=np.float64) + 180.0) % 360.0 - 180.0
  points = np.column_stack([lon, np.asarray(nodes.lat, dtype=np.float64)])
  inside = np.zeros(len(points), dtype=bool)
  for polygon in polygons:
    inside |= matplotlib.path.Path(np.asarray(polygon, dtype=np.float64)).contains_points(points)
  return inside
