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
"""Tests of the graph feature construction in model_utils."""

import numpy as np
import pytest

from mobydick.leviathan import model_utils

# From a node at (0°, 0°) to its neighbours one degree east, north, west and south, and their forward azimuths.
NEIGHBOURS_LON = np.array([0.0, 1.0, 0.0, -1.0, 0.0])
NEIGHBOURS_LAT = np.array([0.0, 0.0, 1.0, 0.0, -1.0])
SENDERS = np.zeros(4, dtype=int)
RECEIVERS = np.arange(1, 5)
AZIMUTHS_DEG = np.array([90.0, 0.0, -90.0, 180.0])

FEATURE_FLAGS = {
  "add_node_position": False,
  "add_node_coordinates": False,
  "add_edge_length": False,
  "add_edge_receiver_coordinates": False,
}


def _graph_edge_features(**flags) -> np.ndarray:
  _, edge_features = model_utils.get_graph_spatial_features(
    node_lon=NEIGHBOURS_LON,
    node_lat=NEIGHBOURS_LAT,
    senders=SENDERS,
    receivers=RECEIVERS,
    **{**FEATURE_FLAGS, **flags},
  )
  return edge_features


def _bipartite_edge_features(**flags) -> np.ndarray:
  # Longitudes in [0, 360) for the bipartite version.
  lon = np.mod(NEIGHBOURS_LON, 360.0)
  _, _, edge_features = model_utils.get_bipartite_graph_spatial_features(
    senders_node_lon=lon[:1],
    senders_node_lat=NEIGHBOURS_LAT[:1],
    senders=SENDERS,
    receivers_node_lon=lon,
    receivers_node_lat=NEIGHBOURS_LAT,
    receivers=RECEIVERS,
    **{**FEATURE_FLAGS, **flags},
  )
  return edge_features


@pytest.mark.parametrize("features", [_graph_edge_features, _bipartite_edge_features], ids=["graph", "bipartite"])
def test_edge_direction_is_the_unit_vector_east_and_north(features):
  """Edge direction features are (sin, cos) of the forward azimuth: (1, 0) eastwards, (0, 1) northwards, (-1, 0)
  westwards, (0, -1) southwards. They were computed from azimuths converted to degrees twice (pyproj returns
  degrees), i.e. sin and cos of meaningless angles."""
  edge_features = features(add_edge_fwd_azimuth=False, add_edge_direction=True)

  expected = np.stack([np.sin(np.deg2rad(AZIMUTHS_DEG)), np.cos(np.deg2rad(AZIMUTHS_DEG))], axis=-1)
  np.testing.assert_allclose(edge_features, expected, rtol=0, atol=1e-9)


@pytest.mark.parametrize("features", [_graph_edge_features, _bipartite_edge_features], ids=["graph", "bipartite"])
def test_edge_forward_azimuth_is_in_turns(features):
  """The forward azimuth feature is in turns (a quarter eastwards, half southwards), as documented."""
  edge_features = features(add_edge_fwd_azimuth=True, add_edge_direction=False)

  np.testing.assert_allclose(edge_features[:, 0], AZIMUTHS_DEG / 360.0, rtol=0, atol=1e-9)


@pytest.mark.parametrize("features", [_graph_edge_features, _bipartite_edge_features], ids=["graph", "bipartite"])
@pytest.mark.parametrize("normalization", [None, 50_000.0], ids=["longest_edge", "given_scale"])
def test_edge_length_is_divided_by_the_normalization_scale(features, normalization):
  """Edge lengths are divided by `edge_normalization`, a length (`model.mesh2grid_edge_normalization_factor` in the
  config is one number), or by the longest edge without it. The code unpacked it as a (location, scale) pair, so
  setting the option failed."""
  edge_features = features(
    add_edge_fwd_azimuth=False, add_edge_direction=False, add_edge_length=True, edge_normalization=normalization
  )

  # Geodesic lengths of one degree along the equator and along the meridian (WGS84).
  lengths = np.array([111_319.49, 110_574.39, 111_319.49, 110_574.39])
  expected = lengths / (lengths.max() if normalization is None else normalization)
  np.testing.assert_allclose(edge_features[:, 0], expected, rtol=1e-6)
