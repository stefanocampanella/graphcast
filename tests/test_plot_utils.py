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
"""Tests of the plotting helpers (plot_utils.py), with matplotlib's non-interactive Agg backend (see conftest.py)."""

import matplotlib.pyplot as plt
import numpy as np
import pytest
import xarray as xr

from graphcast import plot_utils
from graphcast.mesh_graph import EquirectangularGraph


@pytest.fixture
def data() -> xr.Dataset:
  values = np.arange(2 * 3 * 2 * 4 * 5, dtype=float).reshape(2, 3, 2, 4, 5)
  return xr.Dataset(
    {"t": (("batch", "time", "level", "lat", "lon"), values)},
    coords={
      "time": np.arange(1, 4) * np.timedelta64(1, "D"),
      "level": [10, 20],
      "lat": np.linspace(-60, 60, 4),
      "lon": np.arange(0, 360, 72),
    },
  )


def test_select(data):
  """A variable of the first batch element, at a level, for at most max_steps steps."""
  selected = plot_utils.select(data, "t", level=20, max_steps=2)
  assert selected.dims == ("time", "lat", "lon")
  np.testing.assert_array_equal(selected.values, data["t"].isel(batch=0, time=[0, 1]).sel(level=20).values)


@pytest.mark.parametrize(
  ("center", "robust", "limits", "cmap"),
  [
    (None, False, (0.0, 100.0), "viridis"),
    (None, True, (2.0, 98.0), "viridis"),
    (30.0, False, (-40.0, 100.0), "RdBu_r"),
  ],
)
def test_scale(center, robust, limits, cmap):
  """Color limits from the data range (2-98 percentiles if robust), symmetric around the center if given."""
  values = xr.DataArray(np.linspace(0.0, 100.0, 101))
  _, norm, colormap = plot_utils.scale(values, center=center, robust=robust)
  np.testing.assert_allclose([norm.vmin, norm.vmax], limits)
  assert colormap == cmap


def test_plot_data_animates_the_time_steps(data):
  """One panel per field; the update function shows the frame's step and data."""
  fields = {name: plot_utils.scale(plot_utils.select(data, "t", level=level)) for name, level in (("a", 10), ("b", 20))}

  figure, update, num_frames = plot_utils.plot_data(fields, "title", plot_size=1)
  update(2)

  assert num_frames == 3
  panels = [axis for axis in figure.axes if axis.get_title() in fields]
  assert [axis.get_title() for axis in panels] == ["a", "b"]
  np.testing.assert_array_equal(panels[1].images[0].get_array(), fields["b"][0].isel(time=2).values)
  assert "3 days" in figure.get_suptitle()
  plt.close(figure)


def test_points_and_lines_of_a_graph():
  """Points are (longitude, latitude) pairs and lines join the ends of each edge."""
  graph = EquirectangularGraph(
    vertices=(np.array([0.0, 10.0, 20.0]), np.array([100.0, 110.0, 120.0])), edges=(np.array([0, 1]), np.array([1, 2]))
  )
  points, lines = plot_utils.get_points_and_lines(graph)
  assert points == [(100.0, 0.0), (110.0, 10.0), (120.0, 20.0)]
  assert lines == [[(100.0, 0.0), (110.0, 10.0)], [(110.0, 10.0), (120.0, 20.0)]]
  collection = plot_utils.get_line_collection(graph, linewidths=0.5)
  np.testing.assert_array_equal(collection.get_segments()[1], [[110.0, 10.0], [120.0, 20.0]])


def test_mesh_size_plot():
  """The mesh size map is drawn on a Robinson projection, clipped to the quantiles (and normalized to 0-1)."""
  lon = np.arange(0.0, 360.0, 30.0)
  size = xr.DataArray(
    np.random.default_rng(0).uniform(50.0, 150.0, (6, lon.size)),
    dims=("lat", "lon"),
    coords={"lat": np.linspace(-75, 75, 6), "lon": lon},
  )
  figure = plot_utils.mesh_size_plot(size, normalize=True)
  image = figure.axes[0].get_images()[0].get_array()
  assert np.nanmin(image) == 0.0 and np.nanmax(image) == 1.0
  plt.close(figure)
