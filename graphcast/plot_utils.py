import datetime
import math
from typing import Optional, Callable, Sequence

import cartopy.crs as ccrs
import matplotlib
import matplotlib.collections as mc
import matplotlib.pyplot as plt
import numpy as np
import xarray

from graphcast.mesh_graph import WGSGraph

Point = tuple[float, float]
Points = Sequence[Point]
Line = Sequence[Point]
Lines = Sequence[Line]


def select(
    data: xarray.Dataset,
    variable: str,
    level: Optional[int] = None,
    max_steps: Optional[int] = None
) -> xarray.DataArray:
  data = data[variable]
  if "batch" in data.dims:
    data = data.isel(batch=0)
  if max_steps is not None and "time" in data.sizes and max_steps < data.sizes["time"]:
    data = data.isel(time=range(0, max_steps))
  if level is not None and "level" in data.coords:
    data = data.sel(level=level)
  return data

def scale(
    data: xarray.Dataset,
    center: Optional[float] = None,
    robust: bool = False,
) -> tuple[xarray.Dataset, matplotlib.colors.Normalize, str]:
  vmin = np.nanpercentile(data, (2 if robust else 0))
  vmax = np.nanpercentile(data, (98 if robust else 100))
  if center is not None:
    diff = max(vmax - center, center - vmin)
    vmin = center - diff
    vmax = center + diff
  return (data, matplotlib.colors.Normalize(vmin, vmax),
          ("RdBu_r" if center is not None else "viridis"))

def plot_data(
    data: dict[str, xarray.Dataset],
    fig_title: str,
    plot_size: float = 5,
    robust: bool = False,
    cols: int = 4
) -> tuple[matplotlib.figure, Callable, int]:

  first_data = next(iter(data.values()))[0]
  max_steps = first_data.sizes.get("time", 1)
  assert all(max_steps == d.sizes.get("time", 1) for d, _, _ in data.values())

  cols = min(cols, len(data))
  rows = math.ceil(len(data) / cols)
  figure = plt.figure(figsize=(plot_size * 2 * cols,
                               plot_size * rows))
  figure.suptitle(fig_title, fontsize=16)
  figure.subplots_adjust(wspace=0, hspace=0)
  figure.tight_layout()

  images = []
  for i, (title, (plot_data, norm, cmap)) in enumerate(data.items()):
    ax = figure.add_subplot(rows, cols, i+1)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title)
    im = ax.imshow(
      plot_data.isel(time=0, missing_dims="ignore"), norm=norm,
      origin="lower", cmap=cmap)
    plt.colorbar(
      mappable=im,
      ax=ax,
      orientation="vertical",
      pad=0.02,
      aspect=16,
      shrink=0.75,
      cmap=cmap,
      extend=("both" if robust else "neither"))
    images.append(im)

  def update(frame):
    if "time" in first_data.dims:
      td = datetime.timedelta(microseconds=first_data["time"][frame].item() / 1000)
      figure.suptitle(f"{fig_title}, {td}", fontsize=16)
    else:
      figure.suptitle(fig_title, fontsize=16)
    for im, (plot_data, norm, cmap) in zip(images, data.values()):
      im.set_data(plot_data.isel(time=frame, missing_dims="ignore"))

  return figure, update, max_steps


def get_points_and_lines(wgs_graph: WGSGraph) -> tuple[Points, Lines]:
  """Returns points and lines from a Graph.

  Args:
     wgs_graph: geospatially localized vertices and undirected edges between them.
  Returns:
     Tuple with points (pairs of longitude, latitude) and lines (sequences of points) representing graph.

  """
  latitudes, longitudes = wgs_graph.vertices
  # points use (longitude, latitude) ordering to comply with other plot utils and notebooks.
  points = [(lon, lat) for lon, lat in zip(longitudes, latitudes)]
  lines = [[points[n1], points[n2]] for (n1, n2) in zip(wgs_graph.edges[0], wgs_graph.edges[1])]

  return points, lines


def get_line_collection(wgs_graph: WGSGraph, **kwargs) ->  mc.LineCollection:
  """Gets a LineCollection for plotting the graph.

  Args:
     wgs_graph: geospatially localized vertices and undirected edges between them.
     **kwargs: additional arguments to pass to matplotlib.collections.LineCollection.
  Returns:
     LineCollection for plotting the graph.

  """
  _, line_segments = get_points_and_lines(wgs_graph)
  line_segments = np.array(line_segments)
  line_collection = mc.LineCollection(line_segments, **kwargs)
  return line_collection


def _wrap(da: xarray.DataArray, longitudes: np.ndarray):
  """ Wraps around longitude dimension. It assumes that longitude is the last dimension."""
  size = longitudes.size
  dayline_index = np.argmax(longitudes > 180.0)
  data = da.data
  wrapped = da.copy()
  wrapped.data[..., :(size - dayline_index)] = data[..., dayline_index:]
  wrapped.data[..., (size - dayline_index):] = data[..., :dayline_index]
  return wrapped


def _fix_longitude(da: xarray.DataArray):
  longitude = da.coords['longitude'].copy()
  da_wrapped = _wrap(da, longitude.to_numpy())
  lon_wrapped = _wrap(longitude, longitude.to_numpy())
  lon_wrapped = lon_wrapped.to_numpy()
  lon_wrapped = np.where(lon_wrapped <= 180.0, lon_wrapped, lon_wrapped - 360.0)
  lon_wrapped = xarray.DataArray(data=lon_wrapped, coords={'longitude': lon_wrapped}, dims='longitude',
                             name='longitude')
  da_wrapped = da_wrapped.assign_coords(longitude=lon_wrapped)
  return da_wrapped


def fix_longitude(plot_f):

  def new_plot_f(*args, **kwargs):
    args = map(lambda da: _fix_longitude(da) if isinstance(da, xarray.DataArray) else da, args)
    return plot_f(*args, **kwargs)

  return new_plot_f


@fix_longitude
def mesh_size_plot(alpha, q_low=0.2, q_high=0.8, normalize=False):
  min = np.nanquantile(alpha, q_low)
  max = np.nanquantile(alpha, q_high)
  alpha = np.clip(alpha, min, max)
  if normalize:
    alpha = (alpha - min) / (max - min)
  fig = plt.figure()
  ax = fig.add_subplot(111, projection=ccrs.Robinson())
  imshow = ax.imshow(alpha, origin='lower', transform=ccrs.PlateCarree())
  fig.colorbar(imshow, orientation='vertical')
  return fig