import functools
from typing import Literal, Union

import numpy as np
import xarray as xr
from osgeo import gdal, osr
from pyproj import Transformer

FloatingPoint = Union[np.float32, np.float64]
GDFloatingPoint = int


osr.UseExceptions()

stereographic_proj = osr.SpatialReference("+proj=stere +ellps=WGS84 +lat_0=90")
cartesian_proj = osr.SpatialReference("+proj=cart +ellps=WGS84 +units=m +x_0=0 +y_0=0")
cartesian_unit_sphere_proj = osr.SpatialReference("+proj=cart +a=1 +b=1 +units=m +x_0=0 +y_0=0")
platecarree_proj = osr.SpatialReference("+proj=latlong +datum=WGS84 +no_defs")

ProjectionRegistry = {'stereographic': stereographic_proj, 'cartesian': cartesian_proj, 'platecarree': platecarree_proj}
Projection = Literal['stereographic', 'cartesian', 'platecarree']


def unpack_points(func, pack_back=True):
  """Decorator to unpack points in 2D or 3D space, apply a function and eventually pack the result back."""

  @functools.wraps(func)
  def wrapper(points: np.ndarray):
    # We assume that the coordinate dimension is the last one.
    if points.shape[-1] == 2:
      xx = points[..., 0]
      yy = points[..., 1]
      zz = None
    elif points.shape[-1] == 3:
      xx = points[..., 0]
      yy = points[..., 1]
      zz = points[..., 2]
    else:
      raise ValueError(f"Trailing dimension must be 2 or 3, got {points.shape[-1]}")
    result = func(xx, yy, zz)
    if pack_back:
      result = np.stack(result, axis=-1)
    return result

  return wrapper


def get_transform(source: str | osr.SpatialReference, destination: str | osr.SpatialReference, pack_back=True):
  """Gets a function that transforms points from one projection to another."""
  if isinstance(source, str):
    source = ProjectionRegistry[source]
  if isinstance(destination, str):
    destination = ProjectionRegistry[destination]
  transformer = Transformer.from_proj(source.ExportToProj4(), destination.ExportToProj4())
  transform = unpack_points(transformer.transform, pack_back=pack_back)
  return transform


def map_on_grid(func, grid: xr.DataArray, longitude_dim='lon', latitude_dim='lat') -> xr.DataArray:
  """Maps a function expecting points on a regular grid in plate carree projection."""
  grid = grid.transpose(longitude_dim, latitude_dim)
  xx, yy = np.meshgrid(grid[longitude_dim], grid[latitude_dim], indexing='ij')
  xx = np.where(grid.astype(bool), xx, 0.0)
  yy = np.where(grid.astype(bool), yy, 0.0)
  xx = xx.flatten()
  yy = yy.flatten()
  points = np.stack([xx, yy], axis=-1)
  values = func(points, platecarree_proj)
  values = values.reshape(grid.shape)
  values = xr.DataArray(values, dims=grid.dims, coords=grid.coords)

  return values


def xarray_to_gdal_raster(da: xr.DataArray,
                          srs: osr.SpatialReference = platecarree_proj,
                          latitude_dim_name: str = 'lat',
                          longitude_dim_name: str = 'lon',
                          no_data_value: FloatingPoint | None = np.nan,
                          gdal_dtype: GDFloatingPoint = gdal.GDT_Float32) -> gdal.Dataset:
  """
  Convert an xarray Dataset on a regular lat/lon grid to a GDAL raster Dataset.

  Args
  ds : xarray.Dataset
    Dataset with dimensions (lat, lon)
  var_name : str
    Name of the variable to convert
  no_data_value : float, optional
    NoData value for the raster
  gdal_dtype : gdal.DataType, optional
    GDAL data type (default: Float32)

  Returns:
    In-memory GDAL raster dataset of type gdal.Dataset.
  """
  assert da.dims == (latitude_dim_name, longitude_dim_name), \
    f"Dataset must have dimensions ({latitude_dim_name}, {longitude_dim_name})"
  da = da.transpose(latitude_dim_name, longitude_dim_name)
  data = da.to_numpy()
  lat = da[latitude_dim_name].to_numpy()
  lon = da[longitude_dim_name].to_numpy()

  # Grid shape, and resolution
  num_lats, num_lons = data.shape
  lat_delta = np.diff(lat)
  lat_res = lat_delta[0]
  assert np.all(lat_delta == lat_res), f"Latitude grid must be uniformly spaced, got {lat_delta}"
  lon_delta = np.diff(lon)
  lon_res = lon_delta[0]
  assert np.all(lon_delta == lon_delta[0]), f"Longitude grid must be uniformly spaced, got {lon_delta}"

  # Ensure north-up orientation, as GDAL expects
  if lat_res > 0:
    data = np.flipud(data)
    lat_res = -lat_res

  # Compute geotransform
  x_min = lon.min() - lon_res / 2
  y_max = lat.max() - lat_res / 2

  # Create GDAL dataset in memory
  driver = gdal.GetDriverByName("MEM")
  gdal_ds = driver.Create("", num_lons, num_lats, 1, gdal_dtype)
  gdal_ds.SetGeoTransform(x_min, lon_res, 0.0, y_max, 0.0, lat_res)
  gdal_ds.SetProjection(srs.ExportToWkt())

  # Write data
  band = gdal_ds.GetRasterBand(1)
  band.WriteArray(data)

  if no_data_value is not None:
    band.SetNoDataValue(no_data_value)

  band.FlushCache()

  return gdal_ds
