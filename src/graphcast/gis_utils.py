# Copyright 2026 Stefano Campanella.
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
#
# Project code is intended to build models for processing georeferenced data, an tries to be as generic as
# possible. However, it's first and principal goal is to implement a forecasting model, which feeds on numerical models
# data (mainly reanalyses), which at times might use different conventions from the ones used in observational datasets
# and other GIS applications. In particular, it inherits the design choices of the original GraphCast implementation,
# which was trained on ERA5 data from ECMWF. It also depends on gdal and proj to build mesh-graphs, and uses tools as
# cartopy to plot data. Unfortunately, it is impossible to have an all encompassing solution.
# Here's an outline of the problem.
#
#   1. Datasets are stored on regular grids using equirectangular projection, but with longitudes ranging from 0 to 360.
#      When data is provided as an xarray.Dataset the order of dimensions does not matter as one has named dimensions.
#      However, when data is provided as a numpy array, the first dimension corresponds to latitudes and the second to
#      longitudes.
#   2. Traditionally, GIS applications use the (longitude, latitude) order. However, recent versions of proj and gdal
#      honours the order of the Coordinate Reference System (CRS) specified using a WKT string. In theory, this piece
#      of information could be attached as metadata every time some georeferenced data is manipulated, and transformed
#      accordingly. Unfortunately, there is no way of specifying the range and wrapping behaviour of longitudes in WKT
#      strings (see:
#      https://gis.stackexchange.com/questions/256215/wkt-for-epsg4326-with-lon-0-to-360-instead-of-180-to-180)
#   3. Pyproj expects tuples of arrays whose nth element is the nth coordinate, while most manipulations using
#      numpy.ndarrays use the last dimension to index coordinates.
#
# The proposed solution is to wrap a pyproj.Transformer object in a function that takes care of unpacking and
# packing data. The only point where grid-defined data (using the ECMWF convention, i.e., lon wrapping at 180) and the
# mesh-graph data (using standard one, i.e., lon in [-180, 180]) need to be consistent is when computing the grid2mesh
# encoder and mesh2grid decoder graphs in graphcast/model.py. These functions use a kd-tree search in cartesian (R^3)
# coordinates, hence it is enough to ensure that the conversion to cartesian coordinates using is consistent in
# graphcast/mesh_connectivity.py:_grid_lat_lon_to_coordinates.
#
# In brief:
#   - Grid data ALWAYS assumes EPSG:4326 with ECMWF convention.
#   - Mesh data ALWAYS assumes attached WKT string CRS using standard convention and works on np.ndarrays, with the
#     only exception of EquirectangularGraph type and therefore *_to_latlon functions in graphcast/mesh_graph.py.
#   - gdal.WriteArray expects a np.ndarray with shape (latitude, longitude).
# To convert an xarray.Dataset to standard convention use wrap_longitude defined below.
#
# There is more than one class implementing the notion of CRS, specifically osgeo.osr.SpatialReference and
# pyproj.crs.CRS). Pyproj.Transformer.from_crs expects something that can be converted to a pyproj.crs.CRS objects,
# which includes an object with a `to_wkt` method (WKT strings are the preferred way of exchanging CRS information),
# see:
#   - https://proj.org/en/stable/faq.html#what-is-the-best-format-for-describing-coordinate-reference-systems
#   - https://pyproj4.github.io/pyproj/stable/api/crs/crs.html#pyproj.crs.CRS.__init__.
#
# For this reason, we use a thin wrapper around osr.SpatialReference objects with such methods.
# Finally, a few recurring CRS objects are declared here for convenience. SRSRegistry is intended to be used to reach
# these CRS objects from short strings in config files.
import warnings
from collections.abc import Callable
from typing import Literal, cast

import numpy as np
import xarray as xr
from osgeo import gdal, osr
from pyproj import Transformer

FloatingPoint = np.float32 | np.float64 | float
GDFloatingPoint = int
CoordinatesTuple = tuple[np.ndarray, np.ndarray] | tuple[np.ndarray, np.ndarray, np.ndarray]
Coordinates = np.ndarray | CoordinatesTuple

_stereographic_wkt = """
PROJCRS["unknown",
    BASEGEOGCRS["unknown",
        DATUM["Unknown based on WGS 84 ellipsoid",
            ELLIPSOID["WGS 84",6378137,298.257223563,
                LENGTHUNIT["metre",1],
                ID["EPSG",7030]]],
        PRIMEM["Greenwich",0,
            ANGLEUNIT["degree",0.0174532925199433],
            ID["EPSG",8901]]],
    CONVERSION["unknown",
        METHOD["Polar Stereographic (variant A)",
            ID["EPSG",9810]],
        PARAMETER["Latitude of natural origin",90,
            ANGLEUNIT["degree",0.0174532925199433],
            ID["EPSG",8801]],
        PARAMETER["Longitude of natural origin",0,
            ANGLEUNIT["degree",0.0174532925199433],
            ID["EPSG",8802]],
        PARAMETER["Scale factor at natural origin",1,
            SCALEUNIT["unity",1],
            ID["EPSG",8805]],
        PARAMETER["False easting",0,
            LENGTHUNIT["metre",1],
            ID["EPSG",8806]],
        PARAMETER["False northing",0,
            LENGTHUNIT["metre",1],
            ID["EPSG",8807]]],
    CS[Cartesian,2],
        AXIS["(E)",south,
            MERIDIAN[90,
                ANGLEUNIT["degree",0.0174532925199433,
                    ID["EPSG",9122]]],
            ORDER[1],
            LENGTHUNIT["metre",1,
                ID["EPSG",9001]]],
        AXIS["(N)",south,
            MERIDIAN[180,
                ANGLEUNIT["degree",0.0174532925199433,
                    ID["EPSG",9122]]],
            ORDER[2],
            LENGTHUNIT["metre",1,
                ID["EPSG",9001]]]]
"""
_cartesian_wkt = """
GEODCRS["unknown",
    DATUM["Unknown based on WGS 84 ellipsoid",
        ELLIPSOID["WGS 84",6378137,298.257223563,
            LENGTHUNIT["metre",1],
            ID["EPSG",7030]]],
    PRIMEM["Greenwich",0,
        ANGLEUNIT["degree",0.0174532925199433],
        ID["EPSG",8901]],
    CS[Cartesian,3],
        AXIS["(X)",geocentricX,
            ORDER[1],
            LENGTHUNIT["metre",1,
                ID["EPSG",9001]]],
        AXIS["(Y)",geocentricY,
            ORDER[2],
            LENGTHUNIT["metre",1,
                ID["EPSG",9001]]],
        AXIS["(Z)",geocentricZ,
            ORDER[3],
            LENGTHUNIT["metre",1,
                ID["EPSG",9001]]],
    REMARK["PROJ CRS string: +proj=cart +ellps=WGS84 +units=m +x_0=0 +y_0=0"]]
"""
_cartesian_unit_sphere_wkt = """
GEODCRS["unknown",
    DATUM["unknown",
        ELLIPSOID["unknown",1,0,
            LENGTHUNIT["metre",1,
                ID["EPSG",9001]]]],
    PRIMEM["Reference meridian",0,
        ANGLEUNIT["degree",0.0174532925199433,
            ID["EPSG",9122]]],
    CS[Cartesian,3],
        AXIS["(X)",geocentricX,
            ORDER[1],
            LENGTHUNIT["metre",1,
                ID["EPSG",9001]]],
        AXIS["(Y)",geocentricY,
            ORDER[2],
            LENGTHUNIT["metre",1,
                ID["EPSG",9001]]],
        AXIS["(Z)",geocentricZ,
            ORDER[3],
            LENGTHUNIT["metre",1,
                ID["EPSG",9001]]],
    REMARK["PROJ CRS string: +proj=cart +a=1 +b=1 +units=m +x_0=0 +y_0=0"]]
"""
_equirectangular_wkt = """
GEOGCRS["unknown",
    DATUM["World Geodetic System 1984",
        ELLIPSOID["WGS 84",6378137,298.257223563,
            LENGTHUNIT["metre",1]],
        ID["EPSG",6326]],
    PRIMEM["Greenwich",0,
        ANGLEUNIT["degree",0.0174532925199433],
        ID["EPSG",8901]],
    CS[ellipsoidal,2],
        AXIS["longitude",east,
            ORDER[1],
            ANGLEUNIT["degree",0.0174532925199433,
                ID["EPSG",9122]]],
        AXIS["latitude",north,
            ORDER[2],
            ANGLEUNIT["degree",0.0174532925199433,
                ID["EPSG",9122]]]]
"""

osr.UseExceptions()


class CoordinateReferenceSystem(osr.SpatialReference):
  # Dumb hack: StereoMeshSizeField in geospatial_mesh_utils.py exists to be used as a callback inside seamsh functions,
  # which will pass an `osr.SpatialReference`.
  # However, pyproj.Transformer.from_crs (used in `get_transform` and hence in StereoMeshSizeField objects) expects an
  # object with a `to_wkt` method, so `CoordinateReferenceSystem` has been defined here.
  # An alternative way of switching between the two is to get a `projection_crs` of type CoordinateReferenceSystem
  # from a `projection` of type `osr.SpatialReference` with
  # projection_crs = CoordinateReferenceSystem(projection.ExportToWkt())
  # The previous however is slow (requires convertions to and from WKT strings at each call).
  @classmethod
  def from_osr(cls, sr: osr.SpatialReference) -> "CoordinateReferenceSystem":
    sr.__class__ = cls
    sr = cast("CoordinateReferenceSystem", sr)
    return sr

  def to_wkt(self):
    return self.ExportToWkt()


stereographic_crs = CoordinateReferenceSystem(_stereographic_wkt)
cartesian_crs = CoordinateReferenceSystem(_cartesian_wkt)
cartesian_unit_sphere_crs = CoordinateReferenceSystem(_cartesian_unit_sphere_wkt)
equirectangular_crs = CoordinateReferenceSystem(_equirectangular_wkt)

CRSName = Literal["stereographic", "cartesian", "equirectangular"]
CRSRegistry = {
  "stereographic": stereographic_crs,
  "cartesian": cartesian_crs,
  "equirectangular": equirectangular_crs,
}


def get_transform(
  source: CoordinateReferenceSystem,
  destination: CoordinateReferenceSystem,
  pack_back=True,
  **kwargs,
) -> Callable[[Coordinates], Coordinates]:
  """Gets a function that transforms coordinates from one projection to another."""
  transformer = Transformer.from_crs(source, destination)
  assert transformer.source_crs is not None and transformer.target_crs is not None
  num_coords_in = len(transformer.source_crs.axis_info)
  num_coords_out = len(transformer.target_crs.axis_info)

  def wrapped(coordinates: Coordinates) -> Coordinates:
    # We assume that the coordinate dimension is the last one.
    if isinstance(coordinates, tuple):
      assert len(coordinates) == num_coords_in, (
        f"Expected {num_coords_in} coordinates, got {len(coordinates)}"
      )
      if len(coordinates) == 2:
        xx, yy = coordinates  # type: ignore
        zz = np.zeros_like(xx)
      else:
        xx, yy, zz = coordinates  # type: ignore
    elif isinstance(coordinates, np.ndarray):
      assert coordinates.shape[-1] == num_coords_in, (
        f"Expected {num_coords_in} coordinates, got {coordinates.shape[-1]}"
      )
      if coordinates.shape[-1] == 2:
        xx = coordinates[..., 0]
        yy = coordinates[..., 1]
        zz = np.zeros_like(xx)
      else:
        xx = coordinates[..., 0]
        yy = coordinates[..., 1]
        zz = coordinates[..., 2]
    else:
      raise ValueError(f"Coordinates must be a tuple or a numpy array, got {type(coordinates)}")
    result = transformer.transform(xx=xx, yy=yy, zz=zz, inplace=False, **kwargs)
    assert len(result) >= num_coords_out, (
      f"Expected {num_coords_out} coordinates, got {len(result)}"
    )
    # We don't use elevation, in case of conversion to equirectangular projection drop it.
    if len(result) == 3 and num_coords_out == 2:
      if not np.allclose(result[2], 0.0):
        warnings.warn(f"Expected elevation to be zero, got {result[2]}")
      result = result[0], result[1]
    if isinstance(coordinates, np.ndarray) and pack_back:
      result = np.stack(result, axis=-1)
    return result

  return wrapped


def wrap_longitude(da: xr.DataArray, longitude_dim: str = "lon"):
  def _wrap(da: xr.DataArray):
    """Wraps around longitude dimension."""
    longitudes = da[longitude_dim].to_numpy()
    size = len(longitudes)
    dayline_index = np.argwhere(longitudes > 180.0)[0].item()
    wrapped = da.copy()
    wrapped[{longitude_dim: slice(None, size - dayline_index)}] = da[
      {longitude_dim: slice(dayline_index, None)}
    ].to_numpy()
    wrapped[{longitude_dim: slice(size - dayline_index, None)}] = da[
      {longitude_dim: slice(None, dayline_index)}
    ].to_numpy()
    return wrapped

  da_wrapped = _wrap(da)
  longitude_orig = da[longitude_dim].to_numpy()
  longitude_orig = xr.DataArray(
    data=longitude_orig,
    coords={longitude_dim: longitude_orig},
    dims=longitude_dim,
    name=longitude_dim + "_orig",
    attrs=da[longitude_dim].attrs,
  )
  lon_wrapped = _wrap(longitude_orig)
  lon_wrapped = lon_wrapped.to_numpy()
  lon_wrapped = np.where(lon_wrapped <= 180.0, lon_wrapped, lon_wrapped - 360.0)
  lon_wrapped = xr.DataArray(
    data=lon_wrapped,
    coords={longitude_dim: lon_wrapped},
    dims=longitude_dim,
    name=longitude_dim,
    attrs=longitude_orig.attrs,
  )
  da_wrapped = da_wrapped.assign_coords({longitude_dim: lon_wrapped})
  return da_wrapped


def map_on_grid(func, latitude: xr.DataArray, longitude: xr.DataArray) -> xr.DataArray:
  """Maps a GIS function expecting coordinates on a regular grid,
  assumes GIS convention for inputs and outputs (not wrapping longitudes at 180)."""
  num_lats = len(latitude)
  num_lons = len(longitude)
  xx, yy = np.meshgrid(longitude, latitude, indexing="ij")
  coordinates = np.stack([xx, yy], axis=-1)
  values = func(coordinates, equirectangular_crs)
  values = values.reshape(num_lons, num_lats)
  values = xr.DataArray(
    values,
    dims=(longitude.name, latitude.name),
    coords={longitude.name: longitude, latitude.name: latitude},
  )
  values = values.transpose(latitude.name, longitude.name)

  return values


def xarray_to_gdal_raster(
  da: xr.DataArray,
  latitude_dim: str = "lat",
  longitude_dim: str = "lon",
  no_data_value: FloatingPoint | None = np.nan,
  gdal_dtype: GDFloatingPoint = gdal.GDT_Float32,
) -> gdal.Dataset:
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
  assert da.dims == (latitude_dim, longitude_dim), (
    f"Dataset must have dimensions ({latitude_dim}, {longitude_dim}), got {da.dims}"
  )
  da = wrap_longitude(da, longitude_dim)
  da = da.transpose(latitude_dim, longitude_dim)
  data = da.to_numpy()
  lat = da[latitude_dim].to_numpy()
  num_lats = len(lat)
  lon = da[longitude_dim].to_numpy()
  num_lons = len(lon)

  # Grid shape, and resolution
  lat_delta = np.diff(lat)
  lat_res = lat_delta[0]
  assert np.all(lat_delta == lat_res), f"Latitude grid must be uniformly spaced, got {lat_delta}"
  lon_delta = np.diff(lon)
  lon_res = lon_delta[0]
  assert np.all(lon_delta == lon_delta[0]), (
    f"Longitude grid must be uniformly spaced, got {lon_delta}"
  )

  # Ensure north-up orientation, as GDAL expects
  if lat_res > 0:
    data = np.flipud(data)
    lat_res = -lat_res

  # Compute geotransform
  x_min = lon.min() - lon_res / 2
  y_max = lat.max() - lat_res / 2
  geotransform = (x_min, lon_res, 0.0, y_max, 0.0, lat_res)

  # Create GDAL dataset in memory
  driver = gdal.GetDriverByName("MEM")
  gdal_ds = driver.Create("", num_lons, num_lats, 1, gdal_dtype)
  gdal_ds.SetGeoTransform(geotransform)
  gdal_ds.SetProjection(_equirectangular_wkt)

  # Write data
  band = gdal_ds.GetRasterBand(1)
  band.WriteArray(data)

  if no_data_value is not None:
    band.SetNoDataValue(no_data_value)

  band.FlushCache()

  return gdal_ds
