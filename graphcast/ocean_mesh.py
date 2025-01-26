# Copyright 2025 OGS.
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
# Seamsh usage example are taken from: https://jlambrechts.git-page.immc.ucl.ac.be/seamsh/examples/6-stereographics.html
"""Tools for working with triangular ocean meshes."""
import pathlib
import shutil
import tempfile
from typing import Callable

import click
import gmsh
import numpy as np
import pyproj
import seamsh
import xarray as xr
from osgeo import osr
from scipy.interpolate import RegularGridInterpolator

from graphcast.mesh_graph import TriangleMesh

osr.UseExceptions()
stere = osr.SpatialReference("+proj=stere +ellps=WGS84 +lat_0=90")
cart = osr.SpatialReference("+proj=cart +ellps=WGS84 +units=m +x_0=0 +y_0=0")
wgs84 = osr.SpatialReference("+proj=longlat +datum=WGS84 +no_defs")


def map_on_grid(func, mask: xr.DataArray):

  xx, yy = np.meshgrid(mask.longitude, mask.latitude)
  xx = np.where(mask.astype(bool), xx, 0.0)
  yy = np.where(mask.astype(bool), yy, 0.0)
  xx = xx.flatten()
  yy = yy.flatten()
  points = np.stack([xx, yy], axis=-1)
  alpha = func(points, wgs84)
  alpha = alpha.reshape(mask.shape)

  return alpha


class StereoMeshSizeField:

  def __init__(self, size_min, size_max):
    self.min_size = size_min
    self.max_size = size_max

  def criterion(self, x: np.ndarray, projection: osr.SpatialReference) -> np.ndarray:
    pass

  def mesh_size_3d(self, x, projection):
    alpha = self.criterion(x, projection)
    delta = self.min_size + (self.max_size - self.min_size) * alpha
    return delta

  def __call__(self, x, projection: osr.SpatialReference):

    mesh_size = self.mesh_size_3d(x, stere)
    earth_radius_squared = stere.GetSemiMajor() * stere.GetSemiMinor()
    stereo_factor = (4 * earth_radius_squared) / (4 * earth_radius_squared + x[:, 0] ** 2 + x[:, 1] ** 2)
    return mesh_size / stereo_factor

class ConstantField(StereoMeshSizeField):

  def __init__(self, value, size_min, size_max):
    super().__init__(size_min, size_max)
    if value < 0 or value > 1:
      raise ValueError("Value must be between 0 and 1")
    self.value = value

  def criterion(self, x, projection):
    return np.full(x.shape[0], self.value)

class ShoreProximityField(StereoMeshSizeField):

  def __init__(self, domain, sampling, field_min, field_max, size_min, size_max):
    super().__init__(size_min, size_max)
    self.field_min = field_min
    self.field_max = field_max
    self.distance_from_coast_f = seamsh.field.Distance(domain, sampling, projection=cart)

  def criterion(self, x, projection):
    distance_from_coast = np.clip(self.distance_from_coast_f(x, projection), self.field_min, self.field_max)
    alpha = (distance_from_coast - self.field_min) / (self.field_max - self.field_min)
    return alpha

class MaskField(StereoMeshSizeField):

  def __init__(self, mask, q_low, q_high, size_min, size_max):
    super().__init__(size_min, size_max)
    self.q_low = q_low
    self.q_high = q_high
    self.mask = mask
    self.field_min = np.nanquantile(mask, q_low)
    self.field_max = np.nanquantile(mask, q_high)

  def criterion(self, x, projection):
    interp = RegularGridInterpolator((self.mask.longitude, self.mask.latitude), self.mask)
    transformer = pyproj.Transformer.from_proj(projection.ExportToProj4(), wgs84.ExportToProj4())
    field = interp(transformer.transform(x))
    alpha = (field - self.field_min) / (self.field_max - self.field_min)
    return alpha

class HessianField(StereoMeshSizeField):
  pass

class BathymetryField(StereoMeshSizeField):
  pass

class CourantField(StereoMeshSizeField):
  pass


def diff(f, coord):
  grad = f.differentiate(coord=coord)
  if coord == 'longitude':
    grad = grad / np.cos(f.latitude * np.pi / 180.0)
  return grad


def hess(f):
  h = np.empty(f.shape + (2, 2), dtype=f.dtype)

  for (i, coord_i) in enumerate(['latitude', 'longitude']):
    for (j, coord_j) in enumerate(['latitude', 'longitude']):
      h[..., i, j] = diff(diff(f, coord_i), coord_j)

  h = 0.5 * (h + np.swapaxes(h, -1, -2))
  return h


def hess_norm(f):
  h = hess(f)
  singular_values = np.linalg.svd(h, compute_uv=False, hermitian=True)
  h_norm = np.max(singular_values, axis=-1)
  return h_norm


def alpha(f, eps=1.0e-10):
  alpha = np.sqrt(np.abs(f) / np.clip(hess_norm(f), min=eps))
  alpha = xr.DataArray(data=alpha, coords=[f.longitude, f.latitude], dims=['longitude', 'latitude'])
  return alpha


def get_ocean_mesh(domain: seamsh.geometry.Domain, mesh_size: Callable, target_srs=None, save_mesh=None, **kwargs) \
    -> TriangleMesh:
  """Returns the TriangleMesh corresponding to the given domain and mesh size, obtained via seamsh.

  Args:
    domain: domain to mesh
    mesh_size: callable returning the target mesh element size
    target_srs: Target spatial reference system (default cartesian)
    save_mesh: whether to save the mesh to disk, if not None (default None)
    kwargs: keyword arguments to pass to mesh_size
  Returns:
    The computed TriangleMesh

    """

  if target_srs is None:
    target_srs = osr.SpatialReference()
    target_srs.ImportFromProj4("+ellps=WGS84 +proj=cart +units=m +x_0=0 +y_0=0")

  with tempfile.TemporaryDirectory() as tmpdir:
    seamsh.gmsh.mesh(domain, tmpdir + "/natural_earth.msh", mesh_size(**kwargs), output_srs=domain._projection)
    seamsh.gmsh.reproject(tmpdir + "/natural_earth.msh", domain._projection, tmpdir + "/natural_earth_cart.msh", target_srs)
    gmsh.open(tmpdir + "/natural_earth_cart.msh")
    if save_mesh is not None:
      shutil.copy(tmpdir + "/natural_earth_cart.msh", save_mesh)

  node_tags, node_coords, _ = gmsh.model.mesh.get_nodes()
  node_tags_map = {tag: tag - 1 for tag in node_tags}
  node_coords = node_coords.reshape(-1, 3)

  element_tags, _ = gmsh.model.mesh.get_elements_by_type(2)
  element_nodes = []
  for element_tag in element_tags:
    _, element_node_tags, _, _ = gmsh.model.mesh.get_element(element_tag)
    element_node_indices = np.array([node_tags_map[tag] for tag in element_node_tags], dtype=int)
    element_nodes.append(element_node_indices)
  element_nodes = np.vstack(element_nodes)

  return TriangleMesh(vertices=node_coords, faces=element_nodes)


@click.group()
def cli():
  pass


@cli.command()
@click.argument('dataset_path', required=True)
@click.argument('output_path', type=click.Path(path_type=pathlib.Path, writable=True), required=True)
@click.option('--samples', default=100)
def hessian_map(dataset_path, output_path, samples):
  ds = xr.open_mfdataset(dataset_path, engine='zarr', parallel=True)

  def _compute_alpha(da: xr.DataArray):
    time = np.random.choice(da.sizes['time'], size=samples)
    time = np.sort(time)
    da = da.isel(depth=0) if 'depth' in da.dims else da
    da = da.isel(time=time)
    # TODO: why is `da = da.map_blocks(alpha, template=da)` slower?
    da = alpha(da)
    da = da.mean(dim='time')
    return da

  ds = ds.map(_compute_alpha)
  ds.to_netcdf(output_path, mode='w', engine='h5netcdf')


if __name__ == '__main__':
  cli()
