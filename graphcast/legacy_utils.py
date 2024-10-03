from typing import Optional, Any
from graphcast import icosahedral_mesh
from graphcast import model
from graphcast import checkpoint
import chex
import xarray
import numpy as np


@chex.dataclass(frozen=True, eq=True)
class ModelConfig:
  """Legacy version of model.ModelConfig.

  Properties different from model.ModelConfig:
    resolution: The resolution of the data, in degrees (e.g. 0.25 or 1.0).
    mesh_size: How many refinements to do on the multi-mesh.
  """
  resolution: float
  mesh_size: int
  latent_size: int
  gnn_msg_steps: int
  hidden_layers: int
  radius_query_fraction_edge_length: float
  mesh2grid_edge_normalization_factor: Optional[float] = None


@chex.dataclass(frozen=True, eq=True)
class CheckPoint:
  """Legacy version of model.CheckPoint"""
  params: dict[str, Any]
  model_config: ModelConfig
  task_config: model.TaskConfig
  description: str
  license: str


def upgrade_legacy_model_config(old_model_config, grid_lat, grid_lon):
  """Read a model_config for the released GraphCast model, then compute the additional attributes needed
  to work with new versions of the code."""
  grid_mask = np.full((grid_lat.shape[0], grid_lon.shape[0]), True)
  meshes = icosahedral_mesh.get_hierarchy_of_triangular_meshes_for_sphere(
    splits=old_model_config.mesh_size)
  finest_mesh = meshes[-1]
  merged_mesh = icosahedral_mesh.merge_meshes(meshes)
  edges = icosahedral_mesh.faces_to_edges(merged_mesh.faces)
  mesh_graph = icosahedral_mesh.MultiMeshGraph(vertices=finest_mesh.vertices, faces=finest_mesh.faces, edges=edges)
  attributes_to_copy = [name for name in old_model_config.__dict__ if name in model.ModelConfig.__dataclass_fields__]
  unchanged_configs = {name: getattr(old_model_config, name) for name in attributes_to_copy}
  model_config = model.ModelConfig(
    grid_lat=grid_lat,
    grid_lon=grid_lon,
    grid_mask=grid_mask,
    mesh_graph=mesh_graph,
    **unchanged_configs)
  return model_config


def read_legacy_checkpoint(params_file, dataset_path, lat_lon_names=("latitude", "longitude")):
  """Read a checkpoint for the released GraphCast model (i.e. those available in the dm-graphcast bucket on
  Google Cloud), then updates the model_config to work with new versions of the code."""
  with params_file.open("rb") as f:
    # Read original model checkpoint and update model_config to work with the new interface
    old_ckpt = checkpoint.load(f, CheckPoint)
    sample_dataset = xarray.open_zarr(dataset_path)
    lat_name, lon_name = lat_lon_names
    model_config = upgrade_legacy_model_config(old_ckpt.model_config,
                                               grid_lat=getattr(sample_dataset, lat_name),
                                               grid_lon=getattr(sample_dataset, lon_name))
    new_ckpt = model.CheckPoint(params=old_ckpt.params,
                                model_config=model_config,
                                task_config=old_ckpt.task_config,
                                description=old_ckpt.description,
                                license=old_ckpt.license)
    return new_ckpt

