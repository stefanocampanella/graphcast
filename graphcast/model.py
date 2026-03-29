# Copyright 2023 DeepMind Technologies Limited.
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
"""A predictor that runs multiple graph neural networks on mesh data.

It learns to interpolate between the grid and the mesh nodes, with the loss
and the rollouts ultimately computed at the grid level.

It uses ideas similar to those in Keisler (2022):

Reference:
  https://arxiv.org/pdf/2202.07575.pdf

It assumes data across time and level is stacked, and operates only operates in
a 2D mesh over latitudes and longitudes.
"""
import dataclasses
import logging
from typing import Any, Callable, Mapping, Optional, Tuple

import chex
import haiku as hk
import jax
import jax.numpy as jnp
import jraph
import numpy as np
import xarray

from graphcast import deep_typed_graph_net
from graphcast import losses
from graphcast import mesh_connectivity
from graphcast import model_utils
from graphcast import predictor_base
from graphcast import typed_graph
from graphcast import xarray_jax
from graphcast.fourier_features import PositionalEncoder
from graphcast.gis_utils import get_transform, equirectangular_srs, cartesian_srs
from graphcast.mesh_graph import MeshData, MeshGraph, TriangleMesh, faces_to_edges

logger = logging.getLogger(__name__)

Kwargs = Mapping[str, Any]

GNN = Callable[[jraph.GraphsTuple], jraph.GraphsTuple]

# The list of all variables, see: https://github.com/inogs/arco-ocean/blob/main/README.md
ALL_VOLUME_VARS = (
    "thetao",
    "so",
    "uo",
    "vo",
    "glorys_mask",
)
ALL_SURFACE_VARS = (
    "10u",
    "10v",
    "2d",
    "2t",
    "deptho",
    "dis24",
    "z",
    "glofas_mask",
    "i10fg",
    "lsm",
    "mlotst",
    "siconc",
    "sithick",
    "sp",
    "ssrd",
    "strd",
    "tp",
    "uparea",
    "usi",
    "swh",
    "mwd",
    "usd",
    "vsd",
    "vsi",
    "swp",
    "waverys_deptho",
    "waverys_mask",
    "zos"
)
TARGET_VARS = (
    "zos",
    "mlotst",
    "thetao",
    "siconc",
    "sithick",
    "so",
    "uo",
    "usi",
    "vo",
    "vsi",
)
EXTERNAL_FORCING_VARS = (
    "10u",
    "10v",
    "2d",
    "2t",
    "dis24",
    "sp",
    "ssrd",
    "strd",
    "tp",
    "swh",
)
GENERATED_FORCING_VARS = (
    "year_progress_sin",
    "year_progress_cos",
)
FORCING_VARS = EXTERNAL_FORCING_VARS + GENERATED_FORCING_VARS
STATIC_VARS = (
    "deptho",
    "z",
    "glofas_mask",
    "glorys_mask",
    "lsm",
    "uparea",
    "waverys_deptho",
    "waverys_mask",
)
ALL_VARS = TARGET_VARS + FORCING_VARS + STATIC_VARS
LEVELS_10 = (0, 4, 8, 12, 16, 20, 24, 28, 32, 34)


@dataclasses.dataclass(frozen=True, eq=True, repr=True)
class TaskConfig:
  """Defines inputs and targets on which a model is trained and/or evaluated."""
  input_variables: tuple[str, ...]
  # Target variables which the model is expected to predict.
  target_variables: tuple[str, ...]
  forcing_variables: tuple[str, ...]
  levels: tuple[int, ...]
  input_duration: str

# GraphCast includes the forcings within the input variables, but we do not.
# Conceptually, the distinction between inputs, forcings and targets is given by
#
#              +-------+-----------+
#              | past  |  future   |
#  +-----------+-------+-----------+
#  | predictor | inputs | forcings |
#  +-----------+-------+-----------+
#  | predicted |  //   |  targets  |
#  +-----------+-------+-----------+
#
# Hence, the inputs are TARGET_VARS + STATIC_VARS, and not TARGET_VARS + FORCING_VARS + STATIC_VARS as in GraphCast.
DEFAULT_TASK = TaskConfig(
    input_variables=TARGET_VARS + STATIC_VARS,
    target_variables=TARGET_VARS,
    forcing_variables=FORCING_VARS,
    levels=LEVELS_10,
    input_duration="2d",
)


# TODO: In current experiments the value of latent_size, gnn_msg_steps, etc. has been copied from ModelConfig.
#   Are there more sensible choice of values for these quantities, given the differences from the original GraphCast implementation?
@dataclasses.dataclass(frozen=True, eq=True, repr=True)
class ModelConfig:
  """Defines the architecture of the GraphCast neural network architecture.

  Properties:
    gnn_msg_steps: How many Graph Network message passing steps to do.
    latent_size: How many latent features to include in the various MLPs.
    hidden_layers: How many hidden layers for each MLP.
    radius_query_fraction_edge_length: Scalar that will be multiplied by the
        length of the longest edge of the finest mesh to define the radius of
        connectivity to use in the Grid2Mesh graph. Reasonable values are
        between 0.6 and 1. 0.6 reduces the number of grid points feeding into
        multiple mesh nodes and therefore reduces edge count and memory use, but
        1 gives better predictions.
    mesh2grid_edge_normalization_factor: Allows explicitly controlling edge
        normalization for mesh2grid edges. If None, defaults to max edge length.
        This supports using pre-trained model weights with a different graph
        structure to what it was trained on.
  """
  latent_size: int
  gnn_msg_steps: int
  hidden_layers: int
  radius_query_fraction_edge_length: float
  mesh2grid_edge_normalization_factor: Optional[float] = None
  per_variable_weights: Optional[dict] = None
  learnable_fourier_features: bool = True


@dataclasses.dataclass(frozen=True, eq=True, repr=True)
class CheckPoint:
  model_config: ModelConfig
  task_config: TaskConfig
  mesh_data: MeshData
  grid_lat: np.ndarray
  grid_lon: np.ndarray
  grid_mask: Optional[np.ndarray]
  description: str
  license: str
  params: dict[str, Any] = dataclasses.field(default_factory=dict)


# # TODO: add parameters documentation
@dataclasses.dataclass(repr=True)
class GraphCast(hk.Module, predictor_base.Predictor):
  """GraphCast Predictor.

  The model works on graphs that take into account:
  * Mesh nodes: nodes for the vertices of the mesh.
  * Grid nodes: nodes for the points of the grid.
  * Nodes: When referring to just "nodes", this means the joint set of
    both mesh nodes, concatenated with grid nodes.

  The model works with 3 graphs:
  * Grid2Mesh graph: Graph that contains all nodes. This graph is strictly
    bipartite with edges going from grid nodes to mesh nodes using a
    fixed radius query. The grid2mesh_gnn will operate in this graph. The output
    of this stage will be a latent representation for the mesh nodes, and a
    latent representation for the grid nodes.
  * Mesh graph: Graph that contains mesh nodes only. The mesh_gnn will
    operate in this graph. It will update the latent state of the mesh nodes
    only.
  * Mesh2Grid graph: Graph that contains all nodes. This graph is strictly
    bipartite with edges going from mesh nodes to grid nodes such that each grid
    nodes is connected to 3 nodes of the mesh triangular face that contains
    the grid points. The mesh2grid_gnn will operate in this graph. It will
    process the updated latent state of the mesh nodes, and the latent state
    of the grid nodes, to produce the final output for the grid nodes.

  The model is built on top of `TypedGraph`s so the different types of nodes and
  edges can be stored and treated separately.

  """

  _model_config: ModelConfig
  _task_config: TaskConfig
  _mesh_data: MeshData
  _grid_lat: np.ndarray
  _grid_lon: np.ndarray
  _grid_mask: np.ndarray
  _kdtree_workers: int = 1
  _remat: bool = False
  _policy: Callable[..., bool] | None = None
  _prevent_cse: bool = False
  _scan: bool = False
  _name: str | None = None

  def __post_init__(self):

    super().__post_init__(name=self._name or "graphcast")

    self._query_radius = self._model_config.radius_query_fraction_edge_length * self._mesh_data.mesh_size
    self._mesh2grid_edge_normalization_factor = self._model_config.mesh2grid_edge_normalization_factor
    self._per_variable_weights = self._model_config.per_variable_weights
    self._spatial_features_kwargs = dict(
      add_node_position=False,
      add_node_coordinates=True,
      add_edge_fwd_azimuth=False,
      add_edge_direction=True,
      add_edge_length=True,
      add_edge_receiver_coordinates=False)

    # Positional encoder, which encodes the position of the grid and mesh nodes.
    self._positional_encoder_kwargs = dict(
      learnable_fourier_features=self._model_config.learnable_fourier_features,
      num_frequencies=32,
      hidden_dim=32,
      encoding_dim=16,
      remat=self._remat,
      policy=self._policy,
      prevent_cse=self._prevent_cse,
      name="positional_encoder")

    #  Building the encoder and decoder graphs is time-consuming, as it requires a kd-tree search over the mesh or
    #  grid nodes. As there are no Haiku modules instantiated inside the constructor, a GraphCast object can be created
    #  outside a Haiku transform and then serialized. Notice that this requires serializing, among other things,
    #  TypedGraph objects. In the case of graphs, current options are:
    #    1. Extend the TypedGraph class with methods to serialize and deserialize its objects.
    #    2. Register TypedGraph as a pytree and use Orbax.
    #    3. Convert TypedGraph to an object that can be serialized with `graphcast.checkpoint`
    #    4. Use pickle or cloudpickle.
    #  Another use of creating and saving the graphs is for fingerprinting the model (in case of multiple
    #  experiments with different graphs).
    #  When serializing graphs (or the whole model) to avoid repetitive time-consuming tasks, and not
    #  for reproducibility, pickling should be enough.

    # Encoder, which moves data from the grid to the mesh with a single message
    # passing step.
    self._grid2mesh_gnn_kwargs = dict(
        embed_nodes=True,  # Embed raw features of the grid and mesh nodes.
        embed_edges=True,  # Embed raw features of the grid2mesh edges.
        edge_latent_size=dict(grid2mesh=self._model_config.latent_size),
        node_latent_size=dict(
            mesh_nodes=self._model_config.latent_size,
            grid_nodes=self._model_config.latent_size),
        mlp_hidden_size=self._model_config.latent_size,
        mlp_num_hidden_layers=self._model_config.hidden_layers,
        num_message_passing_steps=1,
        use_layer_norm=True,
        include_sent_messages_in_node_update=False,
        activation="swish",
        f32_aggregation=True,
        aggregate_normalization=None,
        remat=self._remat,
        policy=self._policy,
        prevent_cse=self._prevent_cse,
        scan=False,
        name="grid2mesh_gnn")

    # Processor, which performs message passing on the multi-mesh.
    self._mesh_gnn_kwargs = dict(
        embed_nodes=False,  # Node features already embdded by previous layers.
        embed_edges=True,  # Embed raw features of the multi-mesh edges.
        node_latent_size=dict(mesh_nodes=self._model_config.latent_size),
        edge_latent_size=dict(mesh=self._model_config.latent_size),
        mlp_hidden_size=self._model_config.latent_size,
        mlp_num_hidden_layers=self._model_config.hidden_layers,
        num_message_passing_steps=self._model_config.gnn_msg_steps,
        use_layer_norm=True,
        include_sent_messages_in_node_update=False,
        activation="swish",
        f32_aggregation=False,
        remat=self._remat,
        policy=self._policy,
        prevent_cse=self._prevent_cse,
        scan=self._scan,
        name="mesh_gnn")

    num_surface_vars = len(set(self._task_config.target_variables) & set(ALL_SURFACE_VARS))
    assert num_surface_vars == len(set(self._task_config.target_variables) - set(ALL_VOLUME_VARS))
    num_volume_vars = len(set(self._task_config.target_variables) & set(ALL_VOLUME_VARS))
    assert num_volume_vars == len(set(self._task_config.target_variables) - set(ALL_SURFACE_VARS))
    num_outputs = num_surface_vars + len(self._task_config.levels) * num_volume_vars

    # Decoder, which moves data from the mesh back into the grid with a single
    # message passing step.
    self._mesh2grid_gnn_kwargs = dict(
        # Require a specific node dimensionaly for the grid node outputs.
        node_output_size=dict(grid_nodes=num_outputs),
        embed_nodes=False,  # Node features already embdded by previous layers.
        embed_edges=True,  # Embed raw features of the mesh2grid edges.
        edge_latent_size=dict(mesh2grid=self._model_config.latent_size),
        node_latent_size=dict(
            mesh_nodes=self._model_config.latent_size,
            grid_nodes=self._model_config.latent_size),
        mlp_hidden_size=self._model_config.latent_size,
        mlp_num_hidden_layers=self._model_config.hidden_layers,
        num_message_passing_steps=1,
        use_layer_norm=True,
        include_sent_messages_in_node_update=False,
        activation="swish",
        f32_aggregation=False,
        remat=self._remat,
        policy=self._policy,
        prevent_cse=self._prevent_cse,
        scan=False,
        name="mesh2grid_gnn")

    # The `_init_*_properties` methods initialize remaining properties, that is:
    # within `_init_mesh_properties`
    #   self._mesh_graph # MeshGraph
    #   self._boundary_nodes
    #   self._num_mesh_nodes (used also in `__call__`)
    #   self._mesh_nodes # [num_mesh_nodes, 3]
    #   self._mesh_nodes_lat # [num_mesh_nodes]
    #   self._mesh_nodes_lon # [num_mesh_nodes]
    # and within `_init_grid_properties`:
    #   self._grid_mask # [num_lat_points, num_lon_points]
    #   self._grid_lat # [num_lat_points]
    #   self._grid_lon # [num_lon_points]
    #   self._num_grid_nodes # num_lat_points * num_lon_points
    #   self._grid_nodes_lat # [num_grid_nodes]
    #   self._grid_nodes_lon # [num_grid_nodes]
    # The graphs are instantiated within `_init_{grid2mesh,mesh,mesh2grid}_graph` and saved as attributes
    #   self._{grid2mesh,mesh,mesh2grid}_graph_structure

    self._init_mesh_properties()
    self._init_grid_properties()
    self._grid2mesh_graph_structure = self._init_grid2mesh_graph()
    self._mesh_graph_structure = self._init_mesh_graph()
    self._mesh2grid_graph_structure = self._init_mesh2grid_graph()

  def _init_mesh_properties(self):
    """Initializes static properties that have to do with mesh nodes."""
    self._mesh_graph = self._mesh_data.mesh_graph
    self._boundary_nodes_mask = self._mesh_graph.boundary
    self._num_mesh_nodes = self._mesh_graph.vertices.shape[0]

    cartesian2latlon = get_transform(cartesian_srs, equirectangular_srs, pack_back=False)
    mesh_nodes_lon, mesh_nodes_lat = cartesian2latlon(self._mesh_graph.vertices)

    # Convert to f32 to ensure the lat/lon features aren't in f64.
    self._mesh_nodes_lat = mesh_nodes_lat.astype(np.float32)
    self._mesh_nodes_lon = mesh_nodes_lon.astype(np.float32)

  def _init_grid_properties(self):
    """Initializes static properties that have to do with grid nodes"""
    if self._grid_mask is None:
      self._grid_mask = np.ones((self._grid_lat.shape[0], self._grid_lon.shape[0]), dtype=bool)
    assert self._grid_mask.shape == (self._grid_lat.shape[0], self._grid_lon.shape[0])

    self._grid_mask = self._grid_mask.astype(np.bool)
    self._grid_lat = self._grid_lat.astype(np.float32)
    self._grid_lon = self._grid_lon.astype(np.float32)

    # Initialized the counters.
    self._num_grid_nodes = self._grid_lat.shape[0] * self._grid_lon.shape[0]

    # Initialize lat and lon for the grid.
    grid_nodes_lon, grid_nodes_lat = np.meshgrid(self._grid_lon, self._grid_lat)
    self._grid_nodes_lon = grid_nodes_lon.reshape([-1]).astype(np.float32)
    self._grid_nodes_lat = grid_nodes_lat.reshape([-1]).astype(np.float32)

  def _get_grid2mesh_edges(self) -> Tuple[np.ndarray, np.ndarray]:
    # Create some edges according to distance between mesh and grid nodes.
    # NOTICE: grid2mesh connectivity is not influenced by the grid mask.
    assert self._grid_lat is not None and self._grid_lon is not None
    grid_senders, mesh_receivers = mesh_connectivity.radius_query_indices(
      grid_latitude=self._grid_lat,
      grid_longitude=self._grid_lon,
      mesh=self._mesh_graph,
      radius=self._query_radius,
      mask=None,
      workers=self._kdtree_workers)
    return grid_senders, mesh_receivers

  def _get_mesh_edges(self) -> Tuple[np.ndarray, np.ndarray]:
    return self._mesh_graph.edges

  def _get_mesh2grid_edges(self) -> Tuple[np.ndarray, np.ndarray]:
    # Create some edges according to how the grid nodes are contained by mesh triangles.
    mesh_senders, grid_receivers = mesh_connectivity.get_mesh_to_grid_edges(
      grid_latitude=self._grid_lat,
      grid_longitude=self._grid_lon,
      mesh=self._mesh_graph,
      mask=self._grid_mask)
    return mesh_senders, grid_receivers

  def _init_grid2mesh_graph(self) -> typed_graph.TypedGraph:
    """Build Grid2Mesh graph."""

    # Edges sending info from grid to mesh.
    senders, receivers = self._get_grid2mesh_edges()

    # Precompute structural node and edge features according to config options.
    # Structural features are those that depend on the fixed values of the
    # latitude and longitudes of the nodes.
    (senders_node_features, receivers_node_features,
     edge_features) = model_utils.get_bipartite_graph_spatial_features(
      senders_node_lat=self._grid_nodes_lat,
      senders_node_lon=self._grid_nodes_lon,
      receivers_node_lat=self._mesh_nodes_lat,
      receivers_node_lon=self._mesh_nodes_lon,
      senders=senders,
      receivers=receivers,
      edge_normalization=None,
      **self._spatial_features_kwargs,
    )

    # Check that values of n_{grid_node,mesh_node,edge} inferred from feature tensors are correct.
    assert self._num_grid_nodes == senders_node_features.shape[0]
    n_grid_node = np.array([self._num_grid_nodes])
    assert self._num_mesh_nodes == receivers_node_features.shape[0]
    n_mesh_node = np.array([self._num_mesh_nodes])
    assert len(senders) == len(receivers) == edge_features.shape[0]
    n_edge = np.array([len(senders)])
    logger.info(f"Grid2mesh (encoder) graph contains: {n_grid_node[0]} grid nodes "
                f"with {senders_node_features.shape[1]} features, "
                f"{n_mesh_node[0]} mesh nodes with {receivers_node_features.shape[1]} features, "
                f"and {n_edge[0]} edges with {edge_features.shape[1]} features.")
    grid_node_set = typed_graph.NodeSet(
      n_node=n_grid_node, features=senders_node_features)
    mesh_node_set = typed_graph.NodeSet(
      n_node=n_mesh_node, features=receivers_node_features)
    edge_set = typed_graph.EdgeSet(
      n_edge=n_edge,
      indices=typed_graph.EdgesIndices(senders=senders, receivers=receivers),
      features=edge_features)
    nodes = {"grid_nodes": grid_node_set, "mesh_nodes": mesh_node_set}
    edges = {
      typed_graph.EdgeSetKey("grid2mesh", ("grid_nodes", "mesh_nodes")):
        edge_set
    }
    grid2mesh_graph = typed_graph.TypedGraph(
      context=typed_graph.Context(n_graph=np.array([1]), features=()),
      nodes=nodes,
      edges=edges)
    return grid2mesh_graph

  def _init_mesh_graph(self) -> typed_graph.TypedGraph:

    # Possibly get a subset of mesh nodes.
    senders, receivers = self._get_mesh_edges()

    # Precompute structural node and edge features according to config options.
    # Structural features are those that depend on the fixed values of the
    # latitude and longitudes of the nodes.
    assert self._mesh_nodes_lat is not None and self._mesh_nodes_lon is not None
    node_features, edge_features = model_utils.get_graph_spatial_features(
      node_lon=self._mesh_nodes_lon,
      node_lat=self._mesh_nodes_lat,
      senders=senders,
      receivers=receivers,
      **self._spatial_features_kwargs,
    )

    # Check that values of n_{grid_node,mesh_node,edge} inferred from feature tensors are correct.
    assert self._num_mesh_nodes == node_features.shape[0]
    n_mesh_node = np.array([self._num_mesh_nodes])
    assert len(senders) == len(receivers) == edge_features.shape[0]
    n_edge = np.array([len(senders)])
    assert n_mesh_node == len(node_features)
    logger.info(f"Mesh (processor) graph contains: {n_mesh_node[0]} mesh nodes "
                f"with {node_features.shape[1]} features, "
                f"and {n_edge[0]} edges with {edge_features.shape[1]} features.")
    mesh_node_set = typed_graph.NodeSet(
      n_node=n_mesh_node, features=node_features)
    edge_set = typed_graph.EdgeSet(
      n_edge=n_edge,
      indices=typed_graph.EdgesIndices(senders=senders, receivers=receivers),
      features=edge_features)
    nodes = {"mesh_nodes": mesh_node_set}
    edges = {
      typed_graph.EdgeSetKey("mesh", ("mesh_nodes", "mesh_nodes")): edge_set
    }
    mesh_graph = typed_graph.TypedGraph(
      context=typed_graph.Context(n_graph=np.array([1]), features=()),
      nodes=nodes,
      edges=edges)

    return mesh_graph

  def _init_mesh2grid_graph(self) -> typed_graph.TypedGraph:
    """Build Mesh2Grid graph."""

    # Create some edges according to how the grid nodes are contained by
    # mesh triangles.
    senders, receivers = self._get_mesh2grid_edges()

    # Precompute structural node and edge features according to config options.
    assert self._mesh_nodes_lat is not None and self._mesh_nodes_lon is not None
    (senders_node_features, receivers_node_features,
     edge_features) = model_utils.get_bipartite_graph_spatial_features(
      senders_node_lat=self._mesh_nodes_lat,
      senders_node_lon=self._mesh_nodes_lon,
      receivers_node_lat=self._grid_nodes_lat,
      receivers_node_lon=self._grid_nodes_lon,
      senders=senders,
      receivers=receivers,
      edge_normalization=self._mesh2grid_edge_normalization_factor,
      **self._spatial_features_kwargs,
    )

    # Check that values of n_{grid_node,mesh_node,edge} inferred from feature tensors are correct.
    assert self._num_mesh_nodes == senders_node_features.shape[0]
    n_mesh_node = np.array([self._num_mesh_nodes])
    assert self._num_grid_nodes == receivers_node_features.shape[0]
    n_grid_node = np.array([self._num_grid_nodes])
    assert len(senders) == len(receivers) == edge_features.shape[0]
    n_edge = np.array([len(senders)])
    logger.info(f"Mesh2grid (decoder) graph contains: {n_mesh_node[0]} mesh nodes "
                f"with {senders_node_features.shape[1]} features, "
                f"{n_grid_node[0]} grid nodes with {receivers_node_features.shape[1]} features, "
                f"and {n_edge[0]} edges with {edge_features.shape[1]} features.")
    grid_node_set = typed_graph.NodeSet(
      n_node=n_grid_node, features=receivers_node_features)
    mesh_node_set = typed_graph.NodeSet(
      n_node=n_mesh_node, features=senders_node_features)
    edge_set = typed_graph.EdgeSet(
      n_edge=n_edge,
      indices=typed_graph.EdgesIndices(senders=senders, receivers=receivers),
      features=edge_features)
    nodes = {"grid_nodes": grid_node_set, "mesh_nodes": mesh_node_set}
    edges = {
      typed_graph.EdgeSetKey("mesh2grid", ("mesh_nodes", "grid_nodes")):
        edge_set
    }
    mesh2grid_graph = typed_graph.TypedGraph(
      context=typed_graph.Context(n_graph=np.array([1]), features=()),
      nodes=nodes,
      edges=edges)
    return mesh2grid_graph

  def __call__(self,
               inputs: xarray.Dataset,
               targets_template: xarray.Dataset,
               forcings: xarray.Dataset,
               is_training: bool = False,
               ) -> xarray.Dataset:
    # Convert all input data into flat vectors for each of the grid nodes.
    # xarray (batch, time, lat, lon, level, multiple vars, forcings)
    # -> [num_grid_nodes, batch, num_channels]
    grid_node_features = self._inputs_to_grid_node_features(inputs, forcings)

    # Transfer data for the grid to the mesh,
    # [num_mesh_nodes, batch, latent_size], [num_grid_nodes, batch, latent_size]
    latent_mesh_nodes, latent_grid_nodes = self._run_grid2mesh_gnn(grid_node_features)

    # Run message passing in the multimesh.
    # [num_mesh_nodes, batch, latent_size]
    updated_latent_mesh_nodes = self._run_mesh_gnn(latent_mesh_nodes)

    # Transfer data from the mesh to the grid.
    # [num_grid_nodes, batch, output_size]
    output_grid_nodes = self._run_mesh2grid_gnn(updated_latent_mesh_nodes, latent_grid_nodes)

    # Convert output flat vectors for the grid nodes to the format of the output.
    # [num_grid_nodes, batch, output_size] ->
    # xarray (batch, one time step, lat, lon, level, multiple vars)
    return self._grid_node_outputs_to_prediction(output_grid_nodes, targets_template)

  def loss_and_predictions(  # pytype: disable=signature-mismatch  # jax-ndarray
      self,
      inputs: xarray.Dataset,
      targets: xarray.Dataset,
      forcings: xarray.Dataset,
      **kwargs,
      ) -> tuple[predictor_base.LossAndDiagnostics, xarray.Dataset]:
    # Forward pass.
    predictions = self(
        inputs, targets_template=targets, forcings=forcings, is_training=True)
    # Compute loss.
    loss = losses.weighted_mse(
        predictions, targets,
        per_variable_weights=self._per_variable_weights,
        levels_normalization_coord='depth',
        weights_decreasing_with_level=True,
        **kwargs)
    return loss, predictions  # pytype: disable=bad-return-type  # jax-ndarray

  def loss(  # pytype: disable=signature-mismatch  # jax-ndarray
      self,
      inputs: xarray.Dataset,
      targets: xarray.Dataset,
      forcings: xarray.Dataset,
      **kwargs,
      ) -> predictor_base.LossAndDiagnostics:
    loss, _ = self.loss_and_predictions(inputs, targets, forcings, **kwargs)
    return loss  # pytype: disable=bad-return-type  # jax-ndarray

  def _run_grid2mesh_gnn(self, grid_node_input_features: chex.Array,
                         ) -> tuple[chex.Array, chex.Array]:
    """Runs the grid2mesh_gnn, extracting latent mesh and grid nodes."""
    # Concatenate node structural features with input features.
    batch_size = grid_node_input_features.shape[1]

    grid2mesh_graph = self._grid2mesh_graph_structure
    assert grid2mesh_graph is not None
    grid_nodes = grid2mesh_graph.nodes["grid_nodes"]
    mesh_nodes = grid2mesh_graph.nodes["mesh_nodes"]

    # Compute positional encodings and add batch dimension.
    positional_encoder = PositionalEncoder(**self._positional_encoder_kwargs)
    grid_node_position_encodings = positional_encoder(grid_nodes.features.astype(grid_node_input_features.dtype))
    grid_node_position_encodings = _add_batch_second_axis(grid_node_position_encodings, batch_size)
    mesh_node_position_encodings = positional_encoder(mesh_nodes.features.astype(grid_node_input_features.dtype))
    mesh_node_position_encodings = _add_batch_second_axis(mesh_node_position_encodings, batch_size)

    # Add boundary information to mesh nodes.
    if self._boundary_nodes_mask is not None:
      # noinspection PyUnresolvedReferences
      mesh_node_boundary_mask = self._boundary_nodes_mask[..., None].astype(grid_node_input_features.dtype)
      mesh_node_boundary_mask = _add_batch_second_axis(mesh_node_boundary_mask, batch_size)
      mesh_node_position_encodings = jnp.concatenate([mesh_node_position_encodings, mesh_node_boundary_mask], axis=-1)

    # Concatenate input features with positional encodings for grid nodes
    grid_node_features = jnp.concatenate([grid_node_input_features, grid_node_position_encodings], axis=-1)

    # TODO: this comment from the original GraphCast code should be checked.
    #  "To make sure capacity of the embedded is identical for the grid nodes and
    #  the mesh nodes, we also append some dummy zero input features for the
    #  mesh nodes."
    #  In doubt, we add dummy features for the mesh nodes here as done originally.
    assert grid_node_input_features.shape[-1] > 0
    dummy_mesh_node_shape = ((self._num_mesh_nodes,) +
                             grid_node_input_features.shape[1:-1] +
                             (grid_node_input_features.shape[-1] - (1 if self._boundary_nodes_mask is not None
                                                                    else 0),))
    dummy_mesh_node_features = jnp.zeros(dummy_mesh_node_shape, dtype=grid_node_input_features.dtype)
    mesh_node_features = jnp.concatenate([dummy_mesh_node_features, mesh_node_position_encodings], axis=-1)
    assert mesh_node_features.shape[1:] == grid_node_features.shape[1:]

    new_grid_nodes = grid_nodes._replace(features=grid_node_features)
    new_mesh_nodes = mesh_nodes._replace(features=mesh_node_features)

    # Broadcast edge structural features to the required batch size.
    grid2mesh_edges_key = grid2mesh_graph.edge_key_by_name("grid2mesh")
    edges = grid2mesh_graph.edges[grid2mesh_edges_key]
    edge_features = _add_batch_second_axis(edges.features.astype(dummy_mesh_node_features.dtype), batch_size)
    new_edges = edges._replace(features=edge_features)

    input_graph = self._grid2mesh_graph_structure._replace(
        edges={grid2mesh_edges_key: new_edges},
        nodes={
            "grid_nodes": new_grid_nodes,
            "mesh_nodes": new_mesh_nodes
        })
    # Create the GNN.
    grid2mesh_gnn = deep_typed_graph_net.DeepTypedGraphNet(**self._grid2mesh_gnn_kwargs)
    # Run the GNN.
    grid2mesh_out = grid2mesh_gnn(input_graph)
    latent_mesh_nodes = grid2mesh_out.nodes["mesh_nodes"].features
    latent_grid_nodes = grid2mesh_out.nodes["grid_nodes"].features
    return latent_mesh_nodes, latent_grid_nodes

  def _run_mesh_gnn(self, latent_mesh_nodes: chex.Array) -> chex.Array:
    """Runs the mesh_gnn, extracting updated latent mesh nodes."""

    # Add the structural edge features of this graph. Note we don't need
    # to add the structural node features, because these are already part of
    # the latent state, via the original Grid2Mesh gnn, however, we need
    # the edge ones, because it is the first time we are seeing this particular
    # set of edges.
    batch_size = latent_mesh_nodes.shape[1]

    mesh_graph = self._mesh_graph_structure
    assert mesh_graph is not None
    mesh_edges_key = mesh_graph.edge_key_by_name("mesh")
    edges = mesh_graph.edges[mesh_edges_key]

    # We are assuming here that the mesh gnn uses a single set of edge keys
    # named "mesh" for the edges and that it uses a single set of nodes named
    # "mesh_nodes"
    msg = ("The setup currently requires to only have one kind of edge in the"
           " mesh GNN.")
    assert len(mesh_graph.edges) == 1, msg

    edge_features = _add_batch_second_axis(edges.features.astype(latent_mesh_nodes.dtype), batch_size)
    new_edges = edges._replace(features=edge_features)

    nodes = mesh_graph.nodes["mesh_nodes"]
    nodes = nodes._replace(features=latent_mesh_nodes)

    input_graph = mesh_graph._replace(
        edges={mesh_edges_key: new_edges}, nodes={"mesh_nodes": nodes})
    # Create the GNN.
    mesh_gnn = deep_typed_graph_net.DeepTypedGraphNet(**self._mesh_gnn_kwargs)
    # Run the GNN.
    mesh_out = mesh_gnn(input_graph)
    return mesh_out.nodes["mesh_nodes"].features

  def _run_mesh2grid_gnn(self,
                         updated_latent_mesh_nodes: chex.Array,
                         latent_grid_nodes: chex.Array,
                         ) -> chex.Array:
    """Runs the mesh2grid_gnn, extracting the output grid nodes."""

    # Add the structural edge features of this graph. Note we don't need
    # to add the structural node features, because these are already part of
    # the latent state, via the original Grid2Mesh gnn, however, we need
    # the edge ones, because it is the first time we are seeing this particular
    # set of edges.
    batch_size = updated_latent_mesh_nodes.shape[1]

    mesh2grid_graph = self._mesh2grid_graph_structure
    assert mesh2grid_graph is not None
    mesh_nodes = mesh2grid_graph.nodes["mesh_nodes"]
    grid_nodes = mesh2grid_graph.nodes["grid_nodes"]
    new_mesh_nodes = mesh_nodes._replace(features=updated_latent_mesh_nodes)
    new_grid_nodes = grid_nodes._replace(features=latent_grid_nodes)
    mesh2grid_key = mesh2grid_graph.edge_key_by_name("mesh2grid")
    edges = mesh2grid_graph.edges[mesh2grid_key]

    edge_features = _add_batch_second_axis(edges.features.astype(latent_grid_nodes.dtype), batch_size)
    new_edges = edges._replace(features=edge_features)

    input_graph = mesh2grid_graph._replace(
        edges={mesh2grid_key: new_edges},
        nodes={
            "mesh_nodes": new_mesh_nodes,
            "grid_nodes": new_grid_nodes
        })
    # Create the GNN.
    mesh2grid_gnn = deep_typed_graph_net.DeepTypedGraphNet(**self._mesh2grid_gnn_kwargs)
    # Run the GNN.
    mesh2grid_out = mesh2grid_gnn(input_graph)
    output_grid_nodes = mesh2grid_out.nodes["grid_nodes"].features

    return output_grid_nodes

  def _inputs_to_grid_node_features(
      self,
      inputs: xarray.Dataset,
      forcings: xarray.Dataset,
      ) -> chex.Array:
    """xarrays -> [num_grid_nodes, batch, num_channels]."""

    # xarray `Dataset` (batch, time, lat, lon, level, multiple vars)
    # to xarray `DataArray` (batch, lat, lon, channels)
    stacked_inputs = model_utils.dataset_to_stacked(inputs)
    stacked_forcings = model_utils.dataset_to_stacked(forcings)
    stacked_inputs = xarray.concat(
        [stacked_inputs, stacked_forcings], dim="channels")

    # xarray `DataArray` (batch, lat, lon, channels)
    # to single numpy array with shape [lat_lon_node, batch, channels]
    grid_xarray_lat_lon_leading = model_utils.lat_lon_to_leading_axes(
        stacked_inputs)
    grid_node_features = xarray_jax.unwrap(grid_xarray_lat_lon_leading.data).reshape(
        (-1,) + grid_xarray_lat_lon_leading.data.shape[2:])
    return grid_node_features

  def _grid_node_outputs_to_prediction(
      self,
      grid_node_outputs: chex.Array,
      targets_template: xarray.Dataset,
      ) -> xarray.Dataset:
    """[num_grid_nodes, batch, num_outputs] -> xarray."""

    # numpy array with shape [lat_lon_node, batch, channels]
    # to xarray `DataArray` (batch, lat, lon, channels)
    assert self._grid_lat is not None and self._grid_lon is not None
    grid_shape = (self._grid_lat.shape[0], self._grid_lon.shape[0])
    grid_node_outputs = grid_node_outputs[:self._num_grid_nodes, ...]
    grid_outputs_lat_lon_leading = jax.lax.reshape(grid_node_outputs, grid_shape + grid_node_outputs.shape[1:])
    dims = ("lat", "lon", "batch", "channels")
    grid_xarray_lat_lon_leading = xarray_jax.DataArray(
        data=grid_outputs_lat_lon_leading,
        dims=dims)
    grid_xarray = model_utils.restore_leading_axes(grid_xarray_lat_lon_leading)

    # xarray `DataArray` (batch, lat, lon, channels)
    # to xarray `Dataset` (batch, one time step, lat, lon, level, multiple vars)
    return model_utils.stacked_to_dataset(
        grid_xarray.variable, targets_template)


def _add_batch_second_axis(data, batch_size):
  # data [leading_dim, trailing_dim]
  assert data.ndim == 2
  ones = jnp.ones([batch_size, 1], dtype=data.dtype)
  return data[:, None] * ones  # [leading_dim, batch, trailing_dim]


def _get_max_edge_distance(mesh: TriangleMesh | MeshGraph):
  if isinstance(mesh, TriangleMesh):
    senders, receivers = faces_to_edges(mesh.faces)
  else:
    senders, receivers = mesh.edges
  edge_distances = np.linalg.norm(
      mesh.vertices[senders] - mesh.vertices[receivers], axis=-1)
  # Notice: if edge_distances is empty, the following will raise an error.
  return edge_distances.max()
