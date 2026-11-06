# Copyright 2026 Stefano Campanella.
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
import inspect
import logging
import os
import pathlib
import shutil
import typing
from collections.abc import Callable, Mapping
from typing import Any

import haiku as hk
import jax
import numpy as np
import optax
import xarray as xr
from jax import checkpoint_policies as cp
from jax.experimental import mesh_utils
from jax.experimental.multihost_utils import sync_global_devices
from jax.sharding import AxisType, Mesh
from jax.sharding import PartitionSpec as P

from graphcast import xarray_jax
from graphcast.casting import Bfloat16Cast
from graphcast.checkpoint import dump as ckpt_dump
from graphcast.dataloader import InputsTargetsForcings
from graphcast.geospatial_mesh_utils import read_mesh_data
from graphcast.mask import Mask
from graphcast.mesh_graph import MeshData
from graphcast.model import CheckPoint, GraphCast, ModelConfig, TaskConfig
from graphcast.normalization import InputNormalization, OutputNormalization
from graphcast.predictor_base import Predictor
from graphcast.training_config import TrainingConfig

logger = logging.getLogger(__name__)

Datasets = tuple[xr.Dataset, ...]
MaybeDatasets = tuple[Datasets | None, ...]
DatasetsOrDataArrays = tuple[xr.Dataset | xr.DataArray, ...]
Paths = tuple[pathlib.Path, ...]
Params = hk.Params | hk.MutableParams | dict[str, Any]
JAXLossAndDiagnostics = tuple[jax.Array, Mapping[str, jax.Array]]


def _make_hybrid_mesh(
  axis_names: tuple[str, ...],
  local_shape: tuple[int, ...],
  process_shape: tuple[int, ...],
) -> Mesh:
  # `jax.make_mesh` rejects meshes spanning several GPU nodes, which JAX treats as separate slices. Lay out each
  # process's devices by local_shape and the processes by process_shape instead, so that axis i has size
  # local_shape[i] * process_shape[i].
  devices = mesh_utils.create_hybrid_device_mesh(local_shape, process_shape, process_is_granule=True)
  return Mesh(devices, axis_names, axis_types=(AxisType.Explicit,) * len(axis_names))


def make_batch_mesh() -> Mesh:
  """A one-dimensional mesh over all devices, with axis "batch"."""
  return _make_hybrid_mesh(("batch",), (jax.local_device_count(),), (jax.process_count(),))


def check_writable_paths(
  output_path: pathlib.Path,
  checkpoint_dir: pathlib.Path,
  tensorboard_dir: pathlib.Path | None,
  start_fresh: bool = False,
  overwrite: bool = False,
) -> None:
  """Check that the model file, the checkpoint directory and the TensorBoard directory can be written.

  Creates missing directories, and empties the checkpoint directory if start_fresh.
  """

  def ensure_directory_exists(directory: pathlib.Path):
    if jax.process_index() == 0 and not directory.exists():
      directory.mkdir(parents=True, exist_ok=True)
    sync_global_devices(f"ensure_{directory}_exists")

  def empty_directory(directory: pathlib.Path):
    logger.info(f"Emptying {directory}")
    if jax.process_index() == 0:
      if directory.exists():
        shutil.rmtree(directory)
      directory.mkdir(parents=True, exist_ok=True)
    sync_global_devices(f"empty_{directory}")

  # The final model is written to output_path. Its parent directory is created if missing. With overwrite, an
  # existing file must be writable, otherwise the parent directory must be; without overwrite, it must not exist.
  ensure_directory_exists(output_path.parent)
  if overwrite:
    if output_path.exists():
      if not os.access(output_path, os.W_OK):
        raise PermissionError(f"{output_path} is not writable.")
    elif not os.access(output_path.parent, os.W_OK):
      raise PermissionError(f"{output_path.parent} is not writable.")
  elif output_path.exists():
    raise FileExistsError(f"{output_path} already exists.")

  # Checkpoints are saved to and restored from checkpoint_dir, which must be readable and writable.
  ensure_directory_exists(checkpoint_dir)
  if start_fresh and any(checkpoint_dir.iterdir()):
    empty_directory(checkpoint_dir)
  if not os.access(checkpoint_dir, os.R_OK | os.W_OK):
    raise PermissionError(f"{checkpoint_dir} is not readable and/or writable.")

  # TensorBoard logs go to tensorboard_dir, which must be writable.
  if tensorboard_dir is not None:
    ensure_directory_exists(tensorboard_dir)
    if not os.access(tensorboard_dir, os.W_OK):
      raise PermissionError(f"{tensorboard_dir} is not writable.")


def get_mesh(configs: TrainingConfig) -> MeshData:
  mesh_data_path = configs.mesh.path
  logger.info(f"Loading mesh from {mesh_data_path}")
  mesh_data = read_mesh_data(
    mesh_data_path,
    mesh_size_tag_name=configs.mesh.mesh_size_tag_name,
    mesh_size_tag_step=configs.mesh.step,
  )
  return mesh_data


def get_model_config(configs: TrainingConfig) -> ModelConfig:
  return ModelConfig(**configs.model.model_dump())


def get_policy(configs: TrainingConfig) -> Callable:
  return cp.save_and_offload_only_these_names(
    **configs.policy.model_dump(exclude={"remat_loss", "prevent_cse", "scan", "edge_chunk_size"})
  )


def get_predictor(
  configs: TrainingConfig,
  model_config: ModelConfig,
  task_config: TaskConfig,
  mesh_data: MeshData,
  grid_lat: np.ndarray,
  grid_lon: np.ndarray,
  grid_mask: np.ndarray,
  mean_by_level: xr.Dataset | None,
  stddev_by_level: xr.Dataset | None,
  loss_masks: xr.Dataset,
  diffs_stddev_by_level: xr.Dataset | None = None,
) -> Predictor:
  """Wrap GraphCast, outermost first: Mask (per-variable loss masks) -> OutputNormalization (residuals divided by
  diffs_stddev_by_level) -> InputNormalization (only with mean_by_level or stddev_by_level) -> Bfloat16Cast."""
  # Deeper one-step predictor.
  predictor = GraphCast(
    _model_config=model_config,
    _task_config=task_config,
    _grid_lat=grid_lat,
    _grid_lon=grid_lon,
    _grid_mask=grid_mask,
    _mesh_data=mesh_data,
    _scan=configs.policy.scan,
    _remat=True,
    _policy=get_policy(configs),
    _prevent_cse=configs.policy.prevent_cse,
    _edge_chunk_size=configs.policy.edge_chunk_size,
  )

  # Modify inputs/outputs to `graphcast.GraphCast` to handle conversion to from/to float32 to/from BFloat16.
  predictor = Bfloat16Cast(predictor)

  # Normalization wraps `casting.Bfloat16Cast`, so the casting to/from BFloat16 happens on normalized values. The
  # standardized anomaly store needs no input normalization: inputs are normalized only if statistics are given.
  if mean_by_level is not None or stddev_by_level is not None:
    predictor = InputNormalization(
      predictor, scales=stddev_by_level, locations=mean_by_level, skip_names=configs.normalization.skip_names
    )
  predictor = OutputNormalization(
    predictor, residual_scales=diffs_stddev_by_level, scales=stddev_by_level, locations=mean_by_level
  )

  # Mask inputs/outputs. Notice, other not finite values (i.e., inf) are not filled with mask.fill_value.
  predictor = Mask(predictor, mask=loss_masks, value=configs.mask.fill_value)

  return predictor


def get_optimizer(
  configs: TrainingConfig,
) -> tuple[optax.GradientTransformationExtraArgs, optax.Schedule]:
  logger.info(f"Optimizing with {configs.optimizer}")
  schedule_config = configs.optimizer.schedule
  try:
    scheduler_init = getattr(optax.schedules, schedule_config.name)
    scheduler = scheduler_init(**schedule_config.model_dump(exclude={"name"}, exclude_unset=True))
  except (AttributeError, TypeError, ValueError) as error:
    raise ValueError(f"optimizer.schedule ({schedule_config.name}): {error}") from error

  gradient_transformations = []
  for index, config in enumerate(configs.optimizer.gradient_transformation):
    try:
      gradient_transformation_init = getattr(optax, config.name)
      kwargs = config.model_dump(exclude={"name"}, exclude_unset=True)
      if "learning_rate" in inspect.signature(gradient_transformation_init).parameters:
        kwargs["learning_rate"] = scheduler
      gradient_transformation = gradient_transformation_init(**kwargs)
    except (AttributeError, TypeError, ValueError) as error:
      raise ValueError(f"optimizer.gradient_transformation.{index} ({config.name}): {error}") from error
    gradient_transformations.append(gradient_transformation)
  optimizer = optax.chain(*gradient_transformations)

  return optimizer, scheduler


def get_params(
  init_fn: Callable[..., Params],
  sample: InputsTargetsForcings,
  configs: TrainingConfig,
  seed: int | None = None,
) -> Params:
  """Initialize the parameters from a sample (dataloader_utils.get_init_sample): only its shapes matter."""
  seed = configs.seed if seed is None else seed
  rng_key = jax.random.key(seed)
  logger.info(f"Initializing parameters with {seed=}")
  return init_fn(rng_key, xarray_jax.device_put(sample))


class TensorboardLogger:
  def __init__(self, tb_path: pathlib.Path):
    # Tensorflow should be imported after jax initialization, see: https://github.com/google/flax/issues/4942
    if not jax.distributed.is_initialized():
      raise ValueError(
        "JAX distributed should be initialized beforehand, see: https://github.com/google/flax/issues/4942."
      )
    else:
      _ = jax.devices()
      import tensorflow as tf

      # TODO: could this be set using environment variables?
      tf.config.experimental.set_visible_devices([], "GPU")
      from tensorflow import summary

      self._summary = summary
      self._summary_writer = summary.create_file_writer(str(tb_path))

  def log(
    self,
    current_step: int,
    train_metrics: JAXLossAndDiagnostics,
    lr: float | None = None,
    prefix: str = "train",
  ) -> None:
    """Write the metrics of a step on process 0 (other processes do nothing, and wait for nothing)."""
    if jax.process_index() != 0:
      return
    loss, diagnostics = jax.device_get(train_metrics)
    with self._summary_writer.as_default():
      self._summary.scalar(f"{prefix}/loss", loss, step=current_step)
      for key, value in diagnostics.items():
        self._summary.scalar(f"{prefix}/{key}", value, step=current_step)
      if lr is not None:
        self._summary.scalar("learning_rate", lr, step=current_step)


def save_model(
  output_path: pathlib.Path,
  configs: TrainingConfig,
  model_config: ModelConfig,
  task_config: TaskConfig,
  grid_lat: np.ndarray,
  grid_lon: np.ndarray,
  grid_mask: np.ndarray,
  mesh_data: MeshData,
  params: Params,
) -> None:
  graphcast_ckpt = CheckPoint(
    model_config=model_config,
    task_config=task_config,
    mesh_data=mesh_data,
    grid_lat=grid_lat,
    grid_lon=grid_lon,
    grid_mask=grid_mask,
    description=configs.description,
    license=configs.license,
    params=params,
  )

  if jax.process_index() == 0:
    logger.info(f"Saving the model to {output_path}")
    with output_path.open("wb") as file:
      file = typing.cast(typing.BinaryIO, file)
      ckpt_dump(file, graphcast_ckpt)
  sync_global_devices("save_model")


# Per-device rng keys in data parallelism. The model is deterministic today, but a probabilistic variant will need a
# key in the predictor and the loss, different on every device (each holds different samples of the batch). The
# fold-in trick of the UvA tutorial [1], `fold_in(key, axis_index("batch"))`, works in shard_map on our mesh with
# Explicit axes only if the key is an argument of shard_map (spec P(), the same key on every device, folded inside).
# Closing over a key computed outside shard_map fails with
#
#   NotImplementedError: Closing over inputs to shard_map where the input is sharded on `Explicit` axes is not
#   implemented. As a workaround, please pass those inputs as an argument to shard_map.
#   Got input with shape key<fry>[]({Explicit: ('batch',)})
#
# (the error that kept the previous helper from passing keys, see [4]). Splitting the key into one per device and
# sharding it along the batch axis also works, once resharded explicitly
# (`jax.reshard(jax.random.split(key, n), P("batch"))`); without it, shard_map rejects the unsharded keys. On rng keys
# and sharding, see also [2] and [3].
#
# [1] https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/scaling/JAX/data_parallel_fsdp.html
# [2] https://github.com/jax-ml/jax/discussions/22862
# [3] https://github.com/jax-ml/jax/issues/22860
# [4] https://github.com/jax-ml/jax/issues/29162
def data_parallel(
  fn: Callable[..., Any], mesh: Mesh, batch_axis: str = "batch", check_vma: bool = False
) -> Callable[..., Any]:
  """Maps `fn(params, rng, data, static_data)` over the shards of `data` along the `batch_axis` of `mesh`.

  `params`, `rng` and `static_data` are passed whole to every device; each device folds its index along the batch
  axis into `rng`, so the devices get different keys. Results are concatenated along their leading axis.
  """

  def local_fn(params, rng, data, static_data):
    return fn(params, jax.random.fold_in(rng, jax.lax.axis_index(batch_axis)), data, static_data)

  return jax.shard_map(
    local_fn, mesh=mesh, in_specs=(P(), P(), P(batch_axis), None), out_specs=P(batch_axis), check_vma=check_vma
  )
