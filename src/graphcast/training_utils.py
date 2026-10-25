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
from grain.python import DatasetIterator
from jax import checkpoint_policies as cp
from jax.experimental import mesh_utils
from jax.experimental.multihost_utils import sync_global_devices
from jax.experimental.shard_map import shard_map
from jax.sharding import AxisType, Mesh
from jax.sharding import PartitionSpec as P

from graphcast import dataloader_utils, xarray_jax
from graphcast.casting import Bfloat16Cast
from graphcast.checkpoint import dump as ckpt_dump
from graphcast.dataloader import InputsTargetsForcings
from graphcast.geospatial_mesh_utils import read_mesh_data
from graphcast.mask import Mask
from graphcast.mesh_graph import MeshData
from graphcast.model import CheckPoint, GraphCast, ModelConfig, TaskConfig
from graphcast.normalization import InputsAndResiduals
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
  devices = mesh_utils.create_hybrid_device_mesh(
    local_shape, process_shape, process_is_granule=True
  )
  return Mesh(devices, axis_names, axis_types=(AxisType.Explicit,) * len(axis_names))


def make_batch_mesh() -> Mesh:
  """A one-dimensional mesh over all devices, with axis "batch"."""
  return _make_hybrid_mesh(
    ("batch",), (jax.local_device_count(),), (jax.process_count(),)
  )


def make_process_mesh() -> Mesh:
  """A mesh with axes ("process", "local_device"), for Orbax and multihost barriers (see get_checkpoint_manager)."""
  return _make_hybrid_mesh(
    ("process", "local_device"),
    (1, jax.local_device_count()),
    (jax.process_count(), 1),
  )


def check_writable_paths(
    output: pathlib.Path,
    train: pathlib.Path,
    tb: pathlib.Path | None,
  start_fresh: bool = False,
  overwrite: bool = False,
) -> None:
  """Check preconditions for path-arguments to write to, safely creating and deleting directories if necessary."""

  def ensure_directory_exists(path: pathlib.Path):
    if jax.process_index() == 0 and not path.exists():
      path.mkdir(parents=True, exist_ok=True)
    sync_global_devices(f"ensure_{path}_exists")

  def empty_directory(path: pathlib.Path):
    logger.info(f"Emptying {path}")
    if jax.process_index() == 0:
      if path.exists():
        shutil.rmtree(path)
      path.mkdir(parents=True, exist_ok=True)
    sync_global_devices(f"empty_{path}")

  mesh = make_process_mesh()
  with jax.sharding.set_mesh(mesh):
    # output is where the final checkpoint will be saved.
    # We ensure that output' parent directory exists (with write permission if created). Also:
    #   1. If overwrite=True, and output points to an existing file, then it should be writable
    #   2. If overwrite=True, and output doesn't point to an existing file, then the parent directory should be writable.
    #   3. If overwrite=False, then output should not point to an existing file.
    ensure_directory_exists(output.parent)
    if overwrite:
      if output.exists():
        if not os.access(output, os.W_OK):
          raise PermissionError(f"{output} is not writable.")
      else:
        if not os.access(output.parent, os.W_OK):
          raise PermissionError(f"{output.parent} is not writable.")
    else:
      if output.exists():
        raise FileExistsError(f"{output} already exists.")

    # train is the directory where checkpoints will be saved during training.
    # We ensure that the directory exists (with write permission if created) and is empty if start_fresh=True, also:
    #   1. It should be readable and writable.
    ensure_directory_exists(train)
    if start_fresh and any(train.iterdir()):
      empty_directory(train)
    if not os.access(train, os.R_OK | os.W_OK):
      raise PermissionError(f"{train} is not readable and/or writable.")

    # tb is where the summarywriter will write the logs during training:
    # We ensure that the directory exists (with write permission if created).
    #   1. It should be writable.
    if tb is not None:
      ensure_directory_exists(tb)
      if not os.access(tb, os.W_OK):
        raise PermissionError(f"{tb} is not writable.")


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
  return cp.save_and_offload_only_these_names(**configs.policy.model_dump(exclude={"remat_loss", "prevent_cse", "scan", "remat_mlps", "edge_chunk_size"}))


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
  mask_da: xr.DataArray,
    diffs_stddev_by_level: xr.Dataset | None = None,
) -> Predictor:
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
    _remat_mlps=configs.policy.remat_mlps,
    _edge_chunk_size=configs.policy.edge_chunk_size,
  )

  # Modify inputs/outputs to `graphcast.GraphCast` to handle conversion to from/to float32 to/from BFloat16.
  predictor = Bfloat16Cast(predictor)

  # Modify inputs/outputs to `casting.Bfloat16Cast` so the casting to/from BFloat16 happens after applying
  # normalization to the inputs/targets.
  predictor = InputsAndResiduals(
    predictor,
    mean_by_level=mean_by_level,
    stddev_by_level=stddev_by_level,
    diffs_stddev_by_level=diffs_stddev_by_level,
    skip_names=configs.normalization.skip_names,
  )

  # Mask inputs/outputs. Notice, other not finite values (i.e., inf) are not filled with mask.fill_value.
  predictor = Mask(predictor, mask=mask_da, value=configs.mask.fill_value)

  return predictor


def get_optimizer(
    configs: TrainingConfig,
) -> tuple[optax.GradientTransformationExtraArgs, optax.Schedule]:
  logger.info(f"Optimizing with {configs.optimizer}")
  schedule_config = configs.optimizer.schedule
  try:
    scheduler_init = getattr(optax.schedules, schedule_config.name)
    scheduler = scheduler_init(
      **schedule_config.model_dump(exclude={"name"}, exclude_unset=True)
    )
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
      raise ValueError(
        f"optimizer.gradient_transformation.{index} ({config.name}): {error}"
      ) from error
    gradient_transformations.append(gradient_transformation)
  optimizer = optax.chain(*gradient_transformations)

  return optimizer, scheduler


def get_params(
  init_fn: Callable[..., Params],
  iterator: DatasetIterator[InputsTargetsForcings],
    configs: TrainingConfig,
  seed: int | None = None,
) -> Params:
  seed = configs.seed if seed is None else seed
  rng_key = jax.random.key(seed)
  sample_on_host = dataloader_utils.get_first_sample_and_reset(iterator)
  # FIXME: avoid grain workers failing (harmless, but produce lousy logs)
  sample = xarray_jax.device_put(sample_on_host)
  logger.info(f"Initializing parameters with {seed=}")
  return init_fn(rng_key, sample)


class TensorboardLogger:
  def __init__(self, tb_path: pathlib.Path):
    # Tensorflow should be imported after jax initialization, see: https://github.com/google/flax/issues/4942
    if not jax.distributed.is_initialized():
      raise ValueError(
        "JAX distributed should be initialized beforehand, "
        "see: https://github.com/google/flax/issues/4942."
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
  ) -> None:
    with self._summary_writer.as_default():

      def _log(key, value):
        if jax.process_index() == 0:
          self._summary.scalar(key, value, step=current_step)

      loss, diagnostics = jax.tree_util.tree_map(lambda x: x.item(), train_metrics)
      _log("train/loss", loss)
      for key, value in diagnostics.items():
        _log(f"train/{key}", value)

      if lr is not None:
        _log("learning_rate", lr)


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

  device_mesh = make_process_mesh()
  with jax.sharding.set_mesh(device_mesh):
    if jax.process_index() == 0:
      logger.info(f"Saving checkpoint to {output_path}")
      with output_path.open("wb") as file:
        file = typing.cast(typing.BinaryIO, file)
        ckpt_dump(file, graphcast_ckpt)
    sync_global_devices("save_checkpoint")


# get_global_grad_fn supports an apply function which depends on PRNGkeys (after a haiku.transform).
# However, in the case of FSDS, there is the need to have a different rng key for each element of the batch.
# However, it turns out to be non-trivial. See:
# https://github.com/jax-ml/jax/discussions/22862
# https://github.com/jax-ml/jax/issues/22860
# Fold-in trick taken from
# https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/scaling/JAX/data_parallel_fsdp.html
# might not work when using a global mesh, and produce errors like the following.
#
#    File ".../jax/_src/prng.py", line 628, in random_fold_in
#      return random_fold_in_p.bind(keys, msgs)
#             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#   NotImplementedError: Closing over inputs to shard_map where the input is sharded on `Explicit` axes is not
#   implemented. As a workaround, please pass those inputs as an argument to shard_map.
#   Got input with shape key<fry>[]({Explicit: ('batch',)})
#
# See also: https://github.com/jax-ml/jax/issues/29162
#
# An alternative to fold-in might be to split the rng key and shard it, e.g.,
#   rngs = jax.random.split(rng, device_mesh.shape[sharding_dim])
#   rngs = jax.device_put(rngs, device_mesh)
# However, the previous might produce the following error related to addressable devices and require more thinking.


def fsdp_map(local_fn, mesh: jax.sharding.Mesh, batch_dim_name: str = "batch"):
  def _local_grad_fn(params, rng_key, data, static_data):
    local_rng_key = jax.random.fold_in(rng_key, jax.lax.axis_index(batch_dim_name))
    return local_fn(params, local_rng_key, data, static_data)

  def _pmean_grad_fn(params, rng_key, data, static_data):
    return jax.lax.pmean(
      _local_grad_fn(params, rng_key, data, static_data), axis_name=batch_dim_name
    )

  def global_fn(params, rng_key, data, static_data):
    _global_grad_fn = shard_map(
      _pmean_grad_fn,
      mesh=mesh,
      in_specs=(P(), P(), P(batch_dim_name), P()),
      out_specs=P(),
      check_rep=False,
    )
    return _global_grad_fn(params, rng_key, data, static_data)

  return global_fn
