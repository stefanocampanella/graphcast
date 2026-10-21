import logging
import pathlib
import typing
from collections.abc import Mapping
from typing import Any

import jax
import numpy as np
import orbax.checkpoint as ocp
from grain.checkpoint import CheckpointRestore as IterDatasetRestore
from grain.checkpoint import CheckpointSave as IterDatasetSave
from jax.experimental import multihost_utils
from orbax.checkpoint.args import PyTreeRestore, PyTreeSave

from graphcast import grain_utils
from graphcast.cli_utils import OrbaxLogger
from graphcast.dataloader_utils import (
  DatasetStream,
  InputsTargetsForcingsIterator,
)
from graphcast.training_config import TrainingConfig
from graphcast.training_utils import (
  JAXLossAndDiagnostics,
  Params,
  make_process_mesh,
)

logger = logging.getLogger(__name__)


def get_checkpoint_manager(
    ckpt_path: pathlib.Path,
    configs: TrainingConfig,
) -> ocp.CheckpointManager:
  logger.info(f"Reading from and saving checkpoints to {ckpt_path}")
  ckpt_mngr_options = ocp.CheckpointManagerOptions(
    best_fn=lambda metrics: metrics[0],
    best_mode="min",
    **configs.checkpoints.model_dump(exclude_unset=True),
  )
  # `jax.multihost_utils.sync_global_devices` implements the barrier by calling
  # `jax.multihost_utils.broadcast_on_to_all`, which inside uses jax.sharding.Mesh declared for the purpose and
  # generally different from the context mesh, causing an error.
  # For this reason, orbax CheckpointManager (which uses such a barrier) needs to be called using `null_mesh`, or before
  # `jax.sharding.set_mesh()` is called. The same goes for save and restore operations.
  #  See: https://github.com/google/orbax/issues/2545
  mesh = make_process_mesh()
  with jax.sharding.set_mesh(mesh):
    ckpt_mngr = ocp.CheckpointManager(
      ckpt_path, options=ckpt_mngr_options, logger=OrbaxLogger()
    )
  return ckpt_mngr


GRAIN_NUM_WORKERS_ITEM = "grain_num_workers"
# Grain num_workers of each iterator item, one value per JAX process.
GrainNumWorkers = Mapping[str, list[int]]


def gather_grain_num_workers(
    train_stream: DatasetStream, test_stream: DatasetStream
) -> GrainNumWorkers:
  """Gather the Grain num_workers of both streams from all processes, which may autotune them independently."""
  local = np.array([train_stream.options.num_workers, test_stream.options.num_workers])
  gathered = (
    multihost_utils.process_allgather(local) if jax.process_count() > 1 else local[None]
  )
  return {
    "train_iterator": gathered[:, 0].tolist(),
    "test_iterator": gathered[:, 1].tolist(),
  }


def check_grain_num_workers(
    ckpt_mngr: ocp.CheckpointManager,
    step: int,
    grain_num_workers: GrainNumWorkers,
    items: typing.Iterable[str],
) -> None:
  """Raise if the given iterator items of a checkpoint were saved with a different Grain num_workers."""
  if GRAIN_NUM_WORKERS_ITEM not in ckpt_mngr.item_metadata(step):
    logger.warning(
      f"Checkpoint {step=} does not record Grain num_workers; cannot check it matches the current run."
    )
    return
  saved = ckpt_mngr.restore(
    step, args=ocp.args.Composite(**{GRAIN_NUM_WORKERS_ITEM: ocp.args.JsonRestore()})
  )
  saved = saved[GRAIN_NUM_WORKERS_ITEM]
  for item in items:
    if saved.get(item) != grain_num_workers[item]:
      raise ValueError(
        f"Checkpoint {step=} was saved with Grain num_workers={saved.get(item)} for {item} (one value per process), "
        f"but the current run uses {grain_num_workers[item]}. Grain's iterator state depends on num_workers: set "
        f"{grain_utils.GRAIN_ENV_PREFIX}NUM_WORKERS to the saved value (and disable "
        "grain_dataset.pick_performance_config) to restore this checkpoint."
      )


def push_checkpoint(
    ckpt_mngr: ocp.CheckpointManager,
    step: int,
    metrics: JAXLossAndDiagnostics,
    train_iterator: InputsTargetsForcingsIterator,
    test_iterator: InputsTargetsForcingsIterator,
    params: Params,
    opt_state: Params,
    grain_num_workers: GrainNumWorkers,
) -> None:
  metrics = jax.tree_util.tree_map(lambda x: x.item(), metrics)
  # FIXME: Orbax messes up with global mesh, see comment in get_checkpoint_manager. Check if new versions of Orbax fix the issue.
  ckpt_mngr.save(
    step,
    args=ocp.args.Composite(
      train_iterator=IterDatasetSave(train_iterator),
      test_iterator=IterDatasetSave(test_iterator),
      params=PyTreeSave(params),
      opt_state=PyTreeSave(opt_state),
      **{GRAIN_NUM_WORKERS_ITEM: ocp.args.JsonSave(dict(grain_num_workers))},
    ),
    metrics=metrics,
  )


def pull_checkpoint(
    ckpt_mngr: ocp.CheckpointManager,
    step: int,
    train_iterator: InputsTargetsForcingsIterator | None = None,
    test_iterator: InputsTargetsForcingsIterator | None = None,
    params: Params | None = None,
    opt_state: Params | None = None,
    grain_num_workers: GrainNumWorkers | None = None,
) -> Mapping[str, Any]:
  """Restore the given items; iterators require grain_num_workers, which must match the saved values."""
  logger.info(f"Restoring {step=}")
  iterator_items = [
    name
    for name, iterator in (
      ("train_iterator", train_iterator),
      ("test_iterator", test_iterator),
    )
    if iterator is not None
  ]
  if iterator_items:
    if grain_num_workers is None:
      raise ValueError("grain_num_workers is required to restore dataset iterators.")
    check_grain_num_workers(ckpt_mngr, step, grain_num_workers, iterator_items)
  # TODO: check restore and save args options related to sharding and layout
  # FIXME: Orbax messes up with global mesh, see comment in get_checkpoint_manager. Check if new versions of Orbax fix the issue.
  composite_args = {}
  if train_iterator is not None:
    composite_args["train_iterator"] = IterDatasetRestore(train_iterator)
  if test_iterator is not None:
    composite_args["test_iterator"] = IterDatasetRestore(test_iterator)
  if params is not None:
    composite_args["params"] = PyTreeRestore(params)
  if opt_state is not None:
    composite_args["opt_state"] = PyTreeRestore(opt_state)
  if not composite_args:
    raise ValueError("No restore arguments specified.")
  return ckpt_mngr.restore(step=step, args=ocp.args.Composite(**composite_args))
