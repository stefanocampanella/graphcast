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
from graphcast.training_config import CheckpointConfig
from graphcast.training_utils import Params

logger = logging.getLogger(__name__)


# Checkpoint metric used to rank checkpoints (lower is better).
VALID_LOSS = "valid_loss"


def _valid_loss(metrics: Mapping[str, float]) -> float:
  return metrics[VALID_LOSS]


def get_checkpoint_manager(
  ckpt_path: pathlib.Path,
  checkpoints: CheckpointConfig,
  scored: bool,
) -> ocp.CheckpointManager:
  """A manager that keeps the latest checkpoints, and if scored (by validation) also the best by validation loss.

  The training loop decides when to save (save_checkpoint saves unconditionally). Checkpoints without a validation
  loss (preemption saves) are kept only while they are among the latest.
  """
  logger.info(f"Reading from and saving checkpoints to {ckpt_path}")
  policies: list[ocp.checkpoint_managers.PreservationPolicy] = [
    ocp.checkpoint_managers.LatestN(n=checkpoints.keep_latest)
  ]
  if scored and checkpoints.keep_best > 0:
    # BestN keeps the last n checkpoints in sorted order: sort in descending order to keep the lowest losses.
    policies.append(
      ocp.checkpoint_managers.BestN(
        get_metric_fn=_valid_loss, reverse=True, n=checkpoints.keep_best, keep_checkpoints_without_metrics=False
      )
    )
  ckpt_mngr_options = ocp.CheckpointManagerOptions(
    preservation_policy=ocp.checkpoint_managers.AnyPreservationPolicy(policies),
    # best_fn only ranks checkpoints for best_step(); retention is the preservation policy's.
    best_fn=_valid_loss if scored else None,
    best_mode="min",
    enable_async_checkpointing=checkpoints.enable_async_checkpointing,
  )
  return ocp.CheckpointManager(ckpt_path, options=ckpt_mngr_options, logger=OrbaxLogger())


GRAIN_NUM_WORKERS_ITEM = "grain_num_workers"
# Grain num_workers of each iterator item, one value per JAX process.
GrainNumWorkers = Mapping[str, list[int]]


def gather_grain_num_workers(train_stream: DatasetStream) -> GrainNumWorkers:
  """Gather the Grain num_workers of the training stream from all processes, which may autotune it independently."""
  local = np.array([train_stream.options.num_workers])
  gathered = multihost_utils.process_allgather(local) if jax.process_count() > 1 else local[None]
  return {"train_iterator": gathered[:, 0].tolist()}


def check_grain_num_workers(
  ckpt_mngr: ocp.CheckpointManager,
  step: int,
  grain_num_workers: GrainNumWorkers,
  items: typing.Iterable[str],
) -> None:
  """Raise if the given iterator items of a checkpoint were saved with a different Grain num_workers."""
  # Checkpoints saved before Grain num_workers was recorded have no such item.
  if not (ckpt_mngr.directory / str(step) / GRAIN_NUM_WORKERS_ITEM).exists():
    logger.warning(f"Checkpoint {step=} does not record Grain num_workers; cannot check it matches the current run.")
    return
  saved = ckpt_mngr.restore(step, args=ocp.args.Composite(**{GRAIN_NUM_WORKERS_ITEM: ocp.args.JsonRestore()}))
  saved = saved[GRAIN_NUM_WORKERS_ITEM]
  for item in items:
    if saved.get(item) != grain_num_workers[item]:
      raise ValueError(
        f"Checkpoint {step=} was saved with Grain num_workers={saved.get(item)} for {item} (one value per process), "
        f"but the current run uses {grain_num_workers[item]}. Grain's iterator state depends on num_workers: set "
        f"{grain_utils.GRAIN_ENV_PREFIX}NUM_WORKERS to the saved value (and disable "
        "grain_dataset.pick_performance_config) to restore this checkpoint."
      )


def save_checkpoint(
  ckpt_mngr: ocp.CheckpointManager,
  step: int,
  train_iterator: InputsTargetsForcingsIterator,
  params: Params,
  opt_state: Params,
  grain_num_workers: GrainNumWorkers,
  metrics: Mapping[str, float] | None = None,
) -> None:
  """Save a checkpoint of the state after the given step; the caller decides when. Metrics are host values."""
  ckpt_mngr.save(
    step,
    args=ocp.args.Composite(
      train_iterator=IterDatasetSave(train_iterator),
      params=PyTreeSave(params),
      opt_state=PyTreeSave(opt_state),
      **{GRAIN_NUM_WORKERS_ITEM: ocp.args.JsonSave(dict(grain_num_workers))},
    ),
    metrics=None if metrics is None else dict(metrics),
    # Bypass Orbax's own save decision (should_save): the loop decides.
    force=True,
  )


def restore_checkpoint(
  ckpt_mngr: ocp.CheckpointManager,
  step: int,
  train_iterator: InputsTargetsForcingsIterator | None = None,
  params: Params | None = None,
  opt_state: Params | None = None,
  grain_num_workers: GrainNumWorkers | None = None,
) -> Mapping[str, Any]:
  """Restore the given items; the iterator requires grain_num_workers, which must match the saved value.

  Checkpoints saved with a test_iterator item (before the test loss was dropped) restore too: it is not read.
  """
  logger.info(f"Restoring {step=}")
  if train_iterator is not None:
    if grain_num_workers is None:
      raise ValueError("grain_num_workers is required to restore the dataset iterator.")
    check_grain_num_workers(ckpt_mngr, step, grain_num_workers, ["train_iterator"])
  composite_args = {}
  if train_iterator is not None:
    composite_args["train_iterator"] = IterDatasetRestore(train_iterator)
  if params is not None:
    composite_args["params"] = PyTreeRestore(params)
  if opt_state is not None:
    composite_args["opt_state"] = PyTreeRestore(opt_state)
  if not composite_args:
    raise ValueError("No restore arguments specified.")
  return ckpt_mngr.restore(step=step, args=ocp.args.Composite(**composite_args))
