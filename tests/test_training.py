"""Tests of the training command that run without GPUs: arguments, the restart step, the checkpoint settings and which
checkpoints the manager keeps."""

import pathlib

import jax.numpy as jnp
import orbax.checkpoint as ocp
import pytest

from graphcast import cli_utils, orbax_utils
from graphcast.training import first_step, train
from graphcast.training_config import CheckpointConfig, TrainingConfig

PRODUCTION_CONFIG = pathlib.Path(__file__).parents[1] / "configs" / "training" / "production_run.toml"


def test_first_step():
  """A fresh run starts at 0; a checkpoint holds the state after its step, so a restored run continues after it."""
  assert first_step(None) == 0
  assert first_step(0) == 1
  assert first_step(499) == 500


def test_output_path_may_be_an_existing_file(tmp_path):
  """The model is a file: an existing one is accepted by the argument (--overwrite decides later), a directory not."""
  model = tmp_path / "model"
  model.write_bytes(b"")
  train_dir = tmp_path / "train"
  train_dir.mkdir()
  context = train.make_context("train", [str(PRODUCTION_CONFIG), str(model), str(train_dir)])
  assert context.params["output_path"] == model
  with pytest.raises(Exception, match="is a directory"):
    train.make_context("train", [str(PRODUCTION_CONFIG), str(train_dir), str(train_dir)])


def test_save_interval_must_be_a_multiple_of_the_validation_interval():
  configs = cli_utils.read_configs(PRODUCTION_CONFIG)
  configs["checkpoints"]["save_interval_steps"] = 300
  configs["validation"]["interval_steps"] = 250
  with pytest.raises(ValueError, match="multiple of"):
    TrainingConfig.model_validate(configs)
  del configs["validation"]
  TrainingConfig.model_validate(configs)


def test_unknown_checkpoint_settings_are_rejected():
  with pytest.raises(ValueError, match="max_to_keep"):
    CheckpointConfig.model_validate({"max_to_keep": 3})


def _save(ckpt_mngr: ocp.CheckpointManager, step: int, valid_loss: float | None) -> None:
  metrics = None if valid_loss is None else {orbax_utils.VALID_LOSS: valid_loss}
  ckpt_mngr.save(
    step, args=ocp.args.Composite(params=ocp.args.PyTreeSave({"w": jnp.full(2, step)})), metrics=metrics, force=True
  )
  ckpt_mngr.wait_until_finished()


def test_latest_and_best_checkpoints_are_kept(tmp_path):
  """The latest checkpoints survive whatever their loss, the best by validation loss too; an unscored (preemption)
  checkpoint is kept only while it is among the latest."""
  checkpoints = CheckpointConfig(keep_latest=2, keep_best=1, enable_async_checkpointing=False)
  ckpt_mngr = orbax_utils.get_checkpoint_manager(tmp_path, checkpoints, scored=True)
  for step, loss in ((9, 0.5), (19, 0.2), (29, 0.4), (34, None), (39, 0.9)):
    _save(ckpt_mngr, step, loss)
  assert ckpt_mngr.all_steps() == [19, 34, 39]
  assert ckpt_mngr.latest_step() == 39
  assert ckpt_mngr.best_step() == 19
  _save(ckpt_mngr, 49, 0.95)
  assert ckpt_mngr.all_steps() == [19, 39, 49]
  ckpt_mngr.close()


def test_without_validation_the_latest_checkpoints_are_kept(tmp_path):
  checkpoints = CheckpointConfig(keep_latest=2, enable_async_checkpointing=False)
  ckpt_mngr = orbax_utils.get_checkpoint_manager(tmp_path, checkpoints, scored=False)
  for step in (9, 19, 29):
    _save(ckpt_mngr, step, None)
  assert ckpt_mngr.all_steps() == [19, 29]
  assert ckpt_mngr.best_step() == 29
  ckpt_mngr.close()


@pytest.mark.parametrize("path", sorted(PRODUCTION_CONFIG.parent.glob("*.toml")), ids=lambda path: path.name)
def test_shipped_training_configs_are_valid(path):
  """Every shipped training config validates (including its durations)."""
  TrainingConfig.model_validate(cli_utils.read_configs(path))


@pytest.mark.parametrize(("section", "key"), [("task", "input_duration"), ("grain_dataset", "target_lead_times")])
@pytest.mark.parametrize("duration", ["1d", "one day"])
def test_durations_must_parse_without_deprecated_units(section, key, duration):
  """Durations are checked when the config is read: lowercase "d" for days is deprecated by pandas (an error in pandas
  4, so a run would fail when it reads the first sample), and the config shipped it."""
  configs = cli_utils.read_configs(PRODUCTION_CONFIG)
  configs[section][key] = duration
  with pytest.raises(ValueError, match="duration"):
    TrainingConfig.model_validate(configs)
