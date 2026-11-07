# SPDX-FileCopyrightText: 2026 Stefano Campanella
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
"""End-to-end training on CPU: graphcast train on the synthetic standardized store, a tiny model on the icosahedral
multi-mesh, 4 devices in one process, then a restart with --continue.

Training initializes jax.distributed, which must happen before any other JAX call: each run is a subprocess with a
one-process "cluster" on localhost.
"""

import os
import pathlib
import re
import subprocess
import sys
from typing import Any

import numpy as np
import pytest
from click.testing import CliRunner

from graphcast import checkpoint, icosahedral_mesh
from graphcast.downloader import compute_tendency_std
from graphcast.geospatial_mesh_utils import write_multi_mesh
from graphcast.model import CheckPoint

REPOSITORY = pathlib.Path(__file__).parents[1]
PRODUCTION_CONFIG = REPOSITORY / "configs" / "training" / "production_run.toml"
TRAINING_STEPS = 4
INPUT_VARIABLES = '["thetao", "zos", "deptho", "glorys_mask", "lsm", "year_progress_sin", "year_progress_cos"]'

# Runs graphcast train in a fresh process: jax.distributed with one process on localhost, then the CLI (its own call
# to jax.distributed.initialize, which needs a cluster to detect, is skipped).
RUNNER = """
import socket
import sys

import jax

with socket.socket() as s:
  s.bind(("127.0.0.1", 0))
  port = s.getsockname()[1]
jax.distributed.initialize(f"127.0.0.1:{port}", num_processes=1, process_id=0)
jax.distributed.initialize = lambda *args, **kwargs: None

from graphcast.training import train

train.main(sys.argv[1:], prog_name="graphcast train")
"""


def _set(text: str, pattern: str, replacement: str) -> str:
  text, count = re.subn(pattern, replacement, text, flags=re.MULTILINE | re.DOTALL)
  assert count == 1, pattern
  return text


@pytest.fixture(scope="module")
def setup(standardized_path, mask_path, tmp_path_factory) -> dict[str, Any]:
  """The tiny training config (synthetic store, mask, tendency_std, icosahedral mesh) and the output paths."""
  root = tmp_path_factory.mktemp("training")
  tendency_std = root / "tendency_std"
  result = CliRunner().invoke(compute_tendency_std, [standardized_path, str(tendency_std), "--log-level", "warning"])
  assert result.exit_code == 0, result.output
  mesh = root / "mesh.msh"
  write_multi_mesh(icosahedral_mesh.get_multi_mesh_data(1), mesh)

  text = PRODUCTION_CONFIG.read_text()
  for pattern, replacement in (
    (r"^training_steps = [\d_]+", "training_steps = STEPS"),
    (r"^input_variables = \[.*?\]", f"input_variables = {INPUT_VARIABLES}"),
    (r"^target_variables = \[.*?\]", 'target_variables = ["thetao", "zos"]'),
    (r"^forcing_variables = \[.*?\]", 'forcing_variables = ["year_progress_sin", "year_progress_cos"]'),
    (r"^latent_size = \d+", "latent_size = 8"),
    (r"^gnn_msg_steps = \d+", "gnn_msg_steps = 2"),
    (r"^\[model.per_variable_weights\].*?(?=^\[mesh\])", "[model.per_variable_weights]\nzos = 0.5\nthetao = 1.0\n\n"),
    (r'^path = "/data/geometry/[^"]*"', f'path = "{mesh}"'),
    (r'^path = "/data/mask"', f'path = "{mask_path}"'),
    (r'^path = "/data/dataset"', f'path = "{standardized_path}"'),
    (r"^io_threads = \d+", "io_threads = 2"),
    (r"^split_date = .*?$", 'split_date = "2020-02-28"'),
    (r"^local_batch_size = \d+", "local_batch_size = 4"),
    (r"^path = '/data/tendency_std'", f"path = '{tendency_std}'"),
    (r"^warmup_steps = [\d_]+", "warmup_steps = 1"),
    (r"^decay_steps = [\d_]+", "decay_steps = 10"),
    (r"^save_interval_steps = [\d_]+", "save_interval_steps = 2"),
    (r"^interval_steps = [\d_]+", "interval_steps = 2"),
    (r"^num_samples = [\d_]+", "num_samples = 4"),
  ):
    text = _set(text, pattern, replacement)
  configs = {}
  for steps in (TRAINING_STEPS // 2, TRAINING_STEPS):
    configs[steps] = root / f"tiny_{steps}.toml"
    configs[steps].write_text(text.replace("training_steps = STEPS", f"training_steps = {steps}"))
  return {"configs": configs, "model": root / "model.ckpt", "train": root / "train", "tensorboard": root / "tb"}


def _train(config: pathlib.Path, setup: dict, *options: str) -> subprocess.CompletedProcess:
  env = {
    **os.environ,
    "PYTHONPATH": str(REPOSITORY / "src"),
    "JAX_PLATFORMS": "cpu",
    # No Grain worker processes: windows are read in the training process.
    "GRAPHCAST_GRAIN_NUM_WORKERS": "0",
  }
  arguments = [
    str(config),
    str(setup["model"]),
    str(setup["train"]),
    "--tensorboard-logdir",
    str(setup["tensorboard"]),
    "--log-level",
    "info",
    *options,
  ]
  return subprocess.run(
    [sys.executable, "-c", RUNNER, *arguments], env=env, capture_output=True, text=True, check=False, timeout=900
  )


def test_train_and_continue(setup):
  """Training runs its steps (batch split over 4 devices, validation, checkpoints, TensorBoard) and saves the best
  model; a second run with --continue and more steps resumes after the last checkpoint and finishes the steps."""
  first = _train(setup["configs"][TRAINING_STEPS // 2], setup)
  assert first.returncode == 0, first.stderr[-5000:]
  log = first.stderr
  assert "Using a JAX mesh with 4 devices" in log
  assert len(re.findall(r"Step \d+ timings", log)) == TRAINING_STEPS // 2
  losses = [float(loss) for loss in re.findall(r"Step \d+ validation: loss ([\d.]+)", log)]
  assert len(losses) == 1 and np.isfinite(losses).all()
  assert setup["model"].exists()
  assert any(setup["tensorboard"].rglob("events.out.tfevents.*"))

  second = _train(setup["configs"][TRAINING_STEPS], setup, "--continue", "--overwrite")
  assert second.returncode == 0, second.stderr[-5000:]
  log = second.stderr
  assert f"starting at step {TRAINING_STEPS // 2}" in log
  steps = [int(step) for step in re.findall(r"Step (\d+) timings", log)]
  assert steps == list(range(TRAINING_STEPS // 2, TRAINING_STEPS))

  with setup["model"].open("rb") as file:
    model = checkpoint.load(file, CheckPoint)
  assert model.task_config.target_variables == ("thetao", "zos")
  assert model.model_config.latent_size == 8
  assert len(model.mesh_data.mesh_graph.vertices) == 42
  assert model.params
