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
"""Tests of the CLI helpers (cli_utils.py): configs, logging and the compile-time report."""

import logging
from types import SimpleNamespace

import click
import jax
import jax.numpy as jnp
import pytest
from pydantic import BaseModel, ConfigDict

from graphcast import cli_utils


class _Schema(BaseModel):
  model_config = ConfigDict(extra="forbid")
  name: str
  steps: int = 1


def test_read_configs(tmp_path):
  """A TOML file is read as a dict, updated with injected keys, and validated against a schema if given."""
  path = tmp_path / "config.toml"
  path.write_text('name = "a"\nsteps = 2\n')
  assert cli_utils.read_configs(path) == {"name": "a", "steps": 2}
  assert cli_utils.read_configs(path, schema=_Schema, inject={"steps": 3}) == _Schema(name="a", steps=3)


@pytest.mark.parametrize("content", ['name = "a"\nunknown = 1\n', "name = \n", None], ids=["schema", "toml", "missing"])
def test_invalid_configs_are_click_errors(tmp_path, content):
  """Validation errors, invalid TOML and missing files are reported as CLI errors naming the file."""
  path = tmp_path / "config.toml"
  if content is not None:
    path.write_text(content)
  with pytest.raises(click.ClickException, match="Invalid configuration in"):
    cli_utils.read_configs(path, schema=_Schema)


def test_configs_dot_keys(tmp_path):
  """Configs reads nested tables, with dotted keys in get, defaults, required keys and copies."""
  path = tmp_path / "config.toml"
  path.write_text("[a.b]\nc = [1, 2]\n")
  configs = cli_utils.Configs.read(path)
  assert configs.get("a.b.c") == [1, 2]
  assert isinstance(configs.get("a.b"), cli_utils.Configs)
  assert configs.get("a.x", default=5) == 5
  with pytest.raises(KeyError, match="a.x"):
    configs.get("a.x", required=True)
  configs.get("a.b.c").append(3)
  assert configs.get("a.b.c") == [1, 2]
  configs.get("a.b.c", copy=False).append(3)
  assert configs["a"]["b"]["c"] == [1, 2, 3]


def test_configure_logging():
  """The root logger gets the level given by name (the --log-level option)."""
  root = logging.getLogger()
  level, handlers = root.level, root.handlers[:]
  try:
    cli_utils.configure_logging("warning")
    assert root.level == logging.WARNING
    cli_utils.configure_logging("debug")
    assert root.level == logging.DEBUG
  finally:
    root.handlers[:] = handlers
    root.setLevel(level)


def test_memory_usage_summary():
  """The total is temporaries plus arguments plus outputs, minus what outputs share with (donated) arguments."""
  stats = SimpleNamespace(
    argument_size_in_bytes=100,
    output_size_in_bytes=20,
    temp_size_in_bytes=1000,
    alias_size_in_bytes=20,
    host_temp_size_in_bytes=7,
  )
  assert cli_utils.memory_usage_summary(stats) == {
    "argument_size": 100,
    "output_size": 20,
    "temp_size": 1000,
    "host_temp_size": 7,
    "total_size": 1100,
  }


def test_run_analysis_and_report(caplog):
  """The memory XLA plans for a compiled function and its cost are logged."""
  compiled = jax.jit(lambda x: (x @ x).sum()).lower(jnp.ones((64, 64))).compile()
  with caplog.at_level(logging.INFO, logger=cli_utils.__name__):
    cli_utils.run_analysis_and_report(compiled)
  messages = " ".join(record.getMessage() for record in caplog.records)
  assert "Memory usage:" in messages and "Cost:" in messages


def test_orbax_logger(caplog):
  """Orbax messages go to Python logging."""
  with caplog.at_level(logging.INFO):
    cli_utils.OrbaxLogger().log_entry({"event": "save"})
  assert "save" in caplog.text
