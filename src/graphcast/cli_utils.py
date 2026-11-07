# SPDX-FileCopyrightText: 2025 Stefano Campanella
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
import logging
import pathlib
import tomllib
from typing import Any, TypeVar, overload

import click
import humanize
import jax
import orbax.checkpoint
import orbax.checkpoint.logging
from pydantic import BaseModel, ValidationError

logger = logging.getLogger(__name__)

LOG_LEVEL_OPTION = click.option(
  "--log-level",
  default="info",
  type=click.Choice(["debug", "info", "warning", "error", "critical"], case_sensitive=False),
  show_default=True,
  help="Logging level.",
)


def configure_logging(log_level: str) -> None:
  """Configure the root logger of a command: one format for all commands."""
  # force=True: imported packages configure the root logger too, and would otherwise keep their handlers.
  logging.basicConfig(
    format="%(levelname)s - %(asctime)s: %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
    level=log_level.upper(),
    force=True,
  )


ModelT = TypeVar("ModelT", bound=BaseModel)


@overload
def read_configs[ModelT: BaseModel](
  path: str | pathlib.Path, schema: type[ModelT], inject: dict[str, Any] | None = None
) -> ModelT: ...


@overload
def read_configs(
  path: str | pathlib.Path, schema: None = None, inject: dict[str, Any] | None = None
) -> dict[str, Any]: ...


def read_configs[ModelT: BaseModel](
  path: str | pathlib.Path, schema: type[ModelT] | None = None, inject: dict[str, Any] | None = None
) -> ModelT | dict[str, Any]:
  """Read local TOML, apply shallow overrides, and optionally validate a schema."""
  logger.info(f"Reading configs from {path}")
  try:
    with pathlib.Path(path).open("rb") as file:
      configs = tomllib.load(file)
    if inject is not None:
      configs.update(inject)
    return schema.model_validate(configs) if schema is not None else configs
  except (OSError, tomllib.TOMLDecodeError, ValidationError) as error:
    raise click.ClickException(f"Invalid configuration in {path}:\n{error}") from error


class OrbaxLogger(orbax.checkpoint.logging.AbstractLogger):
  def __init__(self):
    self._logger = logging.getLogger(orbax.checkpoint.__name__)

  def log_entry(self, msg, *args, **kwargs):
    self._logger.info(msg, *args, **kwargs)


def memory_usage_summary(compiled_stats):
  summary = {}
  summary["argument_size"] = compiled_stats.argument_size_in_bytes
  summary["output_size"] = compiled_stats.output_size_in_bytes
  summary["temp_size"] = compiled_stats.temp_size_in_bytes
  # Pinned host memory for offloaded activations.
  summary["host_temp_size"] = getattr(compiled_stats, "host_temp_size_in_bytes", 0)
  summary["total_size"] = (
    compiled_stats.temp_size_in_bytes
    + compiled_stats.argument_size_in_bytes
    + compiled_stats.output_size_in_bytes
    - compiled_stats.alias_size_in_bytes
  )
  return summary


def run_analysis_and_report(func_aot):
  memory_analysis = func_aot.memory_analysis()
  cost_analysis = func_aot.cost_analysis()

  if memory_analysis is not None:
    summary = jax.tree_util.tree_map(
      lambda x: humanize.naturalsize(x, binary=True), memory_usage_summary(memory_analysis)
    )
    logger.info(f"Memory usage: {summary}")
  else:
    logger.info("Memory usage: unknown")

  if isinstance(cost_analysis, list):
    cost_analysis = cost_analysis[0] if cost_analysis else None
  if cost_analysis is not None and "flops" in cost_analysis:
    logger.info(f"Cost: {cost_analysis['flops'] * 1e-12} TFLOPs")
  else:
    logger.info("Cost: unknown")


def get_cwd() -> pathlib.Path:
  return pathlib.Path.cwd()
