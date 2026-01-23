import logging
import os
import pathlib
from typing import Callable

import click
import jax

logger = logging.getLogger(__name__)

class DictParamType(click.ParamType):
  """Click ParamType that parses mappings like "a:1,b:2" into dict[str, int].

  Rules:
  - Comma-separated items, each as key:value.
  - Keys are non-empty strings; surrounding whitespace is ignored.
  - Values must be integers; surrounding whitespace is ignored.
  - Empty string yields an empty dict.
  - Duplicate keys: later values overwrite earlier ones.

  Example:
    --param=a:1,b:2,c:3  -> {"a": 1, "b": 2, "c": 3}
  """

  name = "dict"

  def convert(self, value, param, ctx):  # type: ignore[override]
    if isinstance(value, dict):
      # Assume it's already a mapping of str->int; perform minimal validation
      result = {}
      for k, v in value.items():
        if not isinstance(k, str) or k.strip() == "":
          self.fail(f"Invalid key in mapping: {k!r}", param, ctx)
        try:
          result[k.strip()] = int(v)
        except Exception:
          self.fail(f"Invalid integer value for key {k!r}: {v!r}", param, ctx)
      return result

    if not isinstance(value, str):
      self.fail(f"Expected string for {self.name}, got {type(value).__name__}", param, ctx)

    text = value.strip()
    if text == "":
      return {}

    items = [p for p in (s.strip() for s in text.split(",")) if p != ""]
    result: dict[str, int] = {}
    for item in items:
      if ":" not in item:
        self.fail(f"Invalid item {item!r}. Expected 'key:value' pairs separated by commas.", param, ctx)
      key, val = item.split(":", 1)
      key = key.strip()
      val = val.strip()
      if key == "":
        self.fail("Empty key is not allowed in mapping.", param, ctx)
      try:
        result[key] = int(val)
      except Exception:
        self.fail(f"Value for key {key!r} must be an integer, got {val!r}.", param, ctx)
    return result


def get_distributed_logger(logger_name: str | None = None,
                           log_dir: pathlib.Path | None = None,
                           log_name: str | None =None,
                           log_suffix_fn: Callable | None = None,
                           fmt='%(levelname)s - %(asctime)s: %(message)s',
                           datefmt='%Y-%m-%dT%H:%M:%S') -> logging.Logger:

  if logger_name is not None:
    logger = logging.getLogger(logger_name)
  else:
    raise ValueError("logger_name must be specified")

  if log_dir is None:
    log_dir = pathlib.Path.cwd()
  else:
    log_dir = pathlib.Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

  if log_name is None:
    job_name = os.getenv("SLURM_JOB_NAME")
    job_id = os.getenv("SLURM_JOB_ID")
    if job_name is not None and job_id is not None:
      log_name = f"{job_name}-{job_id}"
    else:
      raise ValueError("log_name must be specified")

  if log_suffix_fn is not None:
    log_name = log_name + log_suffix_fn()

  logger.propagate = False
  logger.handlers.clear()
  log_path = log_dir / log_name
  filehandler = logging.FileHandler(log_path, mode="w")
  formatter = logging.Formatter(fmt=fmt, datefmt=datefmt)
  filehandler.setFormatter(formatter)
  logger.addHandler(filehandler)

  return logger


def memory_usage_summary(compiled_stats):
  summary = {}
  summary['argument_size'] = compiled_stats.argument_size_in_bytes
  summary['output_size'] = compiled_stats.output_size_in_bytes
  summary['temp_size'] = compiled_stats.temp_size_in_bytes
  summary['total_size'] = compiled_stats.temp_size_in_bytes + compiled_stats.argument_size_in_bytes \
      + compiled_stats.output_size_in_bytes - compiled_stats.alias_size_in_bytes
  return summary


def run_analysis_and_report(func, *args, **kwargs):
  func_jit = jax.jit(func)
  func_jit_compiled = func_jit.trace(*args, **kwargs).lower().compile()
  memory_analysis = func_jit_compiled.memory_analysis()
  cost_analysis = func_jit_compiled.cost_analysis()

  if memory_analysis is not None:
    summary = memory_usage_summary(memory_analysis)
    try:
      import humanize

      summary = jax.tree_util.tree_map(lambda x: humanize.naturalsize(x, binary=True),
                                       memory_usage_summary(memory_analysis))
    except ImportError:
      logger.debug("Package `humanize` not found, using bytes instead.")
    logger.info(f"Memory usage: {summary}")
  else:
     logger.info("Memory usage: unknown")

  if cost_analysis is not None:
    logger.info(f"Cost: {cost_analysis['flops'] * 1e-12} TFLOPs")
  else:
    logger.info("Cost: unknown")