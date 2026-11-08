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
"""Validated training settings, independent of the training runtime."""

import warnings
from typing import Annotated, Any, Literal

import pandas as pd
from pydantic import (
  AfterValidator,
  BaseModel,
  ConfigDict,
  Field,
  NonNegativeInt,
  PositiveInt,
  ValidationInfo,
  field_validator,
  model_validator,
)

from mobydick.leviathan.config import LeviathanConfig


class ApplicationConfig(BaseModel):
  model_config = ConfigDict(extra="forbid")


def _check_duration(value: str) -> str:
  # Deprecated units (e.g. "d" for days, an error in pandas 4) are rejected now rather than when the first sample is read.
  with warnings.catch_warnings():
    warnings.simplefilter("error")
    try:
      pd.Timedelta(value)
    except (ValueError, Warning) as error:
      raise ValueError(f"{value!r} is not a valid duration (e.g. '1D', '24h'): {error}") from error
  return value


# A pandas duration, e.g. "1D" or "24h".
Duration = Annotated[str, AfterValidator(_check_duration)]


class ReadConfig(BaseModel):
  """Options forwarded to xarray.open_dataset, including backend-specific keywords."""

  model_config = ConfigDict(extra="allow")
  engine: str | None = None
  chunks: int | str | dict[str, int] | None = None
  cache: bool | None = None
  decode_cf: bool | None = None
  mask_and_scale: bool | dict[str, bool] | None = None
  decode_times: bool | dict[str, bool] | None = None
  decode_timedelta: bool | dict[str, bool] | None = None
  use_cftime: bool | None = None
  concat_characters: bool | dict[str, bool] | None = None
  decode_coords: bool | str | None = None
  drop_variables: str | list[str] | None = None
  inline_array: bool = False
  chunked_array_type: str | None = None
  from_array_kwargs: dict[str, Any] = Field(default_factory=dict)
  backend_kwargs: dict[str, Any] = Field(default_factory=dict)
  create_default_indexes: bool = True

  @model_validator(mode="after")
  def reject_path(self):
    if "filename_or_obj" in (self.model_extra or {}):
      raise ValueError("filename_or_obj must be specified using the source's path")
    return self


def source_read() -> ReadConfig:
  return ReadConfig(engine="zarr", chunks=None)


class LevelSliceConfig(ApplicationConfig):
  start: Annotated[int, Field(strict=True, ge=0)] | None = None
  stop: Annotated[int, Field(strict=True, ge=0)] | None = None
  step: Annotated[int, Field(strict=True, gt=0)] = 1

  @model_validator(mode="after")
  def check_bounds(self):
    if self.start is not None and self.stop is not None and self.start > self.stop:
      raise ValueError("level slice start must not exceed stop")
    return self


Levels = tuple[NonNegativeInt, ...] | Literal["all"] | LevelSliceConfig


class TaskConfig(ApplicationConfig):
  input_variables: tuple[str, ...]
  target_variables: tuple[str, ...]
  forcing_variables: tuple[str, ...]
  levels: Levels
  input_duration: Duration


class MeshConfig(ApplicationConfig):
  path: str
  mesh_size_tag_name: str = "MeshSize"
  step: NonNegativeInt = Field(default=0, description="Mesh-size tag timestep")
  log_statistics: bool = Field(
    default=False,
    description="Log statistics of the encoder, processor and decoder graphs at startup (process 0; without the "
    "networkx metrics, see moby-dick graph-statistics)",
  )


class SourceConfig(ApplicationConfig):
  path: str
  read: ReadConfig = Field(default_factory=source_read)

  @model_validator(mode="before")
  @classmethod
  def read_defaults(cls, values):
    if isinstance(values, dict) and isinstance(values.get("read"), dict):
      defaults = source_read().model_dump(exclude_unset=True)
      return {**values, "read": {**defaults, **values["read"]}}
    return values


class StoreConfig(ApplicationConfig):
  """A standardized anomaly store written by moby-dick download, read with TensorStore."""

  path: str
  io_threads: PositiveInt = Field(
    default=16, description="TensorStore concurrency of file I/O and decoding, per process (Grain worker)"
  )


class MaskConfig(ApplicationConfig):
  """A boolean (lat, lon) mask written by moby-dick download-mask."""

  path: str
  fill_value: float


class PerformanceConfig(ApplicationConfig):
  ram_budget_mb: PositiveInt | None = None
  max_workers: NonNegativeInt | None = None
  max_buffer_size: PositiveInt | None = None
  samples_to_check: PositiveInt | None = None


class GrainDatasetConfig(ApplicationConfig):
  target_lead_times: Duration
  split_date: str
  local_batch_size: PositiveInt = Field(default=1, description="Samples per local device")
  pick_performance_config: PerformanceConfig | None = None


class PolicyConfig(ApplicationConfig):
  names_which_can_be_saved: list[str] = Field(default_factory=list)
  names_which_can_be_offloaded: list[str] = Field(default_factory=list)
  offload_src: str = "device"
  offload_dst: str = "pinned_host"
  remat_loss: bool = Field(
    default=True,
    description="Also apply the policy to a checkpoint around the whole loss. The names mark the outputs of "
    "rematerialized blocks, which only an enclosing checkpoint can save or offload: without it they stay on the "
    "device.",
  )
  prevent_cse: bool = Field(
    default=False,
    description="Prevent XLA from merging the recomputation of rematerialized blocks with their forward pass. JAX "
    "recommends it outside of scan: without it, XLA may keep the activations that remat should discard.",
  )
  scan: bool = Field(
    default=False,
    description="Run the message passing steps of the processor in a scan instead of unrolling them.",
  )
  edge_chunk_size: NonNegativeInt = Field(
    default=0,
    description="If positive, update edge sets with more edges in chunks of this many edges, one after the other, "
    "so that only one chunk of gathered features and MLP intermediates is alive at a time. 0: off.",
  )


class NormalizationConfig(ApplicationConfig):
  skip_names: list[str] = Field(default_factory=list)


class NamedOptaxConfig(BaseModel):
  model_config = ConfigDict(extra="allow")
  name: Annotated[str, Field(min_length=1)]


class OptimizerConfig(ApplicationConfig):
  schedule: NamedOptaxConfig
  gradient_transformation: Annotated[list[NamedOptaxConfig], Field(min_length=1)]


class CheckpointConfig(ApplicationConfig):
  save_interval_steps: PositiveInt = Field(
    default=500,
    description="Save after every this many steps, after the last one, and at a preemption (SIGTERM). With "
    "[validation], a multiple of validation.interval_steps, so that every regular checkpoint has a validation loss.",
  )
  keep_latest: PositiveInt = Field(
    default=1,
    description="Keep the latest checkpoints, for resuming (Orbax deletes older ones only after a save is committed).",
  )
  keep_best: NonNegativeInt = Field(
    default=1,
    description="Also keep the checkpoints with the lowest validation loss (with [validation]); the best one becomes "
    "the final model. Without validation, the final model is the latest checkpoint.",
  )
  enable_async_checkpointing: bool = True


class ValidationConfig(ApplicationConfig):
  interval_steps: PositiveInt = Field(
    default=250, description="Compute the validation loss every this many steps, and after the last one."
  )
  num_samples: PositiveInt = Field(
    default=128,
    description="Held-out windows (after grain_dataset.split_date), evenly spaced over the held-out period; rounded "
    "up to a multiple of the global batch size. The same windows at every validation and in every run.",
  )
  read_threads: PositiveInt = Field(
    default=1,
    description="Windows read ahead in each process, by as many threads; each holds one window in host memory.",
  )


class TrainingConfig(ApplicationConfig):
  description: str = ""
  license: str = ""
  training_steps: PositiveInt
  seed: NonNegativeInt
  mesh: MeshConfig
  mask: MaskConfig
  # One model today; with more, a union discriminated by `architecture` (pydantic rejects a one-member one).
  model: LeviathanConfig
  task: TaskConfig
  normalization: NormalizationConfig
  policy: PolicyConfig = Field(default_factory=PolicyConfig)
  grain_dataset: GrainDatasetConfig
  dataset: StoreConfig
  tendency_std: SourceConfig
  optimizer: OptimizerConfig
  checkpoints: CheckpointConfig = Field(default_factory=CheckpointConfig)
  validation: ValidationConfig | None = None

  @model_validator(mode="after")
  def check_save_interval(self) -> "TrainingConfig":
    if self.validation is not None and self.checkpoints.save_interval_steps % self.validation.interval_steps != 0:
      raise ValueError(
        f"checkpoints.save_interval_steps ({self.checkpoints.save_interval_steps}) must be a multiple of "
        f"validation.interval_steps ({self.validation.interval_steps}): checkpoints are ranked by validation loss."
      )
    return self

  @field_validator("tendency_std")
  @classmethod
  def warn_about_chunks(cls, source: SourceConfig, info: ValidationInfo):
    if source.read.chunks is not None:
      warnings.warn(
        f"{info.field_name}.read.chunks={source.read.chunks!r}: training currently requires "
        "chunks=None. The configured value is forwarded unchanged; omit chunks "
        "to use the required default.",
        UserWarning,
        stacklevel=2,
      )
    return source
