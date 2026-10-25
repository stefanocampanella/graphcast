"""Validated training settings, independent of the training runtime."""

import warnings
from typing import Annotated, Any, Literal

from pydantic import (
  BaseModel,
  ConfigDict,
  Field,
  NonNegativeInt,
  PositiveInt,
  ValidationInfo,
  field_validator,
  model_validator,
)


class ApplicationConfig(BaseModel):
  model_config = ConfigDict(extra="forbid")


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
    if "filename_or_obj" in self.model_extra:
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
  input_duration: str


class ModelConfig(ApplicationConfig):
  latent_size: PositiveInt
  gnn_msg_steps: PositiveInt
  hidden_layers: NonNegativeInt
  radius_query_fraction_edge_length: Annotated[float, Field(gt=0)]
  per_variable_weights: dict[str, float] = Field(default_factory=dict)
  mesh2grid_edge_normalization_factor: Annotated[float, Field(gt=0)] | None = None
  ocean_grid_nodes: bool = Field(
    default=False,
    description="Grid nodes for the unmasked (ocean) grid points only: the masked ones are connected to nothing.",
  )


class MeshConfig(ApplicationConfig):
  path: str
  mesh_size_tag_name: str = "MeshSize"
  step: NonNegativeInt = Field(default=0, description="Mesh-size tag timestep")


class SourceConfig(ApplicationConfig):
  path: str
  read: ReadConfig = Field(default_factory=source_read)

  @model_validator(mode="before")
  @classmethod
  def read_defaults(cls, values):
    if isinstance(values, dict) and isinstance(values.get("read"), dict):
      defaults = (
        cls.model_fields["read"].default_factory().model_dump(exclude_unset=True)
      )
      return {**values, "read": {**defaults, **values["read"]}}
    return values


class MaskConfig(SourceConfig):
  name: str
  level: Annotated[
    float, Field(ge=0, description="Depth selected using nearest neighbour")
  ]
  fill_value: float


class PerformanceConfig(ApplicationConfig):
  ram_budget_mb: PositiveInt | None = None
  max_workers: NonNegativeInt | None = None
  max_buffer_size: PositiveInt | None = None
  samples_to_check: PositiveInt | None = None


class GrainDatasetConfig(ApplicationConfig):
  target_lead_times: str
  split_date: str
  local_batch_size: PositiveInt = Field(
    default=1, description="Samples per local device"
  )
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
  remat_mlps: bool = Field(
    default=False,
    description="Also rematerialize each MLP of the graph networks separately: the backward pass of a block keeps "
    "only the inputs and outputs of its MLPs.",
  )


class NormalizationConfig(ApplicationConfig):
  skip_names: list[str] = Field(default_factory=list)


class NamedOptaxConfig(BaseModel):
  model_config = ConfigDict(extra="allow")
  name: Annotated[str, Field(min_length=1)]


class OptimizerConfig(ApplicationConfig):
  schedule: NamedOptaxConfig
  gradient_transformation: Annotated[list[NamedOptaxConfig], Field(min_length=1)]


class CheckpointConfig(BaseModel):
  model_config = ConfigDict(extra="allow")
  max_to_keep: PositiveInt | None = None
  save_interval_steps: PositiveInt = 1
  enable_async_checkpointing: bool = True


class TrainingConfig(ApplicationConfig):
  description: str = ""
  license: str = ""
  training_steps: PositiveInt
  seed: NonNegativeInt
  mesh: MeshConfig
  mask: MaskConfig
  model: ModelConfig
  task: TaskConfig
  normalization: NormalizationConfig
  policy: PolicyConfig = Field(default_factory=PolicyConfig)
  grain_dataset: GrainDatasetConfig
  dataset: SourceConfig
  climatology: SourceConfig
  anomaly_std: SourceConfig
  tendency_std: SourceConfig
  optimizer: OptimizerConfig
  checkpoints: CheckpointConfig = Field(default_factory=CheckpointConfig)

  @field_validator("dataset", "climatology", "anomaly_std", "tendency_std")
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
