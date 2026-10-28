"""Validated settings of the standardized anomaly downloader."""

import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, NonNegativeInt, PositiveFloat, PositiveInt, model_validator


class ApplicationConfig(BaseModel):
  model_config = ConfigDict(extra="forbid")


class CompressorConfig(ApplicationConfig):
  """Blosc settings of all variables."""

  cname: Literal["lz4", "lz4hc", "blosclz", "snappy", "zlib", "zstd"] = "lz4"
  clevel: int = Field(default=1, ge=0, le=9)
  shuffle: Literal["noshuffle", "shuffle", "bitshuffle"] = "shuffle"


class DownloadConfig(ApplicationConfig):
  zarr_format: Literal[2, 3] = 2
  time_shard: PositiveInt = Field(default=256, description="Zarr v3: timesteps per shard of variables without depth")
  depth_time_shard: PositiveInt = Field(default=8, description="Zarr v3: timesteps per shard of variables with depth")
  subchunk_order: Literal["lexicographic", "colexicographic"] = Field(
    default="lexicographic",
    description="Zarr v3: order of the chunks within a shard (by time, then depth, or by depth, then time)",
  )
  compressor: CompressorConfig = Field(default_factory=CompressorConfig)
  num_workers: PositiveInt = Field(default=8, description="Dask threads, each holding a few GB (one shard per array)")
  io_threads: PositiveInt = Field(
    default=16,
    description="TensorStore concurrency of file I/O, S3 requests, decoding and encoding",
  )
  masks: dict[str, list[str]] = Field(
    default_factory=dict,
    description="Static mask variable -> variables it masks: values outside the mask are 0. A mask with depth masks "
    "variables without depth by its first level. Variables without a mask must be finite everywhere",
  )
  static_variables: list[str] = Field(
    default_factory=list,
    description="Time-dependent variables that are static in fact: their first timestep is min-max normalized, "
    "without climatology or anomaly_std",
  )
  equal_tolerance: PositiveFloat = Field(
    default=4.0,
    description="Values within this many float32 epsilons (relative to the larger magnitude) of the climatology "
    "have a z-score of 0, whatever the anomaly_std",
  )
  reference_period: tuple[datetime.date, datetime.date] | None = Field(
    default=None,
    description="Period of the climatology and anomaly_std (inclusive): outside it, z-scores are capped at the largest "
    "absolute value inside it (per variable and level). None: all timesteps are inside",
  )

  @model_validator(mode="after")
  def check_v3_settings(self):
    if self.zarr_format == 2 and self.subchunk_order != "lexicographic":
      raise ValueError("subchunk_order requires zarr_format = 3")
    return self

  @model_validator(mode="after")
  def check_masks(self):
    masked = [name for names in self.masks.values() for name in names]
    if len(masked) != len(set(masked)):
      raise ValueError("a variable is listed under more than one mask")
    if overlap := set(masked) & set(self.masks):
      raise ValueError(f"masks cannot mask each other: {sorted(overlap)}")
    if self.reference_period is not None and self.reference_period[0] > self.reference_period[1]:
      raise ValueError("reference_period must start before it ends")
    return self


class TendencyStdConfig(ApplicationConfig):
  """Settings of the standard deviation of the one-step differences of the standardized anomalies."""

  variables: list[str] | None = Field(default=None, description="Time-dependent variables; all of them if omitted")
  zero_inflated: list[str] = Field(
    default_factory=list,
    description="Variables whose differences are mostly zero (e.g. sea ice): the standard deviation of the non-zero "
    "differences only (0 if there are none)",
  )
  time_chunk: PositiveInt = Field(default=256, description="Timesteps per Dask chunk (of one depth level)")
  compressor: CompressorConfig = Field(default_factory=CompressorConfig)
  zarr_format: Literal[2, 3] = 2
  num_workers: PositiveInt = Field(default=8, description="Dask threads")
  io_threads: PositiveInt = Field(default=16, description="TensorStore concurrency of file I/O, S3 requests, decoding")
  attrs: dict[str, str] = Field(default_factory=dict, description="Global attributes of the output")


class MaskDownloadConfig(ApplicationConfig):
  name: str = Field(default="glorys_mask", description="Mask variable of the dataset")
  level: NonNegativeInt = Field(default=0, description="Zero-based position along depth, as in task.levels")
  zarr_format: Literal[2, 3] = 2
