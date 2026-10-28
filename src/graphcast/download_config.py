"""Validated settings of the standardized anomaly downloader."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, NonNegativeInt, PositiveInt, model_validator


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

  @model_validator(mode="after")
  def check_v3_settings(self):
    if self.zarr_format == 2 and self.subchunk_order != "lexicographic":
      raise ValueError("subchunk_order requires zarr_format = 3")
    return self


class TendencyStdConfig(ApplicationConfig):
  """Settings of the standard deviation of the one-step differences of the standardized anomalies."""

  variables: list[str] | None = Field(default=None, description="Time-dependent variables; all of them if omitted")
  zero_as_missing: bool = Field(
    default=True,
    description="Write NaN where the standard deviation is 0: missing values of the standardized store are filled "
    "with 0, so a series missing at all times is constant (training fills NaN with 1)",
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
