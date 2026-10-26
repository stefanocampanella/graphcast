"""Validated settings of the standardized anomaly downloader."""

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, PositiveInt, model_validator


class ApplicationConfig(BaseModel):
  model_config = ConfigDict(extra="forbid")


class CompressorConfig(ApplicationConfig):
  """Blosc settings of data variables (coordinates use clevel 5, as in the original datasets)."""

  cname: str = "lz4"
  clevel: int = Field(default=1, ge=0, le=9)
  shuffle: Literal["noshuffle", "shuffle", "bitshuffle"] = "shuffle"


class DownloadConfig(ApplicationConfig):
  storage_options: dict[str, Any] = Field(
    default_factory=dict,
    description="fsspec options for all inputs, e.g. {anon = true} for anonymous S3 access.",
  )
  zarr_format: Literal[2, 3] = 2
  time_shard: PositiveInt = Field(default=256, description="Zarr v3: timesteps per shard of variables without depth")
  depth_time_shard: PositiveInt = Field(default=8, description="Zarr v3: timesteps per shard of variables with depth")
  subchunk_write_order: Literal["morton", "lexicographic", "colexicographic", "unordered"] = Field(
    default="morton",
    description="Zarr v3: order of the chunks within a shard. It is used when writing and not stored in metadata. "
    "TensorStore writes lexicographic order only: other orders are written (more slowly) by zarr-python.",
  )
  compressor: CompressorConfig = Field(default_factory=CompressorConfig)
  num_workers: PositiveInt = Field(default=8, description="Dask threads, each holding a few GB (one shard per array)")
  io_threads: PositiveInt = Field(
    default=16, description="TensorStore threads for reading, decoding, encoding and writing local stores"
  )

  @model_validator(mode="after")
  def check_v3_settings(self):
    if self.zarr_format == 2 and self.subchunk_write_order != "morton":
      raise ValueError("subchunk_write_order requires zarr_format = 3")
    return self
