import pathlib

import click
import pydantic
import pytest

from graphcast import cli_utils
from graphcast.download_config import DownloadConfig, MaskDownloadConfig, TendencyStdConfig

CONFIG_DIR = pathlib.Path(__file__).parents[1] / "configs" / "download"
CONFIGS = sorted(CONFIG_DIR.glob("zarr_*.toml"))


def test_defaults():
  configs = DownloadConfig()
  assert configs.zarr_format == 2
  assert configs.subchunk_order == "lexicographic"
  assert (configs.compressor.cname, configs.compressor.clevel, configs.compressor.shuffle) == ("lz4", 1, "shuffle")


def test_colexicographic_order_requires_zarr_v3():
  assert DownloadConfig(zarr_format=3, subchunk_order="colexicographic").subchunk_order == "colexicographic"
  with pytest.raises(pydantic.ValidationError, match="requires zarr_format = 3"):
    DownloadConfig(zarr_format=2, subchunk_order="colexicographic")


@pytest.mark.parametrize(
  "settings",
  [
    {"zarr_format": 4},
    {"subchunk_order": "morton"},
    {"time_shard": 0},
    {"io_threads": -1},
    {"compressor": {"cname": "gzip"}},
    {"compressor": {"clevel": 10}},
    {"storage_options": {"anon": True}},
  ],
)
def test_invalid_settings(settings):
  with pytest.raises(pydantic.ValidationError):
    DownloadConfig.model_validate(settings)


def test_shipped_configs_exist():
  assert {path.stem for path in CONFIGS} >= {"zarr_v2", "zarr_v3_lexi", "zarr_v3_colexi"}


@pytest.mark.parametrize("path", CONFIGS, ids=lambda path: path.stem)
def test_shipped_configs_are_valid(path):
  configs = cli_utils.read_configs(path, schema=DownloadConfig)
  assert configs.zarr_format == (2 if path.stem == "zarr_v2" else 3)
  assert configs.static_variables == ["z", "lsm"]
  assert set(configs.masks) == {"glorys_mask", "waverys_mask", "glofas_mask"}
  assert {"thetao", "so", "uo", "vo", "zos", "siconc"} <= set(configs.masks["glorys_mask"])
  assert configs.masks["glofas_mask"] == ["dis24", "uparea"]
  assert configs.equal_tolerance == 4.0 and configs.reference_period is None


@pytest.mark.parametrize(
  ("settings", "message"),
  [
    ({"masks": {"a_mask": ["x"], "b_mask": ["x"]}}, "more than one mask"),
    ({"masks": {"a_mask": ["b_mask"], "b_mask": ["x"]}}, "cannot mask each other"),
    ({"reference_period": ["2023-12-31", "1993-01-01"]}, "start before it ends"),
    ({"equal_tolerance": 0}, "greater than 0"),
  ],
)
def test_invalid_mask_and_tolerance_settings(settings, message):
  with pytest.raises(pydantic.ValidationError, match=message):
    DownloadConfig.model_validate(settings)


def test_invalid_toml_is_a_click_error(tmp_path):
  path = tmp_path / "invalid.toml"
  path.write_text("zarr_format = 5\n")
  with pytest.raises(click.ClickException, match="zarr_format"):
    cli_utils.read_configs(path, schema=DownloadConfig)


def test_mask_defaults():
  configs = MaskDownloadConfig()
  assert (configs.name, configs.level, configs.zarr_format) == ("glorys_mask", 0, 2)


@pytest.mark.parametrize("settings", [{"level": -1}, {"zarr_format": 4}, {"levels": [0]}, {"compressor": {}}])
def test_invalid_mask_settings(settings):
  with pytest.raises(pydantic.ValidationError):
    MaskDownloadConfig.model_validate(settings)


def test_shipped_mask_config_is_valid():
  configs = cli_utils.read_configs(CONFIG_DIR / "mask.toml", schema=MaskDownloadConfig)
  assert (configs.name, configs.level) == ("glorys_mask", 0)


def test_shipped_tendency_std_config_is_valid():
  configs = cli_utils.read_configs(CONFIG_DIR / "tendency_std.toml", schema=TendencyStdConfig)
  assert configs.variables is None
  assert configs.zero_inflated == ["siconc", "sithick", "usi", "vsi"]
  assert "title" in configs.attrs


@pytest.mark.parametrize("settings", [{"time_chunk": 0}, {"attrs": {"title": 1}}, {"subchunk_order": "lexicographic"}])
def test_invalid_tendency_std_settings(settings):
  with pytest.raises(pydantic.ValidationError):
    TendencyStdConfig.model_validate(settings)
