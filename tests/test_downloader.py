import json

import botocore.credentials
import dask
import numpy as np
import pandas as pd
import pytest
import xarray as xr
import xarray_tensorstore
from click.testing import CliRunner

from graphcast import dataloader_utils, downloader
from graphcast.dataloader import get_tensorstore_context
from graphcast.download_config import DownloadConfig, MaskDownloadConfig, TendencyStdConfig
from graphcast.training_config import MaskConfig

V2 = {"zarr_format": 2}
V3_LEXI = {"zarr_format": 3, "time_shard": 4, "depth_time_shard": 2}
V3_COLEXI = {**V3_LEXI, "subchunk_order": "colexicographic"}
FORMATS = pytest.mark.parametrize("settings", [V2, V3_LEXI, V3_COLEXI], ids=["v2", "v3_lexi", "v3_colexi"])


@pytest.fixture
def opened_inputs(input_paths) -> tuple[xr.Dataset, ...]:
  context = get_tensorstore_context(DownloadConfig().io_threads)
  return tuple(downloader.open_input(path, context) for path in input_paths)


def _write(opened_inputs, output_path, configs: DownloadConfig) -> xr.Dataset:
  standardized = downloader.standardize(*opened_inputs, configs)
  with dask.config.set(scheduler="threads", num_workers=2):
    downloader.write(standardized, output_path, configs, get_tensorstore_context(configs.io_threads))
  return standardized


def _climatology_index(dates) -> np.ndarray:
  return downloader.climatology_index(xr.DataArray(pd.to_datetime(dates), dims="time"))


@pytest.mark.parametrize(
  ("date", "expected"),
  [
    ("2019-01-01", 0),
    ("2019-02-27", 57),
    ("2019-02-28", 58),
    ("2019-03-01", 59),
    ("2019-12-31", 364),
    ("2020-01-01", 0),
    ("2020-02-27", 57),
    ("2020-02-28", 58),
    ("2020-02-29", 58),
    ("2020-03-01", 59),
    ("2020-12-31", 364),
  ],
)
def test_climatology_index(date, expected):
  assert _climatology_index([date]).tolist() == [expected]


@pytest.mark.parametrize("year", [2019, 2020, 2000, 2100])
def test_climatology_index_covers_the_climatology(year):
  index = _climatology_index(pd.date_range(f"{year}-01-01", f"{year}-12-31", freq="D"))
  np.testing.assert_array_equal(np.unique(index), np.arange(365))
  assert np.all(np.diff(index) >= 0)


def _assert_equal_to_reference(written: xr.Dataset, reference: xr.Dataset):
  assert set(written.data_vars) == set(reference.data_vars)
  for name, values in reference.data_vars.items():
    assert written[name].dtype == np.float32, name
    assert written[name].dims == values.dims, name
    np.testing.assert_allclose(written[name].values, values.values, rtol=1e-6, atol=1e-6, err_msg=name)
  np.testing.assert_array_equal(written["time"].values, reference["time"].values)


class TestAwsCredentials:
  @pytest.fixture(autouse=True)
  def environment(self, monkeypatch):
    for name in ("FSSPEC_S3_ANON", "AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_SESSION_TOKEN"):
      monkeypatch.delenv(name, raising=False)

  @staticmethod
  def _botocore_returns(monkeypatch, credentials):
    class Session:
      def get_credentials(self):
        return credentials

    monkeypatch.setattr(downloader.botocore.session, "Session", Session)

  @pytest.mark.parametrize("value", ["true", "True", "1"])
  def test_anonymous_when_requested(self, monkeypatch, value):
    monkeypatch.setenv("FSSPEC_S3_ANON", value)
    self._botocore_returns(monkeypatch, botocore.credentials.Credentials("key", "secret"))
    assert downloader.get_aws_credentials() == {"type": "anonymous"}
    assert "AWS_ACCESS_KEY_ID" not in downloader.os.environ

  def test_anonymous_without_credentials(self, monkeypatch):
    self._botocore_returns(monkeypatch, None)
    assert downloader.get_aws_credentials() == {"type": "anonymous"}

  def test_credentials_exported_to_the_environment(self, monkeypatch):
    self._botocore_returns(monkeypatch, botocore.credentials.Credentials("key", "secret", "token"))
    assert downloader.get_aws_credentials() == {"type": "environment"}
    environ = downloader.os.environ
    assert (environ["AWS_ACCESS_KEY_ID"], environ["AWS_SECRET_ACCESS_KEY"], environ["AWS_SESSION_TOKEN"]) == (
      "key",
      "secret",
      "token",
    )

  def test_stale_session_token_removed(self, monkeypatch):
    monkeypatch.setenv("AWS_SESSION_TOKEN", "stale")
    self._botocore_returns(monkeypatch, botocore.credentials.Credentials("key", "secret"))
    assert downloader.get_aws_credentials() == {"type": "environment"}
    assert "AWS_SESSION_TOKEN" not in downloader.os.environ


def test_open_input_keeps_masks_undecoded(opened_inputs, inputs):
  dataset, _, _ = opened_inputs
  assert dataset["glorys_mask"].dtype == np.int8
  assert np.issubdtype(dataset["time"].dtype, np.datetime64)
  np.testing.assert_array_equal(xarray_tensorstore.read(dataset["thetao"]).values, inputs[0]["thetao"].values)


def test_standardize_matches_the_reference(opened_inputs, reference):
  standardized = downloader.standardize(*opened_inputs, DownloadConfig()).compute()
  _assert_equal_to_reference(standardized, reference)
  assert not any(np.isnan(values).any() for values in standardized.data_vars.values())
  assert "graphcast download" in standardized.attrs["history"]


def test_standardize_requires_statistics(opened_inputs):
  dataset, climatology, anomaly_std = opened_inputs
  with pytest.raises(ValueError, match="zos"):
    downloader.standardize(dataset, climatology.drop_vars("zos"), anomaly_std, DownloadConfig())


@pytest.mark.parametrize(("settings", "thetao", "zos"), [(V2, 8, 15), (V3_LEXI, 2, 4)], ids=["default", "v3_lexi"])
def test_standardize_chunks_by_shard(opened_inputs, settings, thetao, zos):
  """Dask chunks are shards (whole along dimensions other than time), so that each shard is written once."""
  standardized = downloader.standardize(*opened_inputs, DownloadConfig.model_validate(settings))
  assert standardized["thetao"].data.chunksize == (thetao, 3, 4, 5)
  assert standardized["zos"].data.chunksize == (zos, 4, 5)
  assert standardized["glorys_mask"].data.chunksize == (3, 4, 5)


@pytest.mark.parametrize(
  ("dims", "order", "expected"),
  [
    (("time", "depth", "lat", "lon"), "colexicographic", (1, 0, 2, 3)),
    (("time", "depth", "lat", "lon"), "lexicographic", None),
    (("time", "lat", "lon"), "colexicographic", None),
  ],
)
def test_subchunk_transpose(dims, order, expected):
  variable = xr.Variable(dims, np.zeros((1,) * len(dims)))
  configs = DownloadConfig(zarr_format=3, subchunk_order=order)
  assert downloader._subchunk_transpose(variable, configs) == expected


@FORMATS
def test_write(opened_inputs, reference, output_path, settings):
  _write(opened_inputs, output_path, DownloadConfig.model_validate(settings))
  _assert_equal_to_reference(xr.open_zarr(output_path, consolidated=True).load(), reference)
  written = xarray_tensorstore.open_zarr(str(output_path), consolidated=True)
  _assert_equal_to_reference(xarray_tensorstore.read(written).compute(), reference)


def test_write_v2_layout(opened_inputs, output_path):
  _write(opened_inputs, output_path, DownloadConfig.model_validate(V2))
  metadata = json.loads((output_path / ".zmetadata").read_text())["metadata"]
  thetao, zos, mask = (metadata[f"{name}/.zarray"] for name in ("thetao", "zos", "glorys_mask"))
  assert thetao["chunks"] == [1, 1, 4, 5]
  assert zos["chunks"] == [1, 4, 5]
  assert mask["chunks"] == [3, 4, 5]
  assert thetao["compressor"] == {"id": "blosc", "cname": "lz4", "clevel": 1, "shuffle": 1, "blocksize": 0}


@pytest.mark.parametrize(("settings", "filters"), [(V3_LEXI, []), (V3_COLEXI, ["transpose"])], ids=["lexi", "colexi"])
def test_write_v3_layout(opened_inputs, output_path, settings, filters):
  _write(opened_inputs, output_path, DownloadConfig.model_validate(settings))
  metadata = json.loads((output_path / "zarr.json").read_text())["consolidated_metadata"]["metadata"]
  thetao, zos = metadata["thetao"], metadata["zos"]
  assert thetao["chunk_grid"]["configuration"]["chunk_shape"] == [2, 3, 4, 5]
  assert zos["chunk_grid"]["configuration"]["chunk_shape"] == [4, 4, 5]
  assert [codec["name"] for codec in thetao["codecs"]] == [*filters, "sharding_indexed"]
  sharding = thetao["codecs"][-1]["configuration"]
  # Chunks are single timesteps and levels: the transposed chunk shape is the same.
  assert sharding["chunk_shape"] == [1, 1, 4, 5]
  assert [codec["name"] for codec in sharding["codecs"]] == ["bytes", "blosc"]
  if filters:
    assert thetao["codecs"][0]["configuration"]["order"] == [1, 0, 2, 3]


@pytest.mark.parametrize("settings", [V3_LEXI, V3_COLEXI], ids=["lexi", "colexi"])
def test_write_v3_shard_order(opened_inputs, output_path, settings):
  """Chunks are stored in the order of the shard index: by time, then depth, or (transposed) by depth, then time."""
  _write(opened_inputs, output_path, DownloadConfig.model_validate(settings))
  shard = (output_path / "thetao" / "c" / "0" / "0" / "0" / "0").read_bytes()
  # The index is at the end of the shard: (offset, nbytes) per chunk, then a crc32c checksum.
  num_chunks = 2 * 3
  index = np.frombuffer(shard[-(16 * num_chunks + 4) : -4], dtype="<u8").reshape(-1, 2)
  assert np.all(np.diff(index[:, 0].astype(np.int64)) > 0)


def test_write_refuses_existing_output(opened_inputs, output_path):
  configs = DownloadConfig()
  _write(opened_inputs, output_path, configs)
  standardized = downloader.standardize(*opened_inputs, configs)
  with pytest.raises(FileExistsError):
    downloader.write(standardized, output_path, configs, get_tensorstore_context(configs.io_threads))
  downloader.write(standardized, output_path, configs, get_tensorstore_context(configs.io_threads), overwrite=True)


class TestCli:
  @pytest.fixture(autouse=True)
  def local_inputs_only(self, monkeypatch):
    def fail():
      raise AssertionError("S3 credentials resolved for local inputs")

    monkeypatch.setattr(downloader, "get_aws_credentials", fail)

  @staticmethod
  def _run(input_paths, output_path, *options):
    return CliRunner().invoke(downloader.download, [*input_paths, str(output_path), "--log-level", "warning", *options])

  @FORMATS
  def test_download(self, input_paths, output_path, reference, tmp_path, settings):
    config_path = tmp_path / "configs.toml"
    config_path.write_text("".join(f"{key} = {json.dumps(value)}\n" for key, value in settings.items()))
    result = self._run(input_paths, output_path, "--config", str(config_path))
    assert result.exit_code == 0, result.output
    written = xarray_tensorstore.open_zarr(str(output_path), consolidated=True)
    _assert_equal_to_reference(xarray_tensorstore.read(written).compute(), reference)

  def test_download_without_config_writes_zarr_v2(self, input_paths, output_path):
    assert self._run(input_paths, output_path).exit_code == 0
    assert (output_path / ".zmetadata").exists()

  def test_existing_output_requires_overwrite(self, input_paths, output_path):
    assert self._run(input_paths, output_path).exit_code == 0
    result = self._run(input_paths, output_path)
    assert result.exit_code != 0
    assert "--overwrite" in result.output
    assert self._run(input_paths, output_path, "--overwrite").exit_code == 0

  def test_invalid_config(self, input_paths, output_path, tmp_path):
    config_path = tmp_path / "configs.toml"
    config_path.write_text('subchunk_order = "colexicographic"\n')
    result = self._run(input_paths, output_path, "--config", str(config_path))
    assert result.exit_code != 0
    assert "requires zarr_format = 3" in result.output
    assert not output_path.exists()


class TestMask:
  @pytest.mark.parametrize("level", [0, 1])
  def test_get_mask(self, opened_inputs, inputs, level):
    mask = downloader.get_mask(opened_inputs[0], MaskDownloadConfig(level=level))
    assert mask.dtype == bool
    assert mask.dims == ("lat", "lon")
    assert "depth" not in mask.coords
    expected = inputs[0]["glorys_mask"].isel(depth=level).values.astype(bool)
    np.testing.assert_array_equal(mask.values, expected)
    np.testing.assert_array_equal(mask["lat"].values, inputs[0]["lat"].values)

  def test_rejects_time_dependent_variables(self, opened_inputs):
    with pytest.raises(ValueError, match="depends on time"):
      downloader.get_mask(opened_inputs[0], MaskDownloadConfig(name="thetao"))

  def test_rejects_unknown_variables(self, opened_inputs):
    with pytest.raises(ValueError, match="not in the dataset"):
      downloader.get_mask(opened_inputs[0], MaskDownloadConfig(name="missing"))

  def test_rejects_levels_of_variables_without_depth(self, opened_inputs):
    with pytest.raises(ValueError, match="no depth dimension"):
      downloader.get_mask(opened_inputs[0], MaskDownloadConfig(name="deptho", level=1))

  def test_cli_writes_the_mask_read_by_training(self, mask_path, opened_inputs):
    """The store holds one boolean DataArray, loaded as is by dataloader_utils.get_mask."""
    mask = dataloader_utils.get_mask(MaskConfig(path=mask_path, fill_value=0.0))
    expected = downloader.get_mask(opened_inputs[0], MaskDownloadConfig())
    xr.testing.assert_identical(mask.drop_attrs(), expected.drop_attrs())
    assert "graphcast download-mask" in mask.attrs["history"]

  def test_cli_requires_overwrite(self, input_paths, tmp_path):
    output = tmp_path / "mask"
    arguments = [input_paths[0], str(output), "--log-level", "warning"]
    assert CliRunner().invoke(downloader.download_mask, arguments).exit_code == 0
    result = CliRunner().invoke(downloader.download_mask, arguments)
    assert result.exit_code != 0
    assert "--overwrite" in result.output
    assert CliRunner().invoke(downloader.download_mask, [*arguments, "--overwrite"]).exit_code == 0

  def test_cli_level_from_config(self, input_paths, inputs, tmp_path):
    config_path = tmp_path / "mask.toml"
    config_path.write_text("level = 1\nzarr_format = 3\n")
    output = tmp_path / "mask"
    result = CliRunner().invoke(downloader.download_mask, [input_paths[0], str(output), "--config", str(config_path)])
    assert result.exit_code == 0, result.output
    mask = xr.open_dataarray(output, engine="zarr").load()
    np.testing.assert_array_equal(mask.values, inputs[0]["glorys_mask"].isel(depth=1).values.astype(bool))


class TestTendencyStd:
  @staticmethod
  def _expected(values: np.ndarray, zero_as_missing: bool = True) -> np.ndarray:
    diffs = np.diff(values.astype(np.float64), axis=0)
    std = np.sqrt(np.mean(diffs * diffs, axis=0))
    return np.where(std > 0, std, np.nan if zero_as_missing else 0.0).astype(np.float32)

  @pytest.fixture
  def standardized(self, standardized_path) -> xr.Dataset:
    return downloader.open_input(standardized_path, get_tensorstore_context(2))

  @pytest.mark.parametrize("zero_as_missing", [True, False])
  def test_matches_the_reference(self, standardized, zero_as_missing):
    configs = TendencyStdConfig(time_chunk=4, zero_as_missing=zero_as_missing)
    result = downloader.tendency_std(standardized, configs).compute()
    assert set(result.data_vars) == {"thetao", "zos"}
    for name, values in result.data_vars.items():
      assert values.dtype == np.float32
      assert "time" not in values.dims
      source = xarray_tensorstore.read(standardized[name]).values
      np.testing.assert_allclose(values.values, self._expected(source, zero_as_missing), rtol=1e-6, err_msg=name)
    # Below the first level, the land point is missing at all times in the inputs: 0 in the store, NaN here.
    assert bool(np.isnan(result["thetao"].isel(depth=1, lat=0, lon=0))) == zero_as_missing

  def test_variables(self, standardized):
    result = downloader.tendency_std(standardized, TendencyStdConfig(variables=["zos"]))
    assert list(result.data_vars) == ["zos"]
    with pytest.raises(ValueError, match="deptho"):
      downloader.tendency_std(standardized, TendencyStdConfig(variables=["zos", "deptho"]))

  def test_attributes(self, standardized):
    result = downloader.tendency_std(standardized, TendencyStdConfig(attrs={"title": "Tendencies"}))
    assert result.attrs["title"] == "Tendencies"
    assert "graphcast compute-tendency-std" in result.attrs["history"]
    assert "graphcast download" in result.attrs["history"]

  def test_cli(self, standardized_path, standardized, tmp_path):
    """The store has one chunk per depth level (dataloader_utils.get_tendency_std reads it: test_dataloader.py)."""
    output = tmp_path / "tendency_std"
    arguments = [standardized_path, str(output), "--log-level", "warning"]
    result = CliRunner().invoke(downloader.compute_tendency_std, arguments)
    assert result.exit_code == 0, result.output
    metadata = json.loads((output / ".zmetadata").read_text())["metadata"]
    assert metadata["thetao/.zarray"]["chunks"] == [1, 4, 5]
    written = xr.open_zarr(output).load()
    expected = self._expected(xarray_tensorstore.read(standardized["thetao"]).values)
    np.testing.assert_allclose(written["thetao"].values, expected, rtol=1e-6)
    assert CliRunner().invoke(downloader.compute_tendency_std, arguments).exit_code != 0
