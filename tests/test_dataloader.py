"""Tests of the training data path: windows of the standardized anomaly store (StandardizedAnomalySource), the Grain
pipeline built from the training config, and the static training inputs (grid, tendency_std, loss masks).

The synthetic store is written by graphcast download from the inputs of conftest.py; `reference` holds the same
standardized anomalies computed directly.
"""

import pathlib
import pickle

import numpy as np
import pytest
import xarray as xr
from absl import flags
from click.testing import CliRunner

from graphcast import cli_utils, dataloader_utils, grain_utils, xarray_jax
from graphcast.dataloader import StandardizedAnomalySource, get_tensorstore_context, select_levels
from graphcast.downloader import compute_tendency_std
from graphcast.training_config import MaskConfig, SourceConfig, TrainingConfig

FULL_CONFIG = pathlib.Path(__file__).parents[1] / "configs" / "training" / "full.toml"
# Task levels: zero-based depth positions in a non-sorted order, to check that the order is kept.
LEVELS = [2, 0]


def _expected_window(reference: xr.Dataset, start: int, size: int) -> xr.Dataset:
  """The window of the reference as the dataloader returns it: selected levels, batch dimension, relative time."""
  window = select_levels(reference, LEVELS).isel(time=slice(start, start + size))
  window = window.expand_dims(dim="batch", axis=0)
  window = window.assign_coords({"datetime": window["time"].expand_dims(dim="batch", axis=0)})
  return window.assign_coords(time=window["time"] - window["time"][0])


@pytest.fixture
def source(standardized_path) -> StandardizedAnomalySource:
  return StandardizedAnomalySource(standardized_path, levels=LEVELS, window_size=3, io_threads=2)


def test_tensorstore_context():
  """The context limits all TensorStore concurrency resources to io_threads, with optional S3 credentials."""
  limits = {"data_copy_concurrency": {"limit": 7}, "file_io_concurrency": {"limit": 7}}
  limits["s3_request_concurrency"] = {"limit": 7}
  assert get_tensorstore_context(7).spec.to_json() == limits
  context = get_tensorstore_context(7, {"type": "anonymous"})
  assert context.spec.to_json() == {**limits, "aws_credentials": {"type": "anonymous"}}


def test_select_levels_keeps_order_and_depth(reference):
  """Levels are selected by depth position, in the given order, keeping the depth coordinate along them."""
  selected = select_levels(reference, LEVELS)
  assert selected["level"].values.tolist() == LEVELS
  np.testing.assert_array_equal(selected["depth"].values, reference["depth"].values[LEVELS])
  with pytest.raises(ValueError, match="zero levels"):
    select_levels(reference, slice(5, None))


@pytest.mark.parametrize("start", [0, 5, 12])
def test_window_matches_the_reference(source, reference, start):
  """A window holds the reference values (float32) with a batch dimension, datetime and time relative to its start."""
  window = source[start]
  expected = _expected_window(reference, start, 3)
  assert set(window.data_vars) == set(expected.data_vars)
  for name, values in expected.data_vars.items():
    assert window[name].dtype == np.float32, name
    assert window[name].dims == values.dims, name
    np.testing.assert_allclose(window[name].values, values.values, rtol=1e-6, atol=1e-6, err_msg=name)
  for coord in ("time", "datetime", "level", "depth", "lat", "lon"):
    np.testing.assert_array_equal(window[coord].values, expected[coord].values, err_msg=coord)
  assert window["time"].values[0] == np.timedelta64(0, "ns")


def test_length_and_end_date(standardized_path, reference):
  """One window per start day; end_date (the training split) is the last day of the last window."""
  num_days = reference.sizes["time"]
  assert len(StandardizedAnomalySource(standardized_path, levels=LEVELS, window_size=3)) == num_days - 2
  source = StandardizedAnomalySource(standardized_path, levels=LEVELS, window_size=3, end_date="2020-03-01")
  assert len(source) == (np.datetime64("2020-03-01") - np.datetime64("2020-02-20")).astype(int) + 1 - 2
  assert source[len(source) - 1]["datetime"].values[0, -1] == np.datetime64("2020-03-01")


def test_invalid_window_size(standardized_path, reference):
  """A window longer than the store is rejected."""
  with pytest.raises(ValueError, match="window_size"):
    StandardizedAnomalySource(standardized_path, levels=LEVELS, window_size=reference.sizes["time"] + 1)


def test_pickled_source_reopens_the_store(source):
  """Grain workers receive pickled sources: the open store is not pickled, but reopened."""
  window = source[4]
  restored = pickle.loads(pickle.dumps(source))
  assert restored._dataset is None
  xr.testing.assert_identical(restored[4], window)


@pytest.fixture
def training_configs(standardized_path) -> TrainingConfig:
  """The full training config, with the synthetic store and its variables."""
  configs = cli_utils.read_configs(FULL_CONFIG)
  configs["dataset"] = {"path": standardized_path, "io_threads": 2}
  configs["task"].update(
    # Static fields are inputs, as in the shipped configs.
    input_variables=["thetao", "zos", "deptho", "glorys_mask", "lsm", "year_progress_sin", "year_progress_cos"],
    target_variables=["thetao", "zos"],
    forcing_variables=["year_progress_sin", "year_progress_cos"],
    levels=LEVELS,
  )
  configs["grain_dataset"]["split_date"] = "2020-03-01"
  return TrainingConfig.model_validate(configs)


def test_shipped_training_configs_are_valid():
  """The training configs validate and read the store and mask bound by scripts/train.sh."""
  for path in FULL_CONFIG.parent.glob("*.toml"):
    configs = cli_utils.read_configs(path, schema=TrainingConfig)
    assert configs.dataset.path == "/data/dataset"
    assert configs.mask.path == "/data/mask"


@pytest.mark.parametrize("section", ["climatology", "anomaly_std"])
def test_statistics_of_the_old_dataloader_are_rejected(training_configs, section):
  """The raw-data statistics of the previous dataloader are no longer accepted in training configs."""
  with pytest.raises(ValueError, match=section):
    TrainingConfig.model_validate({**training_configs.model_dump(), section: {"path": "/data/" + section}})


def test_map_dataset_yields_inputs_targets_forcings(training_configs):
  """The pipeline built from the config yields batches of inputs, targets and forcings with the task's variables."""
  dataset = dataloader_utils.get_dataset(training_configs)
  assert dataset["level"].values.tolist() == LEVELS
  map_dataset = dataloader_utils.get_map_dataset(training_configs, dataset, seed=0)
  sample = map_dataset[0]
  assert sample is not None
  inputs, targets, forcings = sample
  task = training_configs.task
  assert set(inputs.data_vars) == set(task.input_variables)
  assert set(targets.data_vars) == set(task.target_variables)
  assert set(forcings.data_vars) == set(task.forcing_variables)
  # One input day and one target day, as input_duration and target_lead_times in the config are "1d".
  assert inputs.sizes["time"] == 1 and targets.sizes["time"] == 1
  # Static inputs have no time dimension.
  assert "time" not in inputs["deptho"].dims and "time" not in inputs["lsm"].dims
  assert inputs["thetao"].sizes["level"] == len(LEVELS)
  assert inputs["thetao"].sizes["batch"] == training_configs.grain_dataset.local_batch_size
  assert len(map_dataset) > 0


@pytest.mark.parametrize("num_workers", [0, 1])
def test_stream_with_grain_workers(training_configs, num_workers):
  """Grain workers get the pickled pipeline and reopen the store; batches match those read in this process."""
  # Grain reads absl flags, which the graphcast CLI marks as parsed.
  flags.FLAGS.mark_as_parsed()
  dataset = dataloader_utils.get_dataset(training_configs)
  grain_runtime = grain_utils.GrainRuntimeConfig(num_workers=num_workers, num_threads=1, prefetch_buffer_size=1)
  stream = dataloader_utils.get_dataset_stream(training_configs, dataset, seed=0, grain_runtime=grain_runtime)
  iterator = iter(stream.dataset)
  expected = dataloader_utils.get_map_dataset(training_configs, dataset, seed=0)
  for step in range(2):
    sample = expected[step]
    assert sample is not None
    for streamed, read in zip(next(iterator), sample, strict=True):
      # Batches from workers hold shared-memory arrays, which xarray operations reject: compare copies.
      assert set(streamed.data_vars) == set(read.data_vars)
      for name, values in read.data_vars.items():
        np.testing.assert_allclose(np.array(xarray_jax.unwrap(streamed[name].data)), values.values, err_msg=name)


@pytest.fixture
def tendency_std_path(standardized_path, tmp_path) -> str:
  """tendency_std of the synthetic store, written by graphcast compute-tendency-std."""
  output = tmp_path / "tendency_std"
  result = CliRunner().invoke(compute_tendency_std, [standardized_path, str(output), "--log-level", "warning"])
  assert result.exit_code == 0, result.output
  return str(output)


def test_tendency_std_of_the_store(training_configs, tendency_std_path):
  """graphcast compute-tendency-std writes what training reads: task levels selected, 0 at constant points."""
  configs = training_configs.model_copy(update={"tendency_std": SourceConfig(path=tendency_std_path)})
  tendency_std = dataloader_utils.get_tendency_std(configs)
  assert set(tendency_std.data_vars) == {"thetao", "zos"}
  assert tendency_std["level"].values.tolist() == LEVELS
  # Land (outside the mask) is 0 in the store at all times: constant, tendency_std 0.
  assert float(tendency_std["thetao"].sel(level=2).isel(lat=0, lon=0)) == 0.0
  assert float(tendency_std["zos"].isel(lat=3, lon=4)) == 0.0
  assert int((tendency_std["thetao"] == 0).sum()) == 2 + 1  # land column at both levels, land below level 0 at level 2


def test_tendency_std_with_missing_values_fails(training_configs, tendency_std_path, tmp_path):
  """Missing values in tendency_std are an error (no filling at training time)."""
  broken = xr.open_zarr(tendency_std_path).load()
  broken["zos"][0, 0] = np.nan
  path = tmp_path / "broken"
  broken.drop_encoding().to_zarr(path, zarr_format=2, consolidated=True)
  configs = training_configs.model_copy(update={"tendency_std": SourceConfig(path=str(path))})
  with pytest.raises(ValueError, match="zos has missing or negative values"):
    dataloader_utils.get_tendency_std(configs)


def test_grid(mask_path, inputs):
  """get_grid returns the latitudes, longitudes and boolean (lat, lon) mask of the mask store, as numpy arrays."""
  grid = dataloader_utils.get_grid(MaskConfig(path=mask_path, fill_value=0.0))
  np.testing.assert_array_equal(grid.lat, inputs[0]["lat"].values)
  np.testing.assert_array_equal(grid.lon, inputs[0]["lon"].values)
  np.testing.assert_array_equal(grid.mask, inputs[0]["glorys_mask"].isel(depth=0).values.astype(bool))


def test_loss_masks(training_configs, tendency_std_path, mask_path):
  """Per target variable and level: grid points with a non-constant series."""
  configs = training_configs.model_copy(update={"tendency_std": SourceConfig(path=tendency_std_path)})
  tendency_std = dataloader_utils.get_tendency_std(configs)
  grid = dataloader_utils.get_grid(MaskConfig(path=mask_path, fill_value=0.0))
  masks = dataloader_utils.get_loss_masks(tendency_std, grid, ["thetao", "zos"])
  # Volume variables get one mask per level, surface variables a (lat, lon) one.
  assert masks["thetao"].dtype == bool and masks["thetao"].dims == ("level", "lat", "lon")
  assert masks["zos"].dims == ("lat", "lon")
  # Land below the surface of an ocean column is excluded at that level only.
  assert not masks["thetao"].sel(level=2).isel(lat=0, lon=0) and masks["thetao"].sel(level=0).isel(lat=0, lon=0)
  # The land column is outside the grid mask and constant.
  assert not masks["zos"].isel(lat=3, lon=4) and not masks["thetao"].isel(lat=3, lon=4).any()
  assert int(masks["thetao"].sum()) == 2 * 20 - 3
  # A target variable without tendency_std, or a grid with other coordinates, is an error.
  with pytest.raises(ValueError, match="so"):
    dataloader_utils.get_loss_masks(tendency_std, grid, ["thetao", "so"])
  with pytest.raises(ValueError, match="lat coordinates"):
    dataloader_utils.get_loss_masks(tendency_std, grid._replace(lat=grid.lat + 1), ["thetao"])
