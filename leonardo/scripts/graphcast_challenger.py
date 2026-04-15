import os
import pathlib

import xarray as xr

if (challenger_path := os.getenv("CHALLENGER_DATASET_PATH")) is None:
  raise ValueError("Please set the CHALLENGER_DATASET_PATH environment variable.")

challenger_path = pathlib.Path(challenger_path).resolve()
if not challenger_path.exists():
  raise ValueError(f"Path {challenger_path} does not exist.")

challenger_dataset: xr.Dataset = xr.open_dataset(challenger_path, engine='zarr', chunks={'lead_day_index': 1,
                                                                                         'first_day_datetime': 1})
challenger_dataset