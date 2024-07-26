import argparse
import logging
import pathlib
import tomllib

import xarray as xr
import gcsfs

parser = argparse.ArgumentParser(prog="download-weatherbench2-data",
                                 description="Simple script to download ERA5 data from the WeatherBench 2 GCS")
parser.add_argument("config_file", help="Path to TOML file with source settings.", type=pathlib.Path)
parser.add_argument("--skip-training", action='store_true')
parser.add_argument("--skip-validation", action='store_true')

args = parser.parse_args()

logger = logging.getLogger(name="download-era5-data")
logging.basicConfig(level=logging.INFO)

with args.config_file.open('rb') as file:
    configs = tomllib.load(file)

if __name__ == '__main__':
    fs = gcsfs.GCSFileSystem(token='anon')
    bucket = configs['source_url']
    store = fs.get_mapper(bucket)
    download_path = pathlib.Path(configs['download_path']).absolute()

    def save_zarr(dataset_type, download_path):
        logger.info(f"Downloading {dataset_type} dataset to {download_path / dataset_type}.")
        start_time = configs[dataset_type + '_start_time']
        end_time = configs[dataset_type + '_end_time']
        time_span = slice(start_time, end_time)
        training_ds = xr.open_zarr(store=store).sel(time=time_span)
        training_ds.to_zarr(download_path / dataset_type)

    if not args.skip_training:
        save_zarr('training', download_path)
    if not args.skip_validation:
        save_zarr('validation', download_path)
