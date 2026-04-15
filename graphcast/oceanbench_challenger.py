import logging
from typing import cast

import chex
import click
import grain.python as grain
import haiku as hk
import jax
import numpy as np
import tqdm
import xarray as xr
from etils import epath

from graphcast import cli_utils, training_utils as trn_utils, xarray_jax, checkpoint, rollout
from graphcast.cli_utils import Configs
from graphcast.data_utils import fix_longitude
from graphcast.dataloader import ARCODataSource
from graphcast.model import CheckPoint

logger = logging.getLogger(__name__)


@click.group()
def cli():
  pass


@cli.command()
@click.argument("config_path",
                required=True,
                type=click.Path(path_type=epath.Path,
                                exists=True,
                                file_okay=True,
                                dir_okay=False,
                                readable=True,
                                resolve_path=True))
@click.argument("checkpoint_path",
                required=True,
                type=click.Path(path_type=epath.Path,
                                exists=True,
                                file_okay=True,
                                dir_okay=False,
                                readable=True,
                                resolve_path=True))
@click.argument("input_path",
                required=True,
                type=click.Path(path_type=epath.Path,
                                exists=True,
                                file_okay=True,
                                dir_okay=True,
                                readable=True,
                                resolve_path=True))
@click.argument("output_path",
                required=True,
                type=click.Path(path_type=epath.Path,
                                resolve_path=True))
@click.option("--data-path",
              help="Path to the data directory.",
              default=cli_utils.get_cwd(),
              type=click.Path(path_type=epath.Path,
                              exists=True,
                              file_okay=False,
                              dir_okay=True,
                              readable=True,
                              resolve_path=True))
@click.option("--day-of-week",
              help="Day of the week of the start of a forecast.",
              default=2,
              type=int)
@click.option("--rollout-length",
              help="Length of each forecast rollout in days.",
              default=10,
              type=int)
@click.option("--overwrite/--no-overwrite",
              help="Whether to overwrite the final checkpoint.",
              default=False,
              is_flag=True)
@click.option("--progress/--no-progress",
              help="Whether to display a progress bar.",
              default=False,
              is_flag=True)
@click.option('--log-level',
              default='info',
              type=click.Choice(['debug', 'info', 'warning', 'error', 'critical'],
                                case_sensitive=False))
def make(config_path: epath.Path,
         checkpoint_path: epath.Path,
         input_path: epath.Path,
         output_path: epath.Path,
         data_path: epath.Path,
         day_of_week: int = 0,
         rollout_length: int = 10,
         overwrite: bool = False,
         progress: bool = False,
         log_level: str = 'info'):

  logging.basicConfig(format='%(levelname)s - %(asctime)s: %(message)s',
                      datefmt='%Y-%m-%dT%H:%M:%S',
                      level=log_level.upper(),
                      force=True)

  output_path.parent.mkdir(parents=True, exist_ok=True)
  if output_path.exists() and not overwrite:
    raise FileExistsError(f"Output destination {output_path} already exists")

  configs = cli_utils.Configs.read(config_path)

  with checkpoint_path.open('rb') as file:
    ckpt = checkpoint.load(file, CheckPoint)

  mask = trn_utils.get_mask(data_path, configs)
  mean_by_level, stddev_by_level, diffs_stddev_by_level = trn_utils.get_artifacts(data_path, configs)
  static_data = (mean_by_level, stddev_by_level, diffs_stddev_by_level, mask)
  static_data = xarray_jax.wrap_data(static_data, to_jax=True, np_contiguous=False)

  predictor_fn = get_predictor_fn(configs, ckpt, static_data)

  logger.info(f"Loading data from {input_path}")
  input_dataset = xr.open_dataset(input_path, engine='zarr')
  input_duration_in_days = int(ckpt.task_config.input_duration.rstrip('d'))
  dates = input_dataset['time'].isel(time=slice(None, -(input_duration_in_days + rollout_length) + 1))
  input_day_of_week = (day_of_week - input_duration_in_days) % 7
  input_day_of_week_mask = dates.dt.dayofweek == input_day_of_week
  valid_dates = dates.isel(time=input_day_of_week_mask).to_numpy()
  preview = ", ".join(str(ts) for ts in valid_dates[:5])
  more = "" if len(valid_dates) <= 5 else f" and {len(valid_dates) - 5} more"
  logger.info(f"Valid dates: {preview}{more}")
  datasource = ARCODataSource(input_dataset,
                              task=ckpt.task_config,
                              target_lead_times=slice("1d", f"{rollout_length}d"),
                              valid_dates=valid_dates)


  dataset = (grain.MapDataset.source(datasource)
             .map_with_index(lambda i, ds: (valid_dates[i], ds))
             .to_iter_dataset(read_options=grain.ReadOptions(**configs.get('dataset.read_options'))))

  logger.info("Computing forecast rollouts")
  rng = jax.random.key(configs.get('seed', 0))
  forecasts = []
  for date, (inputs, targets_template, forcings) in tqdm.tqdm(iter(dataset),
                                                              disable=not progress,
                                                              total=len(valid_dates)):
    forecast = rollout.chunked_prediction(predictor_fn, rng, inputs, targets_template, forcings)
    forecast = set_oceanbench_defaults(forecast)
    forecast = forecast.expand_dims(dim="first_day_datetime", axis=0)
    forecast = forecast.assign_coords(first_day_datetime=[date + np.timedelta64(input_duration_in_days, 'D')])
    forecasts.append(forecast)
  challenger = xr.concat(forecasts, dim="first_day_datetime")

  logger.info(f"Saving forecast to {output_path}")
  challenger.to_zarr(output_path, consolidated=True, mode='w', compute=True)


def get_predictor_fn(
    configs: Configs,
    ckpt: CheckPoint,
    static_data: trn_utils.DatasetsOrDataArrays,
    ) -> rollout.PredictorFn:

  @hk.without_apply_rng
  @hk.transform
  def _predictor_fn(inputs: xr.Dataset, targets_template: xr.Dataset, forcings: xr.Dataset,
                    static_data: trn_utils.DatasetsOrDataArrays) -> xr.Dataset:
    mean_by_level, stddev_by_level, diffs_stddev_by_level, mask = static_data
    predictor = trn_utils.get_predictor(configs=configs,
                                        mesh_data=ckpt.mesh_data,
                                        grid_lat=ckpt.grid_lat,
                                        grid_lon=ckpt.grid_lon,
                                        grid_mask=ckpt.grid_mask,
                                        mean_by_level=mean_by_level,
                                        stddev_by_level=stddev_by_level,
                                        mask_da=mask,
                                        diffs_stddev_by_level=diffs_stddev_by_level)
    return predictor(inputs, targets_template, forcings)

  _predictor_fn_apply_jit = jax.jit(_predictor_fn.apply)

  def predictor_fn(_: chex.PRNGKey, inputs: xr.Dataset, targets_template: xr.Dataset,
                   forcings: xr.Dataset) -> xr.Dataset:
    return _predictor_fn_apply_jit(ckpt.params, inputs, targets_template, forcings, static_data)

  return cast(rollout.PredictorFn, predictor_fn)


def set_oceanbench_defaults(ds: xr.Dataset) -> xr.Dataset:
  ds = ds.isel(batch=0, drop=True)
  ds = ds.sel(lat=slice(-78.0, 89.75))
  ds = ds.map(fix_longitude)
  ds = ds.rename({
    'zos': 'sea_surface_height_above_geoid',
    'thetao': 'sea_water_potential_temperature',
    'so': 'sea_water_salinity',
    'vo': 'northward_sea_water_velocity',
    'uo': 'eastward_sea_water_velocity',
  })
  ds = ds.assign_attrs(
    Conventions='CF-1.8',
    area='Global',
    title='OceanBench Challenge'
    )
  ds = ds.rename({'lat': 'latitude', 'lon': 'longitude', 'time': 'lead_day_index'})
  depth = ds['depth']
  ds = ds.drop_vars('depth')
  ds = ds.rename({'level': 'depth'})
  ds = ds.assign_coords(depth=depth.to_numpy())
  rollout_days = ds['lead_day_index'].to_numpy() / np.timedelta64(1, 'D')
  rollout_days = np.round(rollout_days).astype(int)
  ds = ds.assign_coords(lead_day_index=rollout_days - 1)
  ds = ds.set_index({'depth': 'depth'})
  # TODO: detect step from data, make all name paramtric with sensisble default values
  ds['latitude'] = ds['latitude'].assign_attrs(
    axis='Y',
    long_name='Latitude',
    standard_name='latitude',
    step='0.025',
    units='degrees_north',
    units_long='Degrees North',
    valid_max=90,
    valid_min=-90)
  ds['longitude'] = ds['longitude'].assign_attrs(
    axis='X',
    long_name='Longitude',
    standard_name='longitude',
    step='0.025',
    units='degrees_east',
    units_long='Degrees East',
    valid_max=180,
    valid_min=-180)
  ds['depth'] = ds['depth'].assign_attrs(
    axis='Z',
    long_name='Depth',
    standard_name='depth',
    units='m',
    units_long='Meters',
    valid_max=1000,
    valid_min=0
  )
  return ds


if __name__ == "__main__":
  cli()