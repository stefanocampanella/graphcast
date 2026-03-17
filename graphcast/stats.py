#  The implementation of stats computing routines in what follows is the result of a number of experiments,
#  which failed due to performance problems. In most cases, Dask created huge task graphs (22GB), and jobs died.
#  Notice, even in local tests using a partial dataset with about one week of data, the graph was still huge
#  (about 2.3GB). This was related to the Dask client lazily opening datasets or embedding the results in the graph.
#  However, the main problem was related to `groupby` (in the computation of the climatology). Early optimizations[1]
#  reusing intermediate results made things worse, making the task graph even bigger and more complex, and
#  unsurprisingly turned out to be detrimental.
#
#  I henceforth tried the following:
#    1. Compute one stat (clima, mean, std, diff) at a time.
#    2. Read using a different chunking than on disk.
#    2. Investigate the use of flox, and in general optimizations related to groupby operations. [not useful]
#
# Regarding the latter, see, for example:
#    1. https://discourse.pangeo.io/t/optimizing-climatology-calculation-with-xarray-and-dask/2453
#    2. https://flox.readthedocs.io/en/latest/user-stories/climatology.html
#    3. https://xarray.dev/blog/flox
#
#  With these and other optimizations, now we're able to get the job to the end without errors and compute the stats.
#  However, the performance is still horrible and dominated by communication, even for the mean.
#  The other optimizations mentioned above include:
#    1. Compute the climatology using loops.
#    1. Compute, then save.
#
#  Further optimization might be:
#   1. For climatology, prepare a dataset using `day_365` calendar, then read the dataset using chunks of 365 days.
#   2. Try to set up a Dask cluster using UCX.
#   3. Instead of point 1, rechunk using TimeResampler, after converting to calendar='365_day'. [not done, probably not effective]
#
# [1]: mean could be computed, using a reasonable approximation, as the mean of the climatology, and the std could reuse
# the value of the mean.
import logging
import pathlib

import click

from graphcast.dask_distributed_utils import get_client
from graphcast.dataset_utils import open_dataset_wo_static, save_to_zarr
from graphcast.stats_utils import Stats, StatsRegistry
from graphcast.cli_utils import DictParamType

logger = logging.getLogger(__name__)


@click.group()
def cli():
  pass


@cli.command()
@click.argument("stats",
                required=True,
                type=click.Choice(StatsRegistry.keys()))
@click.argument("input",
                required=True,
                type=click.Path(path_type=pathlib.Path, file_okay=True, dir_okay=True, exists=True, readable=True))
@click.argument("output",
                required=True,
                type=click.Path(path_type=pathlib.Path, file_okay=True, dir_okay=True, writable=True))
@click.option("--local/--no-local",
              default=False,
              help="Whether to use Dask LocalCluster.",
              show_default=True)
@click.option("--time-dim",
              default="time",
              help="Name of the time dimension to average over.",
              show_default=True)
@click.option("--chunks",
              default=None,
              show_default=True,
              type=DictParamType(),
              help="String containing chunking specs used when reading.")
@click.option("--compressor-name",
              "cname",
              default="lz4",
              show_default=True,
              help="Name of the compressor to use.")
@click.option("--compressor-level",
              "clevel",
              default=1,
              show_default=True,
              help="Compressor level to use.")
@click.option("--skipna/--no-skipna",
              default=False,
              help="Whether to skip NaNs when averaging.",
              show_default=True)
@click.option("--overwrite/--no-overwrite",
              default=False,
              is_flag=True,
              help="Whether to overwrite existing outputs.")
@click.option("--debug/--no-debug",
              default=False,
              is_flag=True,
              help="Whether to use the serial Dask scheduler.")
@click.option("--log-level",
              default='info',
              type=click.Choice(['debug', 'info', 'warning', 'error', 'critical'], case_sensitive=False),
              show_default=True)
def compute(stats: Stats,
            input: pathlib.Path,
            output: pathlib.Path,
            local: bool = False,
            time_dim: str = "time",
            chunks: str | None = None,
            skipna: bool = False,
            debug: bool = False,
            log_level: str = "info",
            overwrite: bool = False,
            cname: str = "lz4",
            clevel: int = 1):
  """
  Compute the mean, std, and diff std over time.

  Parameters
  ----------
  stats : str
      Type of stats to compute, one of "climatology", "mean", "std", or "diff_std.
  input : pathlib.Path
      Path to the input Zarr, or directory of zipped Zarrs.
  output : pathlib.Path
      Path to the directory of output stats.
  time_dim : str, default "time"
      Name of the time dimension to reduce or group by.
  skipna : bool, default False
      Whether to skip NaNs when computing the mean.
  log_level : {"debug","info","warning","error","critical"}, default "info"
      Logging verbosity.
  overwrite : bool, default False
      Whether to overwrite the output_path if it already exists.
  """

  # Configure logging
  logging.basicConfig(
    format='%(levelname)s - %(asctime)s: %(message)s',
    datefmt='%Y-%m-%dT%H:%M:%S',
    level=getattr(logging, log_level.upper()))

  client = get_client(logger=logger, debug=debug, local=local)

  if not input.exists():
    raise ValueError(f"Input path {input} does not exist")

  # Ensure output dir exists
  output.parent.mkdir(parents=True, exist_ok=True)

  logger.info(f"Opening input dataset from {input} with chunks={chunks}")
  # What is the effect (performance-wise) of chunking at all?
  dataset = open_dataset_wo_static(input, time_dim=time_dim, chunks=chunks)

  # Computing the climatology beforehand could be a source of optimization for large datasets, but it introduces a small numerical error.
  # Say you have a collection of values {x_i} and labels {l_i} so that each label corresponds to multiple values.
  # Then you can compute the mean of the whole collection x_mean, or the mean of the averages for each label x_clim_mean.
  # If the number of values for each label is the same, then the two quantities are strictly equal. But it's not true in general.
  # Indeed, in the case of a daily climatology, the two would differ because of leap years, which would introduce a relative error of
  # the order of 1/365. Finally, the dataset might not start on the 1st of January or end before the 31 of December,
  # which would further distort the results. However, the quantities computed here are used to standardize the input features, and
  # therefore such approximations are reasonably acceptable.

  logger.info(f"Computing daily {stats} over dimension '{time_dim}' (skipna={skipna}), saving to {output}")
  stats_ds = StatsRegistry[stats](dataset, time_dim=time_dim, skipna=skipna, keep_attrs=True)
  # At the beginning of `write_dataset_serial`, stats_ds is computed, meaning that there must be enough memory
  # available to the client process to hold stats_ds in memory. This could be a problem for climatology in some cases.
  save_to_zarr(stats_ds, output, overwrite=overwrite, compressor_kwargs=dict(cname=cname, clevel=clevel))

  client.close()


if __name__ == "__main__":
  cli()
