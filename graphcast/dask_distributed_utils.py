import functools
import inspect
import logging
import socket

import dask
import dask_mpi
import distributed
from mpi4py import MPI

from graphcast.cli_utils import get_distributed_logger


class DummyClient:

  def close(self):
    pass


MaybeClient = distributed.Client | DummyClient


def get_distributed_log_suffix() -> str:
  world_size = MPI.COMM_WORLD.Get_size()
  rank = MPI.COMM_WORLD.Get_rank()
  log_suffix = ""
  if world_size > 1:
    if rank == 0:
      log_suffix += f"_scheduler"
    elif rank == 1:
      log_suffix += f"_client"
    else:
      log_suffix += f"_worker_{rank}"
  log_suffix += ".log"

  return log_suffix


def get_dask_env_options(suffix=None, inherit_params_from=None):

  def decorator(func):

    @functools.wraps(func)
    def wrapped(*args, **kwargs):
      dask_config = dask.config.collect_env()
      func_config = dask_config.get(suffix or func.__name__.lower(), {})
      parameters = []
      if inherit_params_from is not None:
        for f in inherit_params_from:
          parameters.extend(inspect.signature(f).parameters.values())
      else:
        parameters.extend(inspect.signature(func).parameters.values())
      for p in parameters:
        if p.name in func_config:
          kwargs[p.name] = func_config[p.name]
      return func(*args, **kwargs)

    return wrapped

  return decorator


def filter_kwargs(kwargs, func):
  filtered = {p.name: kwargs[p.name] for p in inspect.signature(func).parameters.values() if p.name in kwargs}
  return filtered


@get_dask_env_options(suffix="mpi", inherit_params_from=(get_distributed_logger, dask_mpi.initialize))
def dask_mpi_initialize(*args, **kwargs):
  set_log_handler_kwargs = filter_kwargs(kwargs, get_distributed_logger)
  get_distributed_logger(logger_name="distributed", log_suffix_fn=get_distributed_log_suffix, **set_log_handler_kwargs)
  dask_mpi_initialize_kwargs = filter_kwargs(kwargs, dask_mpi.initialize)
  return dask_mpi.initialize(*args, **dask_mpi_initialize_kwargs)


@get_dask_env_options(inherit_params_from=(get_distributed_logger, distributed.LocalCluster))
def LocalCluster(*args, **kwargs):
  set_log_handler_kwargs = filter_kwargs(kwargs, get_distributed_logger)
  get_distributed_logger(logger_name="distributed", log_name="distributed", **set_log_handler_kwargs)
  local_cluster_kwargs = filter_kwargs(kwargs, distributed.LocalCluster)
  return distributed.LocalCluster(*args, **local_cluster_kwargs)


def get_client(logger=None, local=False, debug=False) -> MaybeClient:

  logger = logger or logging.getLogger(__name__)

  if debug:
    dask.config.set(scheduler="synchronous")

    client = DummyClient()
    logger.info("Using synchronous Dask scheduler.")
  elif local:
    # FIXME: LocalCluster logging is not working as intended.
    cluster = distributed.LocalCluster()
    client = distributed.Client(cluster)
    logger.info("Using local Dask cluster")
  else:
    dask_mpi_initialize()
    client = distributed.Client()
    host = client.run_on_scheduler(socket.gethostname)
    port = client.scheduler_info()['services']['dashboard']
    logger.info(f"Using dask_mpi, Dask dashboard available at {host}:{port}")

  return client
