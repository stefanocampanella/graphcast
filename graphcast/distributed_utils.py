import functools
import inspect
import os
import socket

import distributed
from mpi4py import MPI
import dask
import dask_mpi
import logging
import pathlib


class DummyClient:

  def close(self):
    pass


MaybeClient = distributed.Client | DummyClient


def get_distributed_log_name(log_name=None) -> str:
  world_size = MPI.COMM_WORLD.Get_size()
  rank = MPI.COMM_WORLD.Get_rank()
  log_name = log_name or "distributed"
  if world_size > 1:
    if rank == 0:
      log_name += f"_scheduler"
    elif rank == 1:
      log_name += f"_client"
    else:
      log_name += f"_worker_{rank}"
  log_name += ".log"

  return log_name


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


def set_log_handler(logger_name=None,
                    log_dir=None,
                    log_name=None,
                    fmt='%(levelname)s - %(asctime)s: %(message)s',
                    datefmt='%Y-%m-%dT%H:%M:%S') -> None:

  if logger_name is not None:
    logger = logging.getLogger(logger_name)
  else:
    raise ValueError("logger_name must be specified")

  if log_name is not None:
    job_name = os.getenv("SLURM_JOB_NAME")
    job_id = os.getenv("SLURM_JOB_ID")
    if job_name is not None and job_id is not None:
      log_name = f"{job_name}-{job_id}"
    else:
      log_name = None

  if log_dir is not None:
    log_dir = pathlib.Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
  else:
    log_dir = pathlib.Path.cwd()

  logger.propagate = False
  logger.handlers.clear()
  log_path = log_dir / get_distributed_log_name(log_name=log_name)
  filehandler = logging.FileHandler(log_path, mode="w")
  formatter = logging.Formatter(fmt=fmt, datefmt=datefmt)
  filehandler.setFormatter(formatter)
  logger.addHandler(filehandler)


@get_dask_env_options(suffix="mpi", inherit_params_from=(set_log_handler, dask_mpi.initialize))
def dask_mpi_initialize(*args, **kwargs):
  set_log_handler_kwargs = filter_kwargs(kwargs, set_log_handler)
  set_log_handler(logger_name="distributed", **set_log_handler_kwargs)
  dask_mpi_initialize_kwargs = filter_kwargs(kwargs, dask_mpi.initialize)
  return dask_mpi.initialize(*args, **dask_mpi_initialize_kwargs)


@get_dask_env_options(inherit_params_from=(set_log_handler, distributed.LocalCluster))
def LocalCluster(*args, **kwargs):
  set_log_handler_kwargs = filter_kwargs(kwargs, set_log_handler)
  set_log_handler(logger_name="distributed", **set_log_handler_kwargs)
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
