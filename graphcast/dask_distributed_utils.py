import functools
import inspect
import logging
import socket

import dask
import dask_mpi
import distributed

logger = logging.getLogger(__name__)

class DummyClient:

  def close(self):
    pass


MaybeClient = distributed.Client | DummyClient


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


@get_dask_env_options(suffix="mpi")
def dask_mpi_initialize(*args, **kwargs):
  return dask_mpi.initialize(*args, **kwargs)


@get_dask_env_options()
def LocalCluster(*args, **kwargs):
  return distributed.LocalCluster(*args, **kwargs)


def get_client(local=False, debug=False) -> MaybeClient:

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
