import inspect

from mpi4py import MPI
import dask
import dask_mpi
import logging
import pathlib

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


def set_distributed_log_handler(log_name=None,
                                fmt='%(levelname)s - %(asctime)s: %(message)s',
                                datefmt='%Y-%m-%dT%H:%M:%S') -> None:
  dask_config = dask.config.collect_env()
  dask_mpi_config = dask_config.get("mpi", {})

  log_dir = dask_mpi_config.get("log_dir", None)
  if log_dir is not None:
    log_dir = pathlib.Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
  else:
    log_dir = pathlib.Path.cwd()

  distributed_logger = logging.getLogger("distributed")
  distributed_logger.propagate = False
  distributed_logger.handlers.clear()
  distributed_log_path = log_dir / get_distributed_log_name(log_name=log_name)
  distributed_log_filehandler = logging.FileHandler(distributed_log_path, mode="w")
  formatter = logging.Formatter(fmt=fmt, datefmt=datefmt)
  distributed_log_filehandler.setFormatter(formatter)
  distributed_logger.addHandler(distributed_log_filehandler)


def dask_mpi_initialize():
  dask_config = dask.config.collect_env()
  dask_mpi_config = dask_config.get("mpi", {})
  dask_mpi_initialize_kwargs = {}
  for parameter in inspect.signature(dask_mpi.initialize).parameters.values():
    if parameter.name in dask_mpi_config:
      dask_mpi_initialize_kwargs[parameter.name] = dask_mpi_config[parameter.name]
  dask_mpi.initialize(**dask_mpi_initialize_kwargs)