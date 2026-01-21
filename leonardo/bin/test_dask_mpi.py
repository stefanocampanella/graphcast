#!/usr/bin/env python3
"""
Test a Dask cluster launched with dask_mpi.

This script connects to a running Dask scheduler (typically started by dask-mpi
under MPI/Slurm) and performs a set of sanity checks to ensure scheduler and
workers can communicate and execute tasks.

Usage examples:
  - Using a scheduler file written by dask-mpi:
      srun -n 1 dask-mpi --scheduler-file scheduler.json &
      python leonardo/bin/test_dask_mpi.py --scheduler-file scheduler.json --min-workers 1

  - Using a known scheduler address:
      python leonardo/bin/test_dask_mpi.py --scheduler-address tcp://scheduler-host:8786 --min-workers 4

Environment variables:
  DASK_SCHEDULER_ADDRESS  If set and no CLI option is provided, this address is used.

Exit codes:
  0 = success, all checks passed
  1 = failure, an error occurred or a check failed
"""

import time
import os
from typing import Tuple

import click
import mpi4py
from graphcast.dask_distributed_utils import get_distributed_logger, dask_mpi_initialize
from distributed import Client
from distributed.utils import TimeoutError as DaskTimeoutError



@click.command()
@click.option("--timeout", "-t", type=float, default=60.0, show_default=True,
              help="Seconds to wait for workers.")
@click.option("--log-level",
              default='info',
              type=click.Choice(['debug', 'info', 'warning', 'error', 'critical'], case_sensitive=False),
              show_default=True)
def cli(log_level: str, timeout: float) -> int:
  """Validate a running Dask cluster launched with dask-mpi.

  Returns
  -------
  int
      0 on success, 1 on failure.
  """
  # Configure distributed log handler
  job_name = os.getenv("SLURM_JOB_NAME")
  job_id = os.getenv("SLURM_JOB_ID")
  if job_name is not None and job_id is not None:
    log_name = f"{job_name}_{job_id}"
  else:
    log_name = "dask_mpi_test"
  get_distributed_logger(log_name=log_name)

  try:
    dask_mpi_initialize()

    client = Client()
    with client:
      check_workers(client, timeout)
      check_execution(client)
      addr, n_workers, total_threads = _get_info(client)
      click.echo("Dask MPI cluster OK")
      click.echo(f"- scheduler: {addr}")
      click.echo(f"- number of workers: {n_workers}")
      click.echo(f"- total threads: {total_threads}")
      click.echo("Shutting down the cluster.")
      client.close()
    return 0
  except Exception as e:
    click.echo(f"ERROR: {e}", err=True)
    return 1


def _get_info(client: Client) -> Tuple[str, int, int]:
  scheduler_info = client.scheduler_info(n_workers=-1)
  n_workers = len(scheduler_info["workers"])
  total_threads = sum(int(worker.get("nthreads", 0)) for worker in scheduler_info["workers"].values())
  addr = scheduler_info["address"]
  return addr, n_workers, total_threads


def check_workers(client: Client, timeout: float) -> None:
  n_workers = mpi4py.MPI.COMM_WORLD.Get_size() - 2
  t0 = time.time()
  try:
    client.wait_for_workers(n_workers, timeout=timeout)
  except DaskTimeoutError:
    n = len(client.scheduler_info().get("workers", {}))
    raise RuntimeError(f"Timed out waiting for {n_workers} workers (have {n}).")
  dt = time.time() - t0

  # Basic per-worker sanity: nthreads>=1
  info = client.scheduler_info()
  workers = info.get("workers", {})
  bad = [w for w, d in workers.items() if int(d.get("nthreads", 0)) < 1]
  if bad:
    raise RuntimeError(f"Some workers report nthreads<1: {bad}")
  else:
    click.echo(f"All {n_workers} workers available (waited {dt:.2f}s)")


def check_execution(client: Client) -> None:
  # Simple broadcast to all workers
  res = client.run(lambda: 1)
  if not res or not all(v == 1 for v in res.values()):
    raise RuntimeError("client.run sanity check failed")

  # Distribute a small map and reduction
  def square(x: int) -> int:
    return x * x

  n_workers = len(client.scheduler_info()["workers"])
  xs = list(range(2 * n_workers + 1))
  futures = client.map(square, xs)
  total = client.submit(sum, futures).result()
  expected = sum(x * x for x in xs)
  if total != expected:
    raise RuntimeError(f"Computation mismatch: got {total}, expected {expected}")

  # Verify that data can be scattered/gathered
  scattered = client.scatter({"a": 123, "b": 456})
  gathered = client.gather(scattered)
  if gathered != {"a": 123, "b": 456}:
    raise RuntimeError("Scatter/gather failed")

  click.echo("Execution checks passed")


def main() -> int:
  try:
    # Run Click CLI without standalone mode so we can return exit code
    return cli(standalone_mode=False) or 0
  except SystemExit as exc:
    # In case click raises SystemExit, propagate the code
    return int(getattr(exc, "code", 1) or 0)
  except Exception as e:
    click.echo(f"ERROR: {e}", err=True)
    return 1


if __name__ == "__main__":
  raise SystemExit(main())
