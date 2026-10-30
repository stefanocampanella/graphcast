# SPDX-FileCopyrightText: 2026 Stefano Campanella
# SPDX-License-Identifier: MIT
import click
from absl import flags

from graphcast.basis import compute
from graphcast.downloader import compute_tendency_std, download, download_mask
from graphcast.geospatial_mesh import make_ocean_mesh
from graphcast.training import train


@click.group()
@click.option(
  "--grain-worker-profiling/--grain-no-worker-profiling",
  default=False,
  type=bool,
  help="Start profiler servers in Grain multiprocessing workers.",
)
@click.option(
  "--grain-traceback-filtering",
  default="off",
  type=str,
  help="How grain should handle traceback filtering",
)
@click.option(
  "--grain-debug-mode/--grain-no-debug-mode",
  default=False,
  type=bool,
  help="Enable Grain debug mode",
)
def cli(
    grain_worker_profiling: bool,
    grain_traceback_filtering: str,
    grain_debug_mode: bool,
):
  # Grain reads absl flags, which raise if accessed before parsing (normally done by absl.app.run).
  flags.FLAGS.mark_as_parsed()
  flags.FLAGS.grain_enable_multiprocess_worker_profiling = grain_worker_profiling
  flags.FLAGS.grain_py_traceback_filtering = grain_traceback_filtering
  flags.FLAGS.grain_py_debug_mode = grain_debug_mode


cli.add_command(train)
cli.add_command(make_ocean_mesh)
cli.add_command(compute)
cli.add_command(download)
cli.add_command(download_mask)
cli.add_command(compute_tendency_std)

if __name__ == "__main__":
  cli()
