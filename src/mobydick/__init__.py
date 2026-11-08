# SPDX-FileCopyrightText: 2026 Stefano Campanella
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS-IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import click
from absl import flags

from mobydick.banner import print_banner
from mobydick.basis import compute_grid_basis_command
from mobydick.downloader import compute_tendency_std, download, download_mask
from mobydick.geospatial_mesh import make_icosahedral_mesh, make_ocean_mesh
from mobydick.training import train


class BannerGroup(click.Group):
  """A command group that prints the banner before parsing its arguments: every subcommand, --help and usage errors.

  Shell completion makes contexts with resilient_parsing: no banner there, it would garble every completion.
  """

  def make_context(self, *args, **kwargs):
    if not kwargs.get("resilient_parsing"):
      print_banner()
    return super().make_context(*args, **kwargs)


@click.group(cls=BannerGroup)
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
  """Moby-Dick: data preparation, meshes and training of deep-learning ocean circulation models."""
  # Grain reads absl flags, which raise if accessed before parsing (normally done by absl.app.run).
  flags.FLAGS.mark_as_parsed()
  flags.FLAGS.grain_enable_multiprocess_worker_profiling = grain_worker_profiling
  flags.FLAGS.grain_py_traceback_filtering = grain_traceback_filtering
  flags.FLAGS.grain_py_debug_mode = grain_debug_mode


cli.add_command(train)
cli.add_command(make_ocean_mesh)
cli.add_command(make_icosahedral_mesh)
cli.add_command(compute_grid_basis_command)
cli.add_command(download)
cli.add_command(download_mask)
cli.add_command(compute_tendency_std)

if __name__ == "__main__":
  cli()
