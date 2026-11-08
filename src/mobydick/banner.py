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
"""The banner every command prints first, from the package asset `assets/banner.txt`.

The asset is a template: `{version}` becomes the installed version of the distribution, and lines made only of `~`
are drawn as wide as the widest other line, so the waves fit any version string.
"""

import importlib.metadata
import importlib.resources
import os
import sys

DISTRIBUTION = "moby-dick"
# Set to a non-empty value other than 0 to print no banner.
NO_BANNER_VARIABLE = "MOBYDICK_NO_BANNER"
# Ranks of the process in a parallel launch (Slurm, Open MPI, MPICH/PMI, torchrun-like launchers), checked in order.
RANK_VARIABLES = ("SLURM_PROCID", "OMPI_COMM_WORLD_RANK", "PMI_RANK", "RANK")


def version() -> str:
  """The installed version, or "dev" for sources run without an installed distribution (e.g. from PYTHONPATH)."""
  try:
    return importlib.metadata.version(DISTRIBUTION)
  except importlib.metadata.PackageNotFoundError:
    return "dev"


def banner() -> str:
  template = importlib.resources.files("mobydick").joinpath("assets/banner.txt").read_text(encoding="utf-8")
  lines = template.format(version=version()).splitlines()
  width = max((len(line) for line in lines if set(line) != {"~"}), default=0)
  return "\n".join("~" * width if set(line) == {"~"} else line for line in lines) + "\n"


def is_main_process() -> bool:
  """Whether this is the first process of a parallel launch (or not a parallel launch).

  The command line runs before jax.distributed is initialized, so the rank comes from the launcher's environment.
  """
  for variable in RANK_VARIABLES:
    rank = os.environ.get(variable)
    if rank is not None:
      return rank.strip() == "0"
  return True


def print_banner() -> None:
  """Print the banner to stderr (stdout stays for the commands' output), once per parallel launch."""
  if os.environ.get(NO_BANNER_VARIABLE, "0") not in ("", "0") or not is_main_process():
    return
  stream = sys.stderr
  encoding = getattr(stream, "encoding", None) or "utf-8"
  # A stream that cannot encode a character (e.g. an ASCII locale) gets a replacement instead of an error.
  stream.write(banner().encode(encoding, errors="replace").decode(encoding))
  stream.flush()
