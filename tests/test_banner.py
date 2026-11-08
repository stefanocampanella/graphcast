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
"""Tests of the banner the command line prints first (mobydick.banner)."""

import io

import pytest
from click.testing import CliRunner

from mobydick import banner, cli


@pytest.fixture(autouse=True)
def single_process(monkeypatch):
  """No launcher rank and no opt-out, whatever the environment the tests run in (e.g. a Slurm job)."""
  for variable in (*banner.RANK_VARIABLES, banner.NO_BANNER_VARIABLE):
    monkeypatch.delenv(variable, raising=False)


def test_banner_has_the_version_and_waves_as_wide_as_the_text(monkeypatch):
  """The version is filled in, and the wave lines match the text line whatever the version's length (a fixed number
  of `~` in the asset would be too short or too long for some versions)."""
  for version in ("0.1", "12.34.56.dev7"):
    monkeypatch.setattr(banner, "version", lambda version=version: version)
    lines = banner.banner().splitlines()
    assert len(lines) == 3
    assert f"Moby-Dick {version} · " in lines[1]
    assert lines[0] == lines[2] == "~" * len(lines[1])


def test_version_without_an_installed_distribution(monkeypatch):
  """Sources run from PYTHONPATH (as in an image built before the rename) have no moby-dick metadata: the banner says
  "dev" instead of failing every command."""
  monkeypatch.setattr(banner, "DISTRIBUTION", "moby-dick-not-installed")
  assert banner.version() == "dev"


def test_help_prints_the_banner_to_stderr_only():
  """`moby-dick --help` (and so every command, as the group prints it first) writes the banner to stderr; stdout holds
  only the help, so the output of commands can be piped."""
  result = CliRunner().invoke(cli, ["--help"])
  assert result.exit_code == 0
  assert "Moby-Dick" in result.stderr
  assert "~~~" not in result.stdout
  assert "Usage:" in result.stdout


def test_usage_errors_print_the_banner_too():
  """The banner comes before click parses the arguments, so a wrong command still shows which program answered."""
  result = CliRunner().invoke(cli, ["no-such-command"])
  assert result.exit_code != 0
  assert "Moby-Dick" in result.stderr


@pytest.mark.parametrize("variable", banner.RANK_VARIABLES)
def test_only_the_first_process_prints(monkeypatch, capsys, variable):
  """Under a parallel launcher every process runs the command line: only rank 0 prints, or a 32-process training job
  would log the banner 32 times."""
  monkeypatch.setenv(variable, "3")
  banner.print_banner()
  assert capsys.readouterr().err == ""
  monkeypatch.setenv(variable, "0")
  banner.print_banner()
  assert "Moby-Dick" in capsys.readouterr().err


def test_the_banner_can_be_turned_off(monkeypatch, capsys):
  """MOBYDICK_NO_BANNER (any value but empty or 0) silences it, e.g. for scripts that parse stderr."""
  monkeypatch.setenv(banner.NO_BANNER_VARIABLE, "1")
  banner.print_banner()
  assert capsys.readouterr().err == ""
  monkeypatch.setenv(banner.NO_BANNER_VARIABLE, "0")
  banner.print_banner()
  assert "Moby-Dick" in capsys.readouterr().err


def test_a_stream_that_cannot_encode_the_banner_gets_replacements(monkeypatch):
  """On an ASCII stderr the banner's middle dot becomes a replacement character instead of raising
  UnicodeEncodeError before the command runs."""
  stream = io.TextIOWrapper(io.BytesIO(), encoding="ascii")
  monkeypatch.setattr("sys.stderr", stream)
  banner.print_banner()
  stream.seek(0)
  assert "Moby-Dick" in stream.read()
