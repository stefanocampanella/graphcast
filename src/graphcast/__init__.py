# SPDX-FileCopyrightText: 2026 Stefano Campanella
# SPDX-License-Identifier: MIT
import click

from graphcast.basis import compute
from graphcast.geospatial_mesh import make_ocean_mesh
from graphcast.training import train


@click.group()
def cli():
    pass


cli.add_command(train)
cli.add_command(make_ocean_mesh)
cli.add_command(compute)

if __name__ == "__main__":
    cli()
