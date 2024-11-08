import argparse
import logging
import pathlib
import tomllib

import gcsfs

parser = argparse.ArgumentParser(prog=__file__,
                                 description="Simple script to download ERA5 data from the WeatherBench 2 GCS")
parser.add_argument("url", help="Path to configuration file", type=str)
parser.add_argument("dest", help="Path to output directory", type=pathlib.Path)
parser.add_argument("--overwrite", help="Whether to overwrite existing outputs", action='store_true')
parser.add_argument('--log', default='info')
args = parser.parse_args()

logger = logging.getLogger(__name__)
logging.basicConfig(format='%(levelname)s - %(asctime)s: %(message)s',
                    datefmt='%Y-%m-%dT%H:%M:%S',
                    level=getattr(logging, args.log.upper()))

if args.dest.exists() and not args.overwrite:
  raise ValueError('Output destination already exists')

with args.config.open('rb') as file:
  configs = tomllib.load(file)

fs = gcsfs.GCSFileSystem(token='anon', access='read_only', consistency='md5')

if __name__ == '__main__':

  logger.info(f"Downloading from {args.url} to {args.dest}")
  # FIXME: On Leonardo not casting `dest` to a string raise an exception, however it cannot be reproduced locally.
  fs.get(args.url, args.dest.as_posix(), recursive=True)

