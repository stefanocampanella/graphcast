import argparse
import logging
from datetime import datetime
from pandas import Timedelta
from pathlib import Path

from graphcastmodel.download_utils import open_dataset
from graphcastmodel.graphcast import ALL_VOLUME_VARS, ALL_SURFACE_VARS

parser = argparse.ArgumentParser(prog=__file__)
parser.add_argument('start', help='Start datetime', type=datetime.fromisoformat)
parser.add_argument('delta', help='Number of steps', type=Timedelta)
parser.add_argument('output', help='Output file', type=Path)
parser.add_argument('--log', default='INFO', help='Log level')
args = parser.parse_args()

logger = logging.getLogger(__name__)
cm_console_handler = logging.getHandlerByName('console')
logging.getLogger('copernicus_marine_root_logger').removeHandler(cm_console_handler)
logging.basicConfig(level=args.log.upper())

start_datetime = args.start
end_datetime = start_datetime + args.delta
logger.info(f"Downloading all data from {start_datetime} to {end_datetime}")
ds = open_dataset(
  ALL_VOLUME_VARS + ALL_SURFACE_VARS,
  start_datetime=start_datetime,
  end_datetime=end_datetime)
ds.to_zarr(args.output)
