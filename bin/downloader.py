import argparse
from datetime import datetime
from pathlib import Path
from graphcastmodel.graphcast import DEPTHS_37 as DEPTHS
from graphcastmodel.download_utils import open_dataset
import logging

parser = argparse.ArgumentParser(prog=__file__)
parser.add_argument('datetime_start', help='Start datetime', type=datetime.fromisoformat)
parser.add_argument('steps', help='Number of steps', type=int)
parser.add_argument('output', help='Output file', type=Path)
parser.add_argument('--log', default='INFO', help='Log level')

logger = logging.getLogger(__name__)
copernicus_logging_console_handler = logging.getHandlerByName('console')
logging.getLogger('copernicus_marine_root_logger').removeHandler(copernicus_logging_console_handler)
logging.basicConfig(level=parser.log.upper())

start_datetime = datetime.fromisoformat('2000-01-01')
end_datetime = datetime.fromisoformat('2000-01-02')
ds = open_dataset(("fpco2", ), list(DEPTHS), start_datetime=start_datetime, end_datetime=end_datetime)
