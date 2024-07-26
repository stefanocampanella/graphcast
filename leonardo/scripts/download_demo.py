#! /usr/bin/env python

# Taken more or less verbatim from: 
# https://cloud.google.com/storage/docs/downloading-objects#storage-download-object-python

# Download all of the blobs in a bucket, concurrently in a process pool.

# The filename of each blob once downloaded is derived from the blob name and
# the `destination_directory `parameter. For complete control of the filename
# of each blob, use transfer_manager.download_many() instead.

# Directories will be created automatically as needed, for instance to
# accommodate blob names that include slashes.

# The ID of your GCS bucket
# bucket_name = "your-bucket-name"

# The directory on your computer to which to download all of the files. This
# string is prepended (with os.path.join()) to the name of each blob to form
# the full path. Relative paths and absolute paths are both accepted. An
# empty string means "the current working directory". Note that this
# parameter allows accepts directory traversal ("../" etc.) and is not
# intended for unsanitized end user input.
# destination_directory = ""

# The maximum number of processes to use for the operation. The performance
# impact of this value depends on the use case, but smaller files usually
# benefit from a higher number of processes. Each additional process occupies
# some CPU and memory resources until finished. Threads can be used instead
# of processes by passing `worker_type=transfer_manager.THREAD`.
# workers=8

# The maximum number of results to fetch from bucket.list_blobs(). This
# sample code fetches all of the blobs up to max_results and queues them all
# for download at once. Though they will still be executed in batches up to
# the processes limit, queueing them all at once can be taxing on system
# memory if buckets are very large. Adjust max_results as needed for your
# system environment, or set it to None if you are sure the bucket is not
# too large to hold in memory easily.
# max_results=1000

import argparse
import logging

from pathlib import Path

from google.cloud.storage import Client, transfer_manager

BUCKET_NAME="dm_graphcast"

parser = argparse.ArgumentParser(prog=__file__)
parser.add_argument("destination_directory", type=Path)
parser.add_argument("--workers", type=int, default=8)
parser.add_argument("--max_results", type=int, default=1000)
parser.add_argument("--log", help="Log level", default='info')
cli_args = parser.parse_args()

logging.basicConfig(format='%(levelname)s-%(asctime)s: %(message)s',
                    level=getattr(logging, cli_args.log.upper()))

if __name__ == '__main__':
    storage_client = Client.create_anonymous_client()
    bucket = storage_client.bucket(BUCKET_NAME)

    blob_names = [blob.name for blob in bucket.list_blobs(max_results=cli_args.max_results) if '$' not in blob.name]
    logging.info(f"Start downloading {BUCKET_NAME} bucket...")
    results = transfer_manager.download_many_to_path(
        bucket, 
        blob_names, 
        destination_directory=cli_args.destination_directory, 
        max_workers=cli_args.workers
    )

    for name, result in zip(blob_names, results):
        # The results list is either `None` or an exception for each blob in
        # the input list, in order.

        if isinstance(result, Exception):
            logging.warning("Failed to download {} due to exception: {}".format(name, result))
        else:
            logging.info("Downloaded {} to {}.".format(name, cli_args.destination_directory / name))
