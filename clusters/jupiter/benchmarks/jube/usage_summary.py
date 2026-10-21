"""Summarize the node memory and GPU samples of usage_sampler.sh as BENCH_* lines for the JUBE analysis.

Memory peaks cover the whole job; GPU utilization and power are averaged over steady-state steps (10 and later, as
the step statistics), or over the whole job if training stopped earlier. Runs with the system Python (3.9).
Usage: python3 usage_summary.py WORK_DIR
"""

import csv
import glob
import math
import os
import re
import sys
import time

NAN = float("nan")
STEP_LINE = re.compile(r"(\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d): Step (\d+) timings")


def to_float(value):
  try:
    return float(value)
  except (TypeError, ValueError):
    return NAN  # "[N/A]" from nvidia-smi, or a sample cut short when the sampler was killed.


def read_csv(pattern):
  rows = []
  for path in sorted(glob.glob(pattern)):
    with open(path) as file:
      rows.extend({key: to_float(value) for key, value in row.items()} for row in csv.DictReader(file))
  return [row for row in rows if not math.isnan(row["time"])]


def steady_window(log_path):
  """Times of the first steady-state step (10) and the last step logged by rank 0, or None."""
  times = {}
  try:
    with open(log_path, errors="replace") as file:
      for line in file:
        match = STEP_LINE.search(line)
        if match:
          times[int(match.group(2))] = time.mktime(time.strptime(match.group(1), "%Y-%m-%dT%H:%M:%S"))
  except OSError:
    return None
  if 10 not in times:
    return None
  return times[10], max(times.values())


def finite(values):
  return [value for value in values if not math.isnan(value)]


def mean(values):
  values = finite(values)
  return sum(values) / len(values) if values else NAN


def peak(values):
  values = finite(values)
  return max(values) if values else NAN


def low(values):
  values = finite(values)
  return min(values) if values else NAN


def main(work_dir):
  memory = read_csv(os.path.join(work_dir, "node_mem-*.csv"))
  gpus = read_csv(os.path.join(work_dir, "gpu-*.csv"))
  if memory:
    print(f"BENCH_NODE_MEM_PEAK: {peak([r['used_gb'] for r in memory]):.1f} GB")
    print(f"BENCH_NODE_MEM_AVAIL_MIN: {low([r['available_gb'] for r in memory]):.1f} GB")
    print(f"BENCH_NODE_SHM_PEAK: {peak([r['shared_gb'] for r in memory]):.1f} GB")
  if gpus:
    print(f"BENCH_GPU_MEM_PEAK: {peak([r['memory_used_mib'] for r in gpus]) / 1024:.1f} GB")
    window = steady_window(os.path.join(work_dir, "task-0.err"))
    steady = [r for r in gpus if window is None or window[0] <= r["time"] <= window[1]] or gpus
    print(f"BENCH_GPU_UTIL: {mean([r['utilization_gpu_pct'] for r in steady]):.1f} %")
    print(f"BENCH_GPU_POWER: {mean([r['power_w'] for r in steady]):.0f} W")


if __name__ == "__main__":
  main(sys.argv[1] if len(sys.argv) > 1 else ".")
