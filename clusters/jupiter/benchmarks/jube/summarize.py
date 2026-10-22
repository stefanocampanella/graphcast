"""Summarize a results directory of run_final.sh into summary_jobs.csv and summary_cases.csv.

A job is "ok" if rank 0 finished training after 1024 steps (also when its training step was cancelled after hanging
at exit), "oom" if a Grain worker was killed (exit code -9, lack of host memory), "stopped" if it was cancelled after
at least 100 steps (its statistics cover the steps done), and "failed" otherwise.
Case metrics in summary_cases.csv are means over the ok and stopped repetitions. Runs with the system Python (3.9).
Usage: python3 summarize.py results/<date>-<name>   (after jube result for every experiment)
"""

import csv
import glob
import os
import re
import statistics
import sys

METRICS = [
  "total_avg[s]", "total_std[s]", "total_min[s]", "data_avg[s]", "compute_avg[s]", "compute_min[s]",
  "first_step_elapsed[s]", "train_samples_per_s", "node_mem_peak[GB]", "node_mem_avail_min[GB]",
  "node_shm_peak[GB]", "gpu_mem_peak[GB]", "gpu_util[%]", "gpu_power[W]",
]
CASE_KEYS = [
  "experiment", "format", "workers", "threads", "zarr_pool", "zarr_conc", "nodes", "cpus", "levels", "mesh",
  "msg_steps", "opt",
]


def to_float(value):
  try:
    return float(value)
  except (TypeError, ValueError):
    return None


def read(path):
  with open(path, errors="replace") as file:
    return file.read()


STEP_LINE = re.compile(r"Step (\d+) timings: data ([\d.]+) s, compute ([\d.]+) s")


def rank_statistics(work):
  """Mean over steps 10+ of the longest data wait of any rank, and the median compute time of rank 0.

  The JUBE patterns read rank 0 only. Steps synchronize all ranks (gradient all-reduce), so a data stall of another
  rank shows up as compute time of rank 0: data_stall separates it, compute_median is robust to it.
  """
  data = {}
  compute0 = []
  for path in glob.glob(os.path.join(work, "task-*.err")):
    rank0 = path.endswith("task-0.err")
    for step, data_s, compute_s in STEP_LINE.findall(read(path)):
      if int(step) >= 10:
        data[int(step)] = max(data.get(int(step), 0.0), float(data_s))
        if rank0:
          compute0.append(float(compute_s))
  stall = f"{statistics.mean(data.values()):.4g}" if data else ""
  median = f"{statistics.median(compute0):.4g}" if compute0 else ""
  return stall, median


def job_rows(root):
  for result in sorted(glob.glob(os.path.join(root, "*", "000000", "result", "result_csv.dat"))):
    experiment = result.split(os.sep)[-4]
    with open(result) as file:
      for row in csv.DictReader(file):
        work = os.path.join(root, experiment, "000000", f"{int(row['jube_wp_id']):06d}_submit", "work")
        bench = read(os.path.join(work, "bench.toml"))
        finished = "Training finished" in read(os.path.join(work, "task-0.err"))
        oom = any("exit code -9" in read(path) for path in glob.glob(os.path.join(work, "task-*.err")))
        steps = to_float(row["step_last"]) or 0
        fields = row["case"].split(":")
        mesh = re.search(r'(?m)^path = "/data/geometry/uniform_(\w+)\.msh"', bench)
        msg_steps = re.search(r"(?m)^gnn_msg_steps = (\d+)", bench)
        out = {
          "experiment": experiment, "wp": row["jube_wp_id"], "slurm_job": row["slurm_job"],
          "outcome": "ok" if finished and steps >= 1023 else "oom" if oom else "stopped" if steps >= 100 else "failed",
          "status": row["status"], "format": row["format"], "workers": row["num_workers"],
          "threads": row["num_threads"], "zarr_pool": row["zarr_max_workers"], "zarr_conc": row["zarr_concurrency"],
          "nodes": row["nodes"], "cpus": row["cpus_per_task"],
          "levels": "all" if re.search(r'(?m)^levels = "all"', bench) else "10",
          "mesh": mesh.group(1) if mesh else "", "msg_steps": msg_steps.group(1) if msg_steps else "",
          "opt": fields[7] if len(fields) > 7 else "default", "rep": row["rep"], "steps": row["step_last"],
        }
        out.update({metric: row.get(metric, "") for metric in METRICS})
        out["data_stall_avg[s]"], out["compute_median[s]"] = rank_statistics(work)
        yield out


def main(root):
  jobs = list(job_rows(root))
  with open(os.path.join(root, "summary_jobs.csv"), "w", newline="") as file:
    writer = csv.DictWriter(file, list(jobs[0]))
    writer.writeheader()
    writer.writerows(jobs)
  groups = {}
  for job in jobs:
    groups.setdefault(tuple(job[key] for key in CASE_KEYS), []).append(job)
  cases = []
  for key, group in groups.items():
    ok = [job for job in group if job["outcome"] in ("ok", "stopped")]
    case = dict(zip(CASE_KEYS, key))
    case.update(runs=len(group), ok=len(ok), outcomes="/".join(job["outcome"] for job in sorted(group, key=lambda j: j["rep"])))
    for metric in METRICS + ["data_stall_avg[s]", "compute_median[s]"]:
      values = [to_float(job[metric]) for job in ok if to_float(job[metric]) is not None]
      case[metric] = f"{statistics.mean(values):.4g}" if values else ""
    cases.append(case)
  with open(os.path.join(root, "summary_cases.csv"), "w", newline="") as file:
    writer = csv.DictWriter(file, list(cases[0]))
    writer.writeheader()
    writer.writerows(cases)
  outcomes = {}
  for job in jobs:
    outcomes[job["outcome"]] = outcomes.get(job["outcome"], 0) + 1
  print(f"{len(jobs)} jobs {outcomes}, {len(cases)} cases")
  for job in jobs:
    if job["outcome"] not in ("ok", "stopped") and not (job["outcome"] == "oom" and job["levels"] == "all"):
      print(f"  check: {job['experiment']} wp {job['wp']} {job['format']} {job['workers']}x{job['threads']} "
            f"{job['opt']}: {job['outcome']} status {job['status']} steps {job['steps']}")


if __name__ == "__main__":
  main(sys.argv[1])
