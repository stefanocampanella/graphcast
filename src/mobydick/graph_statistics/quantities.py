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
"""How statistics are represented in the JSON output.

Every quantity is a dict with a "description": the other fields of a dict value (e.g. a summary's mean, std, ...), or
"value" for anything else. Distributions are summaries (`summarize`), possibly with a histogram.
"""

from typing import Any

import numpy as np

# The fields of a summary, in output order.
SUMMARY_KEYS = ("mean", "std", "min", "p25", "p50", "p75", "max")


def described(description: str, value: Any) -> dict[str, Any]:
  """A quantity of the output: the fields of a dict value, or any other value under "value", with a description."""
  if isinstance(value, dict):
    return {"description": description, **value}
  return {"description": description, "value": value}


def summarize(values: np.ndarray) -> dict[str, float]:
  """Mean, standard deviation, minimum, quartiles (25th, 50th and 75th percentiles) and maximum (NaN for no values)."""
  values = np.asarray(values, dtype=np.float64)
  if values.size == 0:
    return dict.fromkeys(SUMMARY_KEYS, float("nan"))
  p25, p50, p75 = np.percentile(values, [25, 50, 75])
  return {
    "mean": float(values.mean()),
    "std": float(values.std()),
    "min": float(values.min()),
    "p25": float(p25),
    "p50": float(p50),
    "p75": float(p75),
    "max": float(values.max()),
  }


def histogram(values: np.ndarray, bins: np.ndarray) -> dict[str, list]:
  """Counts of the values in the bins (`bin_edges` has one more entry than `counts`)."""
  counts, edges = np.histogram(values, bins=bins)
  return {"bin_edges": edges.tolist(), "counts": counts.tolist()}


def degree_summary(degrees: np.ndarray) -> dict[str, Any]:
  """Summary of integer degrees, with the number of nodes of each degree (index = degree)."""
  return {**summarize(degrees), "nodes_by_degree": np.bincount(degrees).tolist()}


def fraction(selected: np.ndarray) -> dict[str, float]:
  """Number and percentage of the selected nodes (a boolean array over all nodes of a set)."""
  return {"count": int(selected.sum()), "percent": float(100 * selected.mean()) if selected.size else float("nan")}
