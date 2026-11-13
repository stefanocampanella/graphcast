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
"""Statistics of the graphs of a model on a grid and a hierarchy of meshes (`moby-dick graph-statistics`).

A model on N meshes (mesh 0 the finest; today's models have N = 1) has 3N graphs, which `graphs.model_graphs` builds
as `leviathan.model.Leviathan` builds them:
- encoders: grid -> mesh 0 and mesh i-1 -> mesh i;
- processors: the edges of each mesh;
- decoders: mesh i -> mesh i-1 and mesh 0 -> grid.

Modules, in the order the command uses them:
- `config`: the settings (`configs/analysis/graph_statistics.toml`);
- `graphs`: the node sets and graphs of the model;
- `structure`: degrees and node counts of every graph, connected components and shortest paths of the processors
  (networkx: hops; scipy: km along the mesh, against geodesic distances), coverage of each pair of levels;
- `geometry`: geodesic edge lengths and the quality of the mesh triangles;
- `squashed`: the squashed grid-to-grid graph (grid node i -> j when the input at i reaches the output at j through
  the encoders, the message passing steps and the decoders), from sampled grid nodes: degrees and horizon;
- `quantities`: how values are summarized and described in the JSON output;
- `report`: the summary tables printed to stdout;
- `cli`: `compute` (all statistics) and the command.

networkx runs only the processors' path metrics. Its backends (e.g. cugraph, from the `graph-gpu` extra) are chosen
with networkx's environment variable `NETWORKX_BACKEND_PRIORITY`; algorithms a backend lacks run in networkx.
"""
