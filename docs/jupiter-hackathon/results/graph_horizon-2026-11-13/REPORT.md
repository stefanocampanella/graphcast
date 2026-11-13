# Distances on the meshes and horizon of the squashed grid-to-grid graph (13 Nov 2026)

`moby-dick graph-statistics` with `configs/analysis/graph_statistics.toml` (32 sampled sources per processor component,
1,024 sampled grid nodes for the squashed graphs, 8 and 16 message passing steps) on the ten uniform meshes and the
two new ones (200 and 150 km), the icosahedral M6 multi-mesh and the 50 km + 120 km hierarchy, all seamsh v0.6.0+fork
meshes of `data/geometry`, on the 693,039 ocean points of the training mask; job 2880077 (one booster node, the 14
cases in parallel, 7 min; 25 km: 6.7 min, of which 2 min processor statistics and 3.8 min squashed graphs).
`distances.csv` and `squashed.csv` hold the values below.

## Hops × edge length is the distance along the mesh, not between the points

The earlier statement "mean shortest path × median edge length (11,400–11,900 km) is the typical distance between two
points of the ocean" is not right. From the same sampled sources to every other node of the largest component:

| Mesh | Nodes | Mean hops | Median edge (km) | Hops × edge (km) | Along the mesh (km) | Geodesic (km) | Detour mean (median) |
|---|---|---|---|---|---|---|---|
| 200 km | 13,743 | 67.4 | 191.7 | 12,912 | 11,999 | 9,988 | 1.22 (1.085) |
| 150 km | 22,960 | 83.3 | 145.9 | 12,155 | 11,877 | 10,008 | 1.20 (1.080) |
| 120 km | 34,527 | 98.9 | 117.4 | 11,611 | 11,404 | 9,881 | 1.16 (1.068) |
| 100 km | 48,520 | 124.2 | 98.4 | 12,213 | 11,873 | 10,030 | 1.21 (1.077) |
| 85 km | 65,775 | 143.3 | 83.9 | 12,026 | 11,812 | 9,894 | 1.22 (1.077) |
| 70 km | 95,079 | 168.4 | 69.4 | 11,678 | 11,537 | 9,999 | 1.16 (1.072) |
| 60 km | 127,768 | 194.9 | 59.6 | 11,611 | 11,495 | 9,965 | 1.17 (1.074) |
| 50 km | 182,023 | 228.9 | 49.7 | 11,372 | 11,273 | 9,962 | 1.14 (1.065) |
| 40 km | 280,528 | 284.4 | 39.8 | 11,324 | 11,245 | 9,899 | 1.14 (1.064) |
| 35 km | 364,295 | 334.4 | 34.9 | 11,659 | 11,566 | 10,018 | 1.16 (1.071) |
| 30 km | 492,254 | 377.6 | 29.9 | 11,293 | 11,229 | 9,893 | 1.14 (1.064) |
| 25 km | 704,232 | 471.0 | 24.9 | 11,747 | 11,652 | 9,936 | 1.19 (1.070) |
| icosahedral M6 | 40,962 | 7.8 | 126.2 | 985 | 10,638 | 10,007 | 1.07 (1.067) |

- **Geodesic:** the mean geodesic distance between two ocean nodes is 9,880–10,030 km on every uniform mesh, i.e. the
  mean distance between two random points of a sphere (πR/2 = 10,008 km for R = 6,371 km): the ocean's shape does not
  shorten it.
- **Along the mesh** (Dijkstra with geodesic edge lengths): 11,230–12,000 km, 14–22 % longer (mean detour 1.14–1.22,
  median 1.06–1.09). The median is the zig-zag of paths along triangle edges; the mean adds the pairs that must go
  around continents (75th percentile 1.12–1.20; largest 10–31, e.g. pairs on either side of an isthmus).
- **Hops × median edge** (11,290–12,910 km) follows the distance along the mesh (within 1–8 %), not the geodesic one.
- The variation between meshes (e.g. 120 km 11,611, 100 km 12,213) is mostly sampling noise of the 32 sources: a run
  of the 120 km mesh with 8 sources gave 11,990 km.
- On the icosahedral multi-mesh the hops are not comparable (edges of all levels, up to 7,076 km); its paths along the
  mesh cross land, so the detour is 1.07.

## The squashed grid-to-grid graph

Grid point i → j when the input at i can reach the output at j through the encoder, L message passing steps and the
decoder (`mobydick.graph_statistics.squashed`; every node update keeps the node's own state, so for one mesh the
adjacency is I + D (I + P)^L E). It is never formed: indicator vectors of 64 sampled grid nodes at a time are propagated
forward (out-degree, horizon) and backward with the transposed operations (in-degree). Horizon: the geodesic length of
the squashed graph's edges (self-loops left out), at most 4,096 per sender, weighted by out-degree.

| Mesh | Steps | Out-degree mean (median) | Share of the ocean | Horizon mean (km) | p50 | p75 | max | Steps × edge (km) |
|---|---|---|---|---|---|---|---|---|
| 200 km | 8 | 18,986 (12,974) | 2.7 % | 1,003 | 1,040 | 1,343 | 2,054 | 1,534 |
| 200 km | 16 | 49,841 (44,827) | 7.2 % | 1,837 | 1,890 | 2,492 | 3,621 | 3,067 |
| 120 km | 8 | 8,637 (4,778) | 1.2 % | 623 | 649 | 830 | 1,208 | 939 |
| 120 km | 16 | 23,778 (16,714) | 3.4 % | 1,146 | 1,192 | 1,542 | 2,173 | 1,878 |
| 85 km | 16 | 13,886 (8,534) | 2.0 % | 824 | 861 | 1,104 | 1,535 | 1,342 |
| 60 km | 16 | 7,936 (4,278) | 1.1 % | 591 | 618 | 787 | 1,107 | 954 |
| 50 km | 8 | 1,956 (875) | 0.28 % | 266 | 279 | 351 | 500 | 398 |
| 50 km | 16 | 5,863 (3,025) | 0.85 % | 495 | 519 | 658 | 907 | 795 |
| 25 km | 16 | 1,764 (778) | 0.25 % | 251 | 264 | 333 | 480 | 398 |
| icosahedral M6 | 8 | 539,636 (560,312) | 78 % | 8,773 | 8,648 | 11,828 | 20,004 | – |
| icosahedral M6 | 16 | 693,038 (all) | 100 % | 9,933 | 9,974 | 13,514 | 20,004 | – |
| 50 km + 120 km | 8 | 15,265 (9,572) | 2.2 % | 873 | 911 | 1,170 | 1,657 | – |
| 50 km + 120 km | 16 | 40,751 (33,655) | 5.9 % | 1,604 | 1,664 | 2,171 | 2,999 | – |

(all meshes and both step counts in `squashed.csv`)

- **Uniform meshes:** the mean horizon is 0.60–0.67 × steps × median edge, the largest 1.13–1.51 × (the encoder
  radius and the decoder's face add to the steps at both ends). The squashed graph is sparse: 50 km × 16, the best configuration, links
  a grid point to 0.85 % of the ocean, within 907 km. In-degrees equal out-degrees on average (the processors are
  symmetric; encoder and decoder differ only in their reach).
- **Median out-degree well below the mean:** a skewed distribution; the 0.25° grid has more points per area at high
  latitudes, and land takes part of a coastal point's neighbourhood (not separated here).
- **Senders without edges** (0.1–0.2 % at 35 and 25 km): ocean points outside the mesh domain, without encoder edges.
- **Icosahedral M6:** with 16 steps every grid point reaches every other (diameter 14 hops); with 8, 78 % of them.
- **Hierarchy (50 km + 120 km, 16 steps on each, a V-cycle):** the horizon triples against 50 km alone (1,604 vs 495
  km), close to 120 km × 16 (1,146 km) plus the 50 km steps.
- **Horizon and skill:** at 16 steps, uniform meshes from 200 to 50 km lose horizon (1,837 → 495 km) while their
  validation loss improves by 14 % (Lambrechts study), and the icosahedral M6 × 16 (global horizon) is 10.7 % worse
  than 50 km × 16 except on `zos` (−16 %): resolution, not reach, limits the skill today, except for sea surface height.
