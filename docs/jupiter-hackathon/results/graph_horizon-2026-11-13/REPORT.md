# Distances on the meshes and horizon of the squashed grid-to-grid graph (13 Nov 2026)

`moby-dick graph-statistics` with the settings of `configs/analysis/graph_statistics.toml` (32 sampled sources per
processor component, 1,024 sampled grid nodes for the squashed graph), one config per mesh and number of message
passing steps (8 and 16 on every mesh): the twelve uniform meshes from 200 to 25 km, the icosahedral M6 multi-mesh and
the 50 km + 120 km hierarchy, all seamsh v0.6.0+fork meshes of `data/geometry`, on the 693,039 ocean points of the
training mask. Job 2881277: the 28 cases in parallel on one booster node, 5.2 min (25 km × 16 the longest). A first run
(job 2880077, float32 propagation with thresholding instead of booleans) gave the same degrees exactly and horizons
within 0.09 % (sampled edges differ). `distances.csv` and `squashed.csv` hold the values below.

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
decoder (`mobydick.graph_statistics.squashed`). Every node update keeps the node's own state, so in boolean algebra
(OR of ANDs) one step is x ← x OR P x and L steps give (I + P)^L = I + P + … + P^L: the nodes within at most L hops.
For one mesh the adjacency is I + D (I + P)^L E; for a hierarchy, a V-cycle: L steps on each mesh on the way down,
then each decoder ORs the coarser mesh's result into the finer mesh's state of the way down (the U-Net skip
connection, so paths that stay on the finer mesh count). The adjacency is never formed: boolean indicator vectors of
64 sampled grid nodes at a time are propagated through the boolean sparse matrices, forward (the grid points each
sender's input reaches: out-degree, domain of influence, horizon) and backward with the transposed operations in
reverse order (the grid points whose input reaches each receiver: in-degree, receptive field). Horizon: the geodesic
length of the squashed graph's edges (self-loops left out), at most 4,096 per sender, weighted by out-degree.

| Mesh | Steps | Out-degree mean (median) | Share of the ocean | Horizon mean (km) | p50 | p75 | max | Steps × edge (km) | Receptive field, median receiver (km) |
|---|---|---|---|---|---|---|---|---|---|
| 200 km | 8 | 18,986 (12,974) | 2.7 % | 1,003 | 1,040 | 1,344 | 2,044 | 1,533 | 1,087 |
| 200 km | 16 | 49,841 (44,827) | 7.2 % | 1,837 | 1,891 | 2,493 | 3,625 | 3,067 | 1,997 |
| 120 km | 8 | 8,637 (4,778) | 1.2 % | 623 | 649 | 830 | 1,208 | 940 | 659 |
| 120 km | 16 | 23,778 (16,714) | 3.4 % | 1,146 | 1,192 | 1,542 | 2,180 | 1,879 | 1,227 |
| 85 km | 16 | 13,886 (8,534) | 2.0 % | 824 | 861 | 1,104 | 1,544 | 1,343 | 879 |
| 60 km | 16 | 7,936 (4,278) | 1.1 % | 590 | 617 | 786 | 1,116 | 953 | 623 |
| 50 km | 8 | 1,956 (875) | 0.28 % | 266 | 279 | 351 | 500 | 397 | 277 |
| 50 km | 16 | 5,863 (3,025) | 0.85 % | 495 | 520 | 658 | 907 | 795 | 520 |
| 25 km | 16 | 1,764 (778) | 0.25 % | 251 | 264 | 332 | 480 | 399 | 261 |
| icosahedral M6 | 8 | 539,636 (560,312) | 77.9 % | 8,770 | 8,645 | 11,826 | 20,004 | – | 8,713 |
| icosahedral M6 | 16 | 693,038 (all) | 100 % | 9,935 | 9,977 | 13,512 | 20,004 | – | 9,910 |
| 50 km + 120 km | 8 | 15,265 (9,572) | 2.2 % | 873 | 911 | 1,170 | 1,662 | – | 928 |
| 50 km + 120 km | 16 | 40,751 (33,655) | 5.9 % | 1,604 | 1,663 | 2,170 | 2,995 | – | 1,724 |

(all meshes and both step counts in `squashed.csv`)

- **Uniform meshes:** the mean horizon is 0.60–0.67 × steps × median edge, the largest 1.13–1.51 × (the encoder
  radius and the decoder's face add to the steps at both ends). The squashed graph is sparse: 50 km × 16, the best configuration, links
  a grid point to 0.85 % of the ocean, within 907 km. Mean in- and out-degrees are equal (both are edges /
  nodes), and the receptive field of a receiver is as wide as the domain of influence of a sender (median mean
  distance within 1 %).
- **Median out-degree well below the mean:** a skewed distribution; the 0.25° grid has more points per area at high
  latitudes, and land takes part of a coastal point's neighbourhood (not separated here).
- **Senders without edges** (0.1–0.2 % at 35 and 25 km): ocean points outside the mesh domain, without encoder edges.
- **Icosahedral M6:** with 16 steps every grid point reaches every other (diameter 14 hops); with 8, 78 % of them.
- **Hierarchy (50 km + 120 km, 16 steps on each, a V-cycle):** the horizon triples against 50 km alone (1,604 vs 495
  km), close to 120 km × 16 (1,146 km) plus the 50 km steps.
- **Horizon and skill:** at 16 steps, uniform meshes from 200 to 50 km lose horizon (1,837 → 495 km) while their
  validation loss improves by 14 % (Lambrechts study), and the icosahedral M6 × 16 (global horizon) is 10.7 % worse
  than 50 km × 16 except on `zos` (−16 %): resolution, not reach, limits the skill today, except for sea surface height.

## Connectivity of semi-enclosed basins and inner seas

`graph-statistics` measures, for each pair of a basin and the nearby ocean it opens to (`[[basin_connections]]` and
`[[regions]]` of `configs/analysis/graph_statistics.toml`), the links of the squashed graph between the two regions.
The values are exact: a column marking every node of a region propagates the union of their reaches, forward and
backward. The fewest message passing steps (the same on every mesh) that link the regions come from a bisection over the
steps, up to 128. Grid nodes in both regions (polygons meeting at a strait) count for the basin. Job 2884014 (every
mesh, 8 and 16 steps, 6.5 min; `basins.csv`).

Regions: the first 8 pairs were drawn by hand; 22 pairs were added for the Sea of Marmara, the Seto Inland Sea, Lake
Maracaibo, the Sea of Japan, Baffin Bay, the White Sea, the Gulf of Corinth, the Indonesian throughflow seas (Java,
Bali, Flores, Banda), the Canadian Arctic Archipelago (Gulf of Boothia and Prince Regent Inlet, Foxe Basin) and the
Philippine inland seas (Sulu, Bohol), one pair per opening. Their polygons come from the IHO Sea Areas of Marine Regions
(Flanders Marine Institute (2018), IHO Sea Areas, version 3, https://www.marineregions.org/, https://doi.org/10.14284/323, CC BY-NC-SA 4.0; all regions are in `configs/analysis/ocean_regions.toml`, under the same license: grown by 0.3°
for the coastal grid points, the neighbouring IHO seas removed, simplified), or are drawn
after the documented position of their straits where IHO defines no area (Gulf of Corinth, Lake Maracaibo, Bohol Sea,
Foxe Basin). Maps of all pairs on the training mask: https://claude.ai/artifact/CTpqbrKCmtQkMzpuiKjgnq. The 0.25° mask
resolves the smallest basins poorly: Lake Maracaibo has 2 grid points, the Gulf of Corinth 8, the Sea of Marmara 35,
and narrow islands (Cebu, Bohol, most of Leyte) are ocean cells.

The production configuration, 50 km × 16 (shares of each region's nodes linked to the other region):

| Basin → ocean | Nodes | Basin nodes reaching the ocean | Ocean nodes reached | Ocean nodes reaching the basin | Basin nodes reached |
|---|---|---|---|---|---|
| Mediterranean Sea → Gulf of Cadiz and off Portugal | 4,497 / 807 | 6.5 % | 82.9 % | 80.2 % | 6.2 % |
| Black Sea → Aegean Sea | 821 / 225 | 0.0 % | 0.0 % | 0.0 % | 0.0 % |
| Sea of Azov → Black Sea | 85 / 821 | 100.0 % | 19.1 % | 23.5 % | 100.0 % |
| Red Sea → Gulf of Aden | 741 / 320 | 27.7 % | 85.3 % | 90.0 % | 28.7 % |
| Persian Gulf → Gulf of Oman | 401 / 206 | 63.1 % | 100.0 % | 99.0 % | 66.1 % |
| Baltic Sea → North Sea | 1,121 / 637 | 0.1 % | 59.0 % | 55.1 % | 0.1 % |
| Gulf of California → Eastern tropical Pacific off the Gulf of California | 323 / 500 | 69.7 % | 100.0 % | 100.0 % | 68.1 % |
| Hudson Bay → Labrador Sea | 2,207 / 896 | 0.0 % | 0.0 % | 0.0 % | 0.0 % |
| Sea of Marmara → Aegean Sea | 35 / 225 | 60.0 % | 99.1 % | 95.6 % | 62.9 % |
| Sea of Marmara → Black Sea | 35 / 821 | 82.9 % | 56.8 % | 59.3 % | 85.7 % |
| Seto Inland Sea → Philippine Sea south of Shikoku | 57 / 120 | 71.9 % | 100.0 % | 100.0 % | 98.2 % |
| Seto Inland Sea → Sea of Japan | 57 / 1,960 | 52.6 % | 50.6 % | 52.8 % | 68.4 % |
| Lake Maracaibo → Gulf of Venezuela | 2 / 25 | 0.0 % | 0.0 % | 100.0 % | 100.0 % |
| Sea of Japan → East China Sea off the Korea Strait | 1,961 / 168 | 24.5 % | 100.0 % | 100.0 % | 25.9 % |
| Sea of Japan → North Pacific east of the Tsugaru Strait | 1,961 / 169 | 47.8 % | 100.0 % | 100.0 % | 49.6 % |
| Sea of Japan → Sea of Okhotsk off the La Perouse Strait | 1,961 / 142 | 43.0 % | 100.0 % | 100.0 % | 44.1 % |
| Baffin Bay → Labrador Sea | 2,851 / 896 | 0.0 % | 0.0 % | 0.0 % | 0.0 % |
| Baffin Bay → Lincoln Sea | 2,851 / 447 | 23.4 % | 100.0 % | 98.0 % | 24.5 % |
| White Sea → Barents Sea off the White Sea | 348 / 420 | 99.4 % | 100.0 % | 100.0 % | 100.0 % |
| Gulf of Corinth → Ionian Sea off the Gulf of Patras | 8 / 25 | 0.0 % | 0.0 % | 100.0 % | 100.0 % |
| Indonesian seas (Java, Bali, Flores, Banda) → Indian Ocean south of the Lesser Sunda Islands | 1,964 / 1,060 | 57.0 % | 100.0 % | 100.0 % | 57.3 % |
| Indonesian seas (Java, Bali, Flores, Banda) → South China Sea off the Karimata Strait | 1,964 / 425 | 33.7 % | 100.0 % | 99.3 % | 34.2 % |
| Indonesian seas (Java, Bali, Flores, Banda) → Pacific east of Mindanao and Halmahera | 1,964 / 490 | 8.7 % | 27.8 % | 29.8 % | 8.8 % |
| Gulf of Boothia and Prince Regent Inlet → Baffin Bay | 469 / 2,851 | 74.2 % | 60.0 % | 56.7 % | 76.5 % |
| Gulf of Boothia and Prince Regent Inlet → Foxe Basin | 469 / 707 | 0.0 % | 0.0 % | 0.0 % | 0.0 % |
| Foxe Basin → Hudson Strait | 707 / 603 | 98.4 % | 66.5 % | 64.8 % | 100.0 % |
| Sulu Sea → South China Sea west of Palawan and Mindoro | 478 / 342 | 100.0 % | 100.0 % | 100.0 % | 100.0 % |
| Sulu Sea → Celebes Sea | 478 / 627 | 99.2 % | 100.0 % | 99.8 % | 98.5 % |
| Bohol Sea → Sulu Sea | 62 / 478 | 100.0 % | 100.0 % | 100.0 % | 100.0 % |
| Bohol Sea → Philippine Sea east of the Surigao Strait | 62 / 103 | 100.0 % | 100.0 % | 100.0 % | 100.0 % |

Fewest message passing steps that link the basin and the ocean (basin → ocean / ocean → basin when they differ; –:
not linked within 128 steps). The same number runs on every mesh: in the 50+120 km column, n means n steps on each of
the hierarchy's two meshes, 2n in all:

| Basin → ocean | 200 km | 120 km | 85 km | 50 km | 40 km | 25 km | ico M6 | 50+120 km |
|---|---|---|---|---|---|---|---|---|
| Mediterranean Sea → Gulf of Cadiz and off Portugal | 0 | 0 | 0 | 1 | 1 / 2 | 2 | 0 | 0 |
| Black Sea → Aegean Sea | 29 / 28 | 30 / 29 | 30 / 29 | 29 | 29 / 28 | 30 | 1 | 14 / 15 |
| Sea of Azov → Black Sea | 0 / 1 | 0 / 1 | 0 | 0 | 0 | 0 | 0 | 0 |
| Red Sea → Gulf of Aden | 0 / 1 | 1 | 1 / 2 | 3 / 2 | 3 | 6 | 1 / 0 | 1 / 0 |
| Persian Gulf → Gulf of Oman | 0 | 0 / 1 | 0 / 3 | 4 | 3 | 6 | 0 / 1 | 0 / 1 |
| Baltic Sea → North Sea | 2 | 4 / 3 | 5 | 9 | 12 / 11 | 18 / 17 | 1 / 2 | 3 / 2 |
| Gulf of California → Eastern tropical Pacific off the Gulf of California | 0 | 1 / 0 | 1 | 2 | 3 / 2 | 5 | 0 | 0 |
| Hudson Bay → Labrador Sea | 6 | 9 / 8 | 13 | 21 | 26 | 41 / 40 | 2 | 6 |
| Sea of Marmara → Aegean Sea | 2 / 1 | 1 | 1 | 1 | 2 / 1 | 2 / 1 | 0 | 0 / 1 |
| Sea of Marmara → Black Sea | 1 | 1 | 1 | 1 | 1 | 1 / 2 | 0 | 1 / 0 |
| Seto Inland Sea → Philippine Sea south of Shikoku | 0 | 0 | 1 / 0 | 1 | 3 / 1 | 4 / 3 | 0 | 0 |
| Seto Inland Sea → Sea of Japan | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| Lake Maracaibo → Gulf of Venezuela | 0 | – / 0 | 0 | – / 0 | – / 0 | – / 1 | 0 | – / 0 |
| Sea of Japan → East China Sea off the Korea Strait | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| Sea of Japan → North Pacific east of the Tsugaru Strait | 0 | 0 | 0 | 0 | 0 / 1 | 1 | 0 | 0 |
| Sea of Japan → Sea of Okhotsk off the La Perouse Strait | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| Baffin Bay → Labrador Sea | 5 / 6 | 8 | 12 | 20 | 25 / 26 | 42 | 2 / 3 | 6 |
| Baffin Bay → Lincoln Sea | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| White Sea → Barents Sea off the White Sea | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| Gulf of Corinth → Ionian Sea off the Gulf of Patras | 0 | – / 0 | 1 / 0 | – / 0 | – / 1 | – / 2 | 0 | – / 0 |
| Indonesian seas (Java, Bali, Flores, Banda) → Indian Ocean south of the Lesser Sunda Islands | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| Indonesian seas (Java, Bali, Flores, Banda) → South China Sea off the Karimata Strait | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| Indonesian seas (Java, Bali, Flores, Banda) → Pacific east of Mindanao and Halmahera | 2 / 3 | 5 | 6 / 7 | 12 | 14 / 15 | 23 | 2 | 4 / 3 |
| Gulf of Boothia and Prince Regent Inlet → Baffin Bay | 1 | 2 | 3 | 4 | 6 / 7 | 9 / 10 | 1 | 1 |
| Gulf of Boothia and Prince Regent Inlet → Foxe Basin | 0 | 28 / 0 | 39 / 40 | 62 | 76 | 118 | 0 | 18 / 0 |
| Foxe Basin → Hudson Strait | 0 | 0 | 1 | 1 | 2 / 3 | 4 | 0 | 0 |
| Sulu Sea → South China Sea west of Palawan and Mindoro | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| Sulu Sea → Celebes Sea | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| Bohol Sea → Sulu Sea | 0 | 0 | 0 | 0 | 0 | 1 / 0 | 0 | 0 |
| Bohol Sea → Philippine Sea east of the Surigao Strait | 0 | 0 | 0 | 1 | 1 | 2 | 0 | 0 |

- **Regions that meet at a wide opening link with no step** (0): a grid node on each side shares a mesh vertex (the
  Sea of Japan with its three neighbours, the White Sea, Sulu, the Indonesian seas with the Indian Ocean and the South
  China Sea, Baffin Bay with the Lincoln Sea). For them the shares at 16 steps say how far into each region the links
  go (e.g. 43 % of the Sea of Japan reaches the Sea of Okhotsk region, 23 % of Baffin Bay the Lincoln Sea).
- **The Black Sea is cut off in every uniform-mesh model.** Linking it to the Aegean takes 28–30 steps on every mesh
  from 200 to 25 km: the hops go through the elements of the Turkish straits and the Sea of Marmara, refined to 3 km
  (growing by 0.3 km per km), so their number does not depend on the base mesh size. The Sea of Marmara itself links to
  the Aegean and the Black Sea in 1–2 steps: the hops are spent inside it. The Sea of Azov is linked to the Black Sea
  directly (0–1 steps). The icosahedral multi-mesh needs 1 step; the 50 km + 120 km hierarchy 14–15 on each mesh
  (28–30 in all: both meshes contain the refined straits).
- **Basins outside the mesh domain:** Lake Maracaibo and the Gulf of Corinth are behind straits narrower than the 10 km
  coastline coarsening (Tablazo, Rion), so their grid points have no encoder edges (–: their input reaches nothing) and
  are decoded from the vertices of the neighbouring gulf (ocean → basin in 0–2 steps), as the Black Sea was before
  12 Nov 2026.
- **Distance-limited basins:** the fewest steps grow as the mesh gets finer where the opening is far, in hops, from the
  ocean region: Hudson Bay (21 at 50 km), Baffin Bay to the Labrador Sea (20, through Davis Strait), the Baltic (9),
  the Indonesian seas to the Pacific east of Mindanao and Halmahera (12, through the Molucca and Halmahera Seas), the
  Gulf of Boothia to Baffin Bay (4, through Lancaster Sound). The Gulf of Boothia and Foxe Basin need 62 steps at 50 km
  (118 at 25 km): the Fury and Hecla Strait (about 2 km wide) is closed in the meshes, so the path goes round through
  Lancaster Sound, Baffin Bay, Davis Strait and Hudson Strait. At 120 km and in the hierarchy the two directions
  differ (0 steps from Foxe Basin, 28 and 18 from the Gulf of Boothia), probably because Boothia grid points by the
  strait are decoded from Foxe Basin vertices without being encoded themselves (not checked).
- **Open basins:** the Mediterranean, the Red Sea, the Persian Gulf, the Gulf of California, the Seto Inland Sea (to
  the Pacific through the Kii and Bungo Channels), Foxe Basin (to Hudson Strait) and the Bohol Sea connect within 0–6
  steps on every mesh.
- **Fixes to consider:** a coarser refinement of the Turkish straits (sizes of 10–20 km instead of 3 km, if they still
  keep the straits open), long-range edges across the straits, or a hierarchy with more steps on the coarse mesh;
  refinements like the Black Sea's for the Tablazo and Rion Straits only if those basins matter.
