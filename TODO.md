# TODO list

## Core
* Fix notebooks (e.g., mesh_comparison.ipynb goes OOM). **HIGH PRIORITY**
* Implement optimization using autoregressive steps/noise injection. **HIGH PRIORITY**
* Retrain with higher resolution mesh. **HIGH PRIORITY**
* Use a different GNN to encode processor mesh boundaries, exclude boundaries from processor message-passing steps. **HIGH PRIORITY**.
* Rename `msl` to `mslp` in ARCO-OCEAN v2. **HIGH PRIORITY**

## Checks and improvements
* Run the Grain profiler (execution summary per stage) for the store layouts (zarr v2, v3, their SquashFS images) and
  add it to the benchmark reports. The loader no longer limits training, so this is for comparison, not tuning.
  (Low priority)
* Simplify SLURM scripts by forwarding arguments to Python CLI, without building flags using getopt. (Medium priority)
* Check if checkpointing rng data, and not rng JAX array, remove some of the warnings from Orbax. **HIGH PRIORITY**
* Check if and why mesh and grid nodes must have the same number of features before applying the encoder graph (as it is currently implemented). **HIGH PRIORITY**
* Investigate bug resulting in the release of SharedMemoryArrays when using multiple workers in Grain and checkpointing parameters as pytree in Orbax. **HIGH PRIORITY**
* Investigate training using large batch sizes, determine new learning rate. **HIGH PRIORITY**
* Investigate why the compilation time increase with the number of JAX processes. **HIGH PRIORITY**
* Break down GPU memory by component (encoder, processor, decoder), e.g. with the xprof memory profile, and record
  reference numbers. Indirect numbers from the memfit sweep (XLA plan against message passing steps, 50 levels,
  `results/2026-10-29-memfit-alllevels`): up to 70 km the plan is flat at 50-57 GiB whatever the steps; edge chunks
  cut the step-independent part from ~54 to ~35 GiB on the 120 km mesh (grid-mesh edge updates of encoder and
  decoder); the processor costs ~1.2 GiB per step at 50 km, ~2 GiB at 35-40 km (with chunks), 5.6-5.8 GiB on the
  fine mesh. **HIGH PRIORITY**
* Project icosahedral meshes produced by functions in `icosahedral_mesh.py` to WGS84 using GIS utilities, and produce either a `.msh` file or a `mesh_graph.MeshData`, such that the global ocean model can be init with either a triangular mesh or a geodesic grid. **HIGH PRIORITY**
* Add routines to check graph statistics (number of nodes, edges, degree, etc.). In particular, one should check that all grid nodes are connected to the processor. **HIGH PRIORITY** 
* Check that accumulated variables in ARCO-OCEAN are consistent (e.g. they use the (t-24h, t) convention). **HIGH PRIORITY**
* Simplify `xarray_jax.py` with the Python array API standard: JAX arrays implement it, so xarray can hold them
  directly, without `JaxArrayWrapper`, `wrap`/`unwrap` or the wrapping in `apply_ufunc` (the custom constructors would
  remain only for dynamic coordinates). Xarray-JAX, a standalone package derived from GraphCast's `xarray_jax`, did
  this: https://github.com/mishasinitcyn/Xarray-JAX/pull/4 (closes issue #1 there). Inside `jax.jit`, numpy functions
  are then no longer redirected to `jax.numpy` (e.g. `np.abs` on a traced array fails): use operators, built-ins or
  `xarray_jax.apply_ufunc` with the `jax.numpy` function, and check every caller.
* Add logging settings to training toml config file. (Medium priority)
* Implement sea-ice loss term accounting for zero-inflated variables (change the loss, inject noise, train a deeper network, or for longer). (Medium priority)
* Investigate why JAX initialization gets logged twice, on stdout and stderr, in training.py. (Low priority)
* Experiment with typed graphs context (i.e., a global graph attribute which might embed variables as year_progress).
* Make the model serializable and save along other data (e.g., mesh info). (Medium priority)
* Solve protobuf gencode warnings. (Medium priority)
* Implement a script that produces a shapefile with an equilateral triangle with some edge length at the south pole, then use the `geospatial_mesh.py` functions to produce a mesh pretending that was the coastline. Then vary the edge length and check the mesh quality metrics, one should see some peaks corresponding to geodetic polyhedra. (Low priority)
* Refactor code base to have a `models` module, move the `model.py` and all global ocean related code to a model-specific module. Update `pyproject.toml` accordingly. (Low priority)
* Revise older code to use Python logging properly and check that expensive format strings are not interpolated. (Low priority)

## Tests, documentation and data
* Update README with current status. **HIGH PRIORITY**
* Add utilities and a notebook to inspect the encoder and decoder graphs. **HIGH PRIORITY**
* Add routines to compute mesh quality metrics (e.g., mean edge length, mean node degree, etc.). In particular, the volume-length metric should be implemented, see: https://scicomp.stackexchange.com/a/27095 **HIGH PRIORITY**
* Set up documentation with Sphinx and add documentation generation to GitHub CI. (Low priority)
* Add a proof that geodetic polyhedra are optimal (i.e., they maximize the volume-length metric). (Low priority)
* Add a proof of the algorithm sketched in `graph_pruning.py`. (Low priority)
* Finish the notebook with technical notes on data download and post-processing.
* Run all notebooks on Leonardo using papermill, validate results. (Low priority)
* Download a reduced-resolution version of the ARCO-OCEAN and update to AWS. (Medium priority)
* Integrate and revise notebooks to explain details of mesh generation. (Medium priority)
* Port the original codebase tests (abseil/unittest, probably broken) to pytest, adopted for new code in `tests/`. Add pytest to the image. Implement CI using GitHub actions. (Low priority)
* Add animation and plot routines to `plot_utils.py`. (Low priority)
* Merge into main branch. (Low priority)
* Most of the documentation is missing or outdated, fix it. (Low priority)
* Notebooks have not been updated to the current version of library code and might be broken. (Low priority)

## Done
* PGLE (Nov 2026): no gain. The train step is compiled ahead of time, which automatic PGLE skips; the "Found 0
  instructions" warnings came from one-operation modules. Called through jit, the step gets a full profile but is no
  faster on 1 or 4 nodes (120 km x 8), for ~25 s of startup. See `clusters/jupiter/.env`.
* Preemption: Orbax saves at a sync point after SIGTERM (sent 2 minutes before the time limit by `train.slurm`),
  Grain workers ignore SIGTERM, and the run exits with status 75; resume with `--continue` (Nov 2026).
* Checkpoints: parameters and optimizer state reach Orbax as JAX arrays (no numpy conversion); metrics are converted
  only when saved or logged, and TensorBoard writes on process 0 only (Nov 2026).
* Number of training steps: estimated from GraphCast's schedule and the dataset (about 100-200 epochs), see
  `configs/training/production_run.toml` and the 29 Oct report (30 Oct 2026).
* Shape mismatch on retrieval and recompilation: not reproduced with the current code and tests (30 Oct 2026).
* Dataloader performance measured (JUBE loader sweep at 50 levels, all store layouts and Grain settings): compute
  bound (29 Oct 2026).
* Validation loss on held-out windows during training (29 Oct 2026).
* Normalization with climatology and anomaly_std: standardized anomaly stores (`graphcast download`, PR #1).
* Apptainer image from the NVIDIA JAX container (before the ESiWACE JUPITER hackathon).
* Train script: CLI and documentation (`clusters/jupiter/slurm/train.slurm`, `scripts/train.sh`).
* Multiprocessing dataloader (Grain workers reopen the store).
* Zarr v3: the downloader writes zarr v2 and sharded zarr v3 stores.

## Long-term goals

* Experiment with gated activations.
* Implement smoothing (Gaussian blurring) within the dataloader, as an alternative to different dataset resolution, as
  they do in Samudra.
* Investigate the potential of computing advection terms for tracers explicitly, or other techniques that take tracer
  gradients into account.
* Investigate the potential of bilinear MLPs, or bilinear forms for feature augmentation.
* Implement the CPRS-based losses and code for ensemble generation (conditional layer normalizations, as in
  adaspeech/aifs-cprs, and sharding).
* Investigate the impact on performance of free-running vs reanalysis datasets, spatial and temporal resolution,
  hyperparameters, mesh-size, and input normalizations in a controlled experiment.
* Experiment with typed graphs, especially for coastline boundary conditions (i.e., use a typed graph connecting GLOFAS only to processor graph boundary nodes). (Low priority)
* Experiment with ideas contained in ICLR25 proposal, add routines to fit on meshes and project back to Cartesian grids, perform experiments on reconstruction errors. (Low priority)
* Create a similar model for Mediterranean sea biogeochemistry, it should require minimal changes to download scripts and model code. However, more thinking is needed on how to deal with open boundaries. (Low priority)
* Implement pipeline-parallelism for the autoregressive predictor. (Medium priority)
* Implement the algorithm sketched in `graph_pruning.py`, then implement a derived predictor class that does domain decomposition and measure memory consumption of the graph pruned version. (Low priority)
* Implement hierarchical graph model.
* Implement latent space rollout.

## Completed tasks

* ~~Read the single standardized anomaly dataset written by `graphcast download` in the dataloader, instead of dataset, climatology and anomaly_std; write the training mask with `graphcast download-mask`.~~ **DONE**
* ~~Replace learnable Fourier features with fixed $\sin$ and $\cos$ encodings using concatenation.~~ **DONE**
* ~~Streamline dataset downloading and post-processing, in particular the download-and-postprocess then
  merge-and-postprocess workflow should be implemented as a single operation.~~ **DONE, now in a separate package:
  arco-make**
* ~~Add `oceanbench` to dependencies~~ **DONE**
* ~~Fix dataloader to use multiple workers and reduce training bubble.~~ **DONE**
* ~~Some nodes might be faulty or have systematic issues. Add hostname to log file names.~~ **DONE**
* ~~Filter warnings and skip day_of_year_sin/cos in normalization warnings.~~ **DONE**
* ~~Revise functions in `model_utils.py` to use GIS utilities.~~ **DONE**
* ~~Compute edge length and azimuth using GIS utilities, and add the latter to edge features.~~ **DONE**
* ~~Investigate the use of fourier features for nodes and edges, and revise embedder as outlined in `model_util.py` comments. See also https://bmild.github.io/fourfeat/index.html.~~ **DONE**
* ~~Experiment with argument buffer donation to accelerate training.~~ **DONE**
* ~~Check memory consumption without scan.~~ **DONE**
* ~~Revise level weighting using depth.~~ **DONE**
* ~~Revise per variable weighting according to GraphCast.~~ **DONE**
* ~~Check that checkpointing dataloader state does not dump large datasets on disk.~~ **DONE**
* ~~Write a training loop using Optax.~~ **DONE**
* ~~Checkpoint training loop with Orbax~~ **DONE**
* ~~Log training metrics to TensorBoard.~~ **DONE**
* ~~Implement restart from checkpoint logic.~~ **DONE**
* ~~Checkpoint dataloader state, and implement restart logic~~. **DONE**
* ~~Log each process independently using `cli_utils.py`.~~ **DONE**
* ~~Merge `init` and `train` in training.py: start training from configs, instead of initial checkpoint. Optionally stop at the `init` phase.~~ **DONE**
* ~~Split dataset for training and validation.~~ **DONE**
* ~~Move mask and artifacts to GPU memory once.~~ **DONE**
* ~~Move training stuff to `training_utils.py`~~ **DONE**
* ~~Move from `Dataloader` to `Dataset` interface in Grain, try using performance autotune to avoid OOM.~~ **DONE**
* ~~Use `optax.schedule.warmup_cosine_decay_schedule` instead of chaining schedules.~~ **DONE**
* ~~Train the f*cking (PoC) model.~~ **DONE**
* ~~Fix 'start-fresh' bug (if start-fresh is false and there are no checkpoints don't try restoring one).~~ **DONE**
* ~~Revise implementation of the dataloader to allow a subset of dates (used in validation) but not slices (those should be implemented outside the dataloader by selecting the relevant part of the dataset).~~ **DONE**
