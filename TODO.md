# TODO list

## Core
* Train the f*cking model. **EXTREME PRIORITY**
* Determine how many training steps would make sense based on GraphCast training schedule and dataset. **HIGH PRIORITY**
* Use `optax.schedule.warmup_cosine_decay_schedule` instead of chaining schedules. **HIGH PRIORITY**
* Implement the validation code. **HIGH PRIORITY**

## Checks and improvements
* Check if checkpointing rng data, and not rng JAX array, remove some of the warnings from Orbax. **HIGH PRIORITY**
* Check if and why mesh and grid nodes must have the same number of features before applying the encoder graph (as it is currently implemented). **HIGH PRIORITY**
* Investigate bug resulting in the release of SharedMemoryArrays when using multiple workers in Grain and checkpointing parameters as pytree in Orbax. **HIGH PRIORITY**
* Investigate training using large batch sizes, determine new learning rate. **HIGH PRIORITY**
* Investigate why the compilation time increase with the number of JAX processes. **HIGH PRIORITY**
* Check that tensorboard is logging asynchornously. **HIGH PRIORITY**
* Check that artifacts contain only zero std residuals for sea ice variables. **HIGH PRIORITY**
* Profile training loop memory usage using xprof/Tensorboard plugin. **HIGH PRIORITY**
* Project icosahedral meshes produced by functions in `icosahedral_mesh.py` to WGS84 using GIS utilities, and produce either a `.msh` file or a `mesh_graph.MeshData`, such that the global ocean model can be init with either a triangular mesh or a geodesic grid. **HIGH PRIORITY**
* Add routines to check graph statistics (number of nodes, edges, degree, etc.). In particular, one should check that all grid nodes are connected to the processor. **HIGH PRIORITY** 
* Check that accumulated variables in ARCO-OCEAN are consistent (e.g. they use the (t-24h, t) convention). **HIGH PRIORITY**
* Debug currently broken multi-processing dataloader. **HIGH PRIORITY**
* Fix probably-broken DASK scripts (e.g., stats) with new `get_distributed_logger`/`set_up_root_distributed_logger`. (Medium priority)
* Add logging settings to training toml config file. (Medium priority)
* Implement sea-ice loss term accounting for zero-inflated variables (change the loss, inject noise, train a deeper network, or for longer). (Medium priority)
* Investigate why JAX initialization gets logged twice, on stdout and stderr, in training.py. (Low priority)
* Experiment with typed graphs context (i.e., a global graph attribute which might embed variables as year_progress).
* Make the model serializable and save along other data (e.g., mesh info). (Medium priority)
* Solve protobuf gencode warnings. (Medium priority)
* Implement a script that produces a shapefile with an equilateral triangle with some edge length at the south pole, then use the `geospatial_mesh.py` functions to produce a mesh pretending that was the coastline. Then vary the edge length and check the mesh quality metrics, one should see some peaks corresponding to geodetic polyhedra. (Low priority)
* Streamline dataset downloading and post-processing, in particular the download-and-postprocess then merge-and-postprocess workflow should be implemented as a single operation. (Low priority)
* Port download dataset script to zarr3. (Low priority)
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
* Choose a test suite (e.g. Pytest), update original codebase tests (using abseil/unittest), and add tests for the new code. Implement CI using GitHub actions. (Low priority)
* Add animation and plot routines to `plot_utils.py`. (Low priority)
* Document `--add-spack-mirrors` in `leonardo/environment/setup.sh`, check that other users are able to reproduce environment creation. (Low priority)
* Merge into main branch. (Low priority)
* Most of the documentation is missing or outdated, fix it. (Low priority)
* Notebooks have not been updated to the current version of library code and might be broken. (Low priority)

## Long-term goals
* Experiment with typed graphs, especially for coastline boundary conditions (i.e., use a typed graph connecting GLOFAS only to processor graph boundary nodes). (Low priority)
* Experiment with ideas contained in ICLR25 proposal, add routines to fit on meshes and project back to Cartesian grids, perform experiments on reconstruction errors. (Low priority)
* Create a similar model for Mediterranean sea biogeochemistry, it should require minimal changes to download scripts and model code. However, more thinking is needed on how to deal with open boundaries. (Low priority)
* Implement pipeline-parallelism for the autoregressive predictor. (Medium priority)
* Implement the algorithm sketched in `graph_pruning.py`, then implement a derived predictor class that does domain decomposition and measure memory consumption of the graph pruned version. (Low priority)

## Completed tasks
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
