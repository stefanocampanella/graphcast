# TODO list

* Fix dataloader to use multiple workers. **HIGH PRIORITY**
* Write a training loop using Optax. **HIGH PRIORITY**
* Checkpoint training loop with Orbax **HIGH PRIORITY**
* Log training metrics to TensorBoard. **HIGH PRIORITY**
* Profile training loop memory usage using xprof/Tensorboard plugin. **HIGH PRIORITY**
* Experiment with typed graphs context (i.e., a global graph attribute which might embed variables as year_progress).
* ~~Revise functions in `model_utils.py` to use GIS utilities.~~ **DONE**
* ~~Compute edge length and azimuth using GIS utilities, and add the latter to edge features.~~ **DONE**
* Investigate the use of fourier features for nodes and edges, and revise embedder as outlined in `model_util.py` comments. See also https://bmild.github.io/fourfeat/index.html. (Medium priority)
* Add utilities and a notebook to inspect the encoder and decoder graphs. **HIGH PRIORITY**
* Project icosahedral meshes produced by functions in `icosahedral_mesh.py` to WGS84 using GIS utilities, and produce either a `.msh` file or a `mesh_graph.MeshData`, such that the global ocean model can be init with either a triangular mesh or a geodesic grid. **HIGH PRIORITY**
* Refactor code base to have a `models` module, move the `model.py` and all global ocean related code to a model-specific module. Update `pyproject.toml` accordingly. (Low priority)
* Add routines to check graph statistics (number of nodes, edges, degree, etc.). In particular, one should check that all grid nodes are connected to the processor. **HIGH PRIORITY** 
* Add routines to compute mesh quality metrics (e.g., mean edge length, mean node degree, etc.). In particular, the volume-length metric should be implemented, see: https://scicomp.stackexchange.com/a/27095 **HIGH PRIORITY**
* Implement a script that produces a shapefile with an equilateral triangle with some edge length at the south pole, then use the `geospatial_mesh.py` functions to produce a mesh pretending that was the coastline. Then vary the edge length and check the mesh quality metrics, one should see some peaks corresponding to geodetic polyhedra. (Low priority)
* Add a proof that geodetic polyhedra are optimal (i.e., they maximize the volume-length metric). (Low priority)
* Add a proof of the algorithm sketched in `graph_pruning.py`. (Low priority)
* Implement the algorithm sketched in `graph_pruning.py`, then implement a derived predictor class that does domain decomposition and measure memory consumption of the graph pruned version. (Low priority)
* Implement pipeline-parallelism for the autoregressive predictor. (Medium priority)
* Set up documentation with Sphinx and add documentation generation to GitHub CI. (Low priority)
* Streamline dataset downloading and post-processing, in particular the download-and-postprocess then merge-and-postprocess workflow should be implemented as a single operation. (Low priority)
* Revise older code to use Python logging properly (e.g., use a `logger = logging.getLogger(__name__)`) and check that expensive format strings are not interpolated. (Low priority)
* Add `oceanbench` to dependencies and implement the validation code. **HIGH PRIORITY**
* Finish the notebook with technical notes on data download and post-processing.
* Port download dataset script to zarr3. (Low priority)
* Run all notebooks on Leonardo using papermill, validate results. (Low priority)
* Download a reduced resolution version of the ARCO-OCEAN and update to AWS. (Medium priority)
* Integrate and revise notebooks to explain details of mesh generation. (Medium priority)
* Experiment with ideas contained in ICLR25 proposal, add routines to fit on meshes and project back to Cartesian grids, perform experiments on reconstruction errors. (Low priority)
* Choose a test suite (e.g. Pytest), update original codebase tests (using abseil/unittest), and add tests for the new code. Implement CI using GitHub actions. (Low priority)
* Add animation and plot routines to `plot_utils.py`. (Low priority)
* Check that accumulated variables in ARCO-OCEAN are consistent (e.g. they use the (t-24h, t) convention). **HIGH PRIORITY**
* Document `--add-spack-mirrors` in `leonardo/environment/setup.sh`, check that other users are able to reproduce environment creation. (Low priority)
* Update README with current status. (Low priority)
* Merge into main branch. (Low priority)
* Most of the documentation is missing or outdated, fix it. (Low priority)
* Notebooks have not been updated to the current version of library code and might be broken. (Low priority)
* Create a similar model for Mediterranean sea biogeochemistry, it should require minimal changes to download scripts and model code. However, more thinking is needed on how to deal with open boundaries. (Low priority)
* Experiment with typed graphs, especially for coastline boundary conditions (i.e., use a typed graph connecting GLOFAS only to processor graph boundary nodes). (Low priority)
* Train the f*cking model. **EXTREME PRIORITY**