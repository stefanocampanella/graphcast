# TODO list

0. Finish the notebook with technical notes on data download and post-processing.
1. Port download dataset script to zarr3.
2. Download test datasets.
3. Run notebooks on Leonardo using papermill, validate results.
4. Add scripts to merge GLORY12 and ERA5 datasets (note: **they use a different convention for coordinates**).
5. Add scripts to compute climatologies, averages, and diff averages (required for normalization).
6. Add a notebook to explain details of mesh generation (target mesh size criteria).
7. Add routines to fit on meshes and project back to Cartesian grids.
8. Perform experiments on reconstruction errors.
9. Add tests for the new code.
10. Switch to Pytest from abseil/unittest, implement GitHub actions CI.
11. Add graph analysis of icosahedral multi-mesh and hierarchical meshes (edge length distribution, number of edges and nodes, nodes degree).
12. Move (write/rewrite) dataloader to PyGrain, remove PyTorch from dependencies.
