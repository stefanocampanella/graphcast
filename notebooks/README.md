# About notebooks

The purposes of the notebooks are

1. Document meshes, datasets, training reports, model performance, etc.
2. Validate code locally and on Leonardo
3. Doing analysis, exploratory work and prototyping
4. Generate images for publication material

Running the notebooks requires an internet connection (i.e., on Leonardo can run on login or serial nodes), 
but the datasets should be downloaded beforehand and located to `notebooks/data/datasets`.

The dataset naming scheme follows the one used for configuration files, and inspired by the GraphCast and WeatherBench 
GCS buckets: 

`<source name>[_<variables subset name>]_tres-<time resolution>_res-<spatial resolution in degrees>_levels-<number of (depth) levels>.zip`