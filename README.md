# GraphCast: Learning skillful medium-range global weather forecasting

> [!NOTE]
> This repository contains a fork of the original GraphCast model. The code has been changed 
> to deploy it on Leonardo, the supercomputer hosted and managed by CINECA. 
> These include
>   1. Addition of a few scripts to setup a working environment in the [leonardo directory](./leonardo) (loading the right modules, download and install packages, 
downloading local copies of the datasets and parameters from the Google Cloud bucket of the project, and ERA5 data from the WeatherBench2 dataset).
>   2. Slight modifications to [graphcast_demo.ipynb](./graphcast_demo.ipynb) to work with local data.
>   3. Custom dataloaders (with sharding support to use multiple GPUs) to run experiments on Leonardo using data from the WeatherBench2 dataset.
>   4. Additional gradient checkpointing to comply with the available memory on Leonardo (64GB per GPU). [WIP]
>
> Notice that, with the current NVIDIA drivers on Leonardo, the latest working version of JAX is 0.4.16. Indeed, from the documentation of Jax: “You should use 
> an NVIDIA driver version that is at least as new as your NVIDIA CUDA toolkit’s corresponding driver version”. The available versions on Leonardo are CUDA/12.1 and CUDA/12.3, 
> while the version of the drivers is 530.30.02, as can be read [here](https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html#id5) the version to use is CUDA/12.1.
> 
> The end goal of this project is to adapt the model to ocean biogeochemistry forecasting. 
> Intermediate steps in the process are making GraphCast work with arbitray grid/meshes, and understand how to deal with boundary conditions.  


This package contains example code to run and train [GraphCast](https://www.science.org/doi/10.1126/science.adi2336).
It also provides three pretrained models:

1.  `GraphCast`, the high-resolution model used in the GraphCast paper (0.25 degree
resolution, 37 pressure levels), trained on ERA5 data from 1979 to 2017,

2.  `GraphCast_small`, a smaller, low-resolution version of GraphCast (1 degree
resolution, 13 pressure levels, and a smaller mesh), trained on ERA5 data from
1979 to 2015, useful to run a model with lower memory and compute constraints,

3.  `GraphCast_operational`, a high-resolution model (0.25 degree resolution, 13
pressure levels) pre-trained on ERA5 data from 1979 to 2017 and fine-tuned on
HRES data from 2016 to 2021. This model can be initialized from HRES data (does
not require precipitation inputs).

The model weights, normalization statistics, and example inputs are available on [Google Cloud Bucket](https://console.cloud.google.com/storage/browser/dm_graphcast).

Full model training requires downloading the
[ERA5](https://www.ecmwf.int/en/forecasts/datasets/reanalysis-datasets/era5)
dataset, available from [ECMWF](https://www.ecmwf.int/). This can best be
accessed as Zarr from [Weatherbench2's ERA5 data](https://weatherbench2.readthedocs.io/en/latest/data-guide.html#era5) (see the 6h downsampled versions).

## Overview of files

The best starting point is to open `graphcast_demo.ipynb` in [Colaboratory](https://colab.research.google.com/github/deepmind/graphcast/blob/master/graphcast_demo.ipynb), which gives an
example of loading data, generating random weights or load a pre-trained
snapshot, generating predictions, computing the loss and computing gradients.
The one-step implementation of GraphCast architecture, is provided in
`graphcast.py`.

### Brief description of library files:

*   `autoregressive.py`: Wrapper used to run (and train) the one-step GraphCast
    to produce a sequence of predictions by auto-regressively feeding the
    outputs back as inputs at each step, in JAX a differentiable way.
*   `casting.py`: Wrapper used around GraphCast to make it work using
    BFloat16 precision.
*   `checkpoint.py`: Utils to serialize and deserialize trees.
*   `data_utils.py`: Utils for data preprocessing.
*   `deep_typed_graph_net.py`: General purpose deep graph neural network (GNN)
    that operates on `TypedGraph`'s where both inputs and outputs are flat
    vectors of features for each of the nodes and edges. `graphcast.py` uses
    three of these for the Grid2Mesh GNN, the Multi-mesh GNN and the Mesh2Grid
    GNN, respectively.
*   `grid_mesh_connectivity.py`: Tools for converting between regular grids on a
    sphere and triangular meshes.
*   `icosahedral_mesh.py`: Definition of an icosahedral multi-mesh.
*   `losses.py`: Loss computations, including latitude-weighting.
*   `model.py`: The main GraphCast model architecture for one-step of
    predictions.
*   `model_utils.py`: Utilities to produce flat node and edge vector features
    from input grid data, and to manipulate the node output vectors back
    into a multilevel grid data.
*   `normalization.py`: Wrapper for the one-step GraphCast used to normalize
    inputs according to historical values, and targets according to historical
    time differences.
*   `predictor_base.py`: Defines the interface of the predictor, which GraphCast
    and all of the wrappers implement.
*   `rollout.py`: Similar to `autoregressive.py` but used only at inference time
    using a python loop to produce longer, but non-differentiable trajectories.
*   `solar_radiation.py`: Computes Top-Of-the-Atmosphere (TOA) incident solar
    radiation compatible with ERA5. This is used as a forcing variable and thus
    needs to be computed for target lead times in an operational setting.
*   `typed_graph.py`: Definition of `TypedGraph`'s.
*   `typed_graph_net.py`: Implementation of simple graph neural network
    building blocks defined over `TypedGraph`'s that can be combined to build
    deeper models.
*   `xarray_jax.py`: A wrapper to let JAX work with `xarray`s.
*   `xarray_tree.py`: An implementation of tree.map_structure that works with
    `xarray`s.


### Dependencies.

[Chex](https://github.com/deepmind/chex),
[Dask](https://github.com/dask/dask),
[Haiku](https://github.com/deepmind/dm-haiku),
[JAX](https://github.com/google/jax),
[JAXline](https://github.com/deepmind/jaxline),
[Jraph](https://github.com/deepmind/jraph),
[Numpy](https://numpy.org/),
[Pandas](https://pandas.pydata.org/),
[Python](https://www.python.org/),
[SciPy](https://scipy.org/),
[Tree](https://github.com/deepmind/tree),
[Trimesh](https://github.com/mikedh/trimesh) and
[XArray](https://github.com/pydata/xarray).


### License and attribution

The Colab notebook and the associated code are licensed under the Apache
License, Version 2.0. You may obtain a copy of the License at:
https://www.apache.org/licenses/LICENSE-2.0.

The model weights are made available for use under the terms of the Creative
Commons Attribution-NonCommercial-ShareAlike 4.0 International
(CC BY-NC-SA 4.0). You may obtain a copy of the License at:
https://creativecommons.org/licenses/by-nc-sa/4.0/.

The weights were trained on ECMWF's ERA5 and HRES data. The colab includes a few
examples of ERA5 and HRES data that can be used as inputs to the models.
ECMWF data product are subject to the following terms:

1. Copyright statement: Copyright "© 2023 European Centre for Medium-Range Weather Forecasts (ECMWF)".
2. Source www.ecmwf.int
3. Licence Statement: ECMWF data is published under a Creative Commons Attribution 4.0 International (CC BY 4.0). https://creativecommons.org/licenses/by/4.0/
4. Disclaimer: ECMWF does not accept any liability whatsoever for any error or omission in the data, their availability, or for any loss or damage arising from their use.

### Disclaimer

This is not an officially supported Google product.

Copyright 2023 DeepMind Technologies Limited.

### Citation

If you use this work, consider citing our paper ([blog post](https://deepmind.google/discover/blog/graphcast-ai-model-for-faster-and-more-accurate-global-weather-forecasting/), [Science](https://www.science.org/doi/10.1126/science.adi2336), [arXiv](https://arxiv.org/abs/2212.12794)):

```latex
@article{lam2023learning,
  title={Learning skillful medium-range global weather forecasting},
  author={Lam, Remi and Sanchez-Gonzalez, Alvaro and Willson, Matthew and Wirnsberger, Peter and Fortunato, Meire and Alet, Ferran and Ravuri, Suman and Ewalds, Timo and Eaton-Rosen, Zach and Hu, Weihua and others},
  journal={Science},
  volume={382},
  number={6677},
  pages={1416--1421},
  year={2023},
  publisher={American Association for the Advancement of Science}
}
```
