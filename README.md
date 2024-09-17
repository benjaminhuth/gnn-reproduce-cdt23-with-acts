# gnn-reproduce-cdt23-with-acts


## Setup

*As of 2024/09/17*

* Build custom branch ModuleMapGraph
  * https://gitlab.cern.ch/bhuth/ModuleMapGraph, refactor-module-map-graph, ee992a140baacba9d3fb86201e0ef75a9d938621
  * With cuda support

* Dependencies for ACTS:
  * libtorch (2.4.0 tested to be working)
  * torch-scatter (https://github.com/rusty1s/pytorch_scatter)
  * ModuleMapGraph
  * ROOT, Boost

* Build custom version of ACTS (not yet in main)
  * https://github.com/benjaminhuth/acts, feature/integrate-module-map, 8f26a22fea096b43996186d9b7ead5cf53886b83
  * cmake options: `-D ACTS_BUILD_PLUGIN_EXATRKX=ON -D ACTS_BUILD_EXAMPLES_PYTHONBINDINGS=ON -D ACTS_BUILD_EXAMPLES_EXATRKX=ON -D ACTS_EXATRKX_ENABLE_TORCH=ON -D ACTS_EXATRKX_ENABLE_MODULEMAP=ON -D ACTS_EXATRKX_ENABLE_CUDA=ON`
  * Optionally with `-D ACTS_BUILD_PLUGIN_GEOMODEL=ON` when fitting should be supported (requires GeoModel ITK databases for strips and pixels)

* Main pipeline is steered in `scripts/common_pipeline.py`
  * Variants (module map, metric learning) are configured in `scripts/module_map_pipeline.py` / `scripts/metric_learning_pipeline.py`
  * Fit is only performed with `--fit` option
  * Setup environment with `source <acts-build>/python/setup.sh`

* Required files:
  * Root files with event data
  * serialized NNs in torchscript format `(*.pt)`

