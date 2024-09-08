#!/bin/bash

source $HOME/setup_lcg_cuda.sh
export LD_LIBRARY_PATH=$HOME/gnn/ModuleMapGraph/lib64:$LD_LIBRARY_PATH

$HOME/gnn/ModuleMapGraph/bin/GraphBuilder.cu.exe -j1 -n1 \
	--input-dir=$HOME/gnn/reproduce_cdt23/tmp/ctd23/feature_store/trainset \
	--input-csv=1 \
	--input-filename-pattern=event \
	--input-module-map=$HOME/gnn/reproduce_cdt23/ctd23/data/ModuleMap.90k \
	--gpu-nb-blocks=512 \
	--min-pt-cut=1 \
	--output-dir=. \
	--extra-features=0 \
	--strip-hit-pair=0 \
	--save-graph-on-disk-graphml=0 \
	--save-graph-on-disk-npz=0 \
	--save-graph-on-disk-pyg=0 \
	--save-graph-on-disk-csv=0 \
	--give-true-graph=0
