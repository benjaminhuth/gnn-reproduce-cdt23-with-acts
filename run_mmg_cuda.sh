#!/bin/bash

export LD_LIBRARY_PATH=$HOME/software/ModuleMapGraph/lib64:$LD_LIBRARY_PATH

$HOME/software/ModuleMapGraph/bin/GraphBuilder.cu.exe -j1 -n1 \
	--input-dir=$HOME/reproduce_gnn_results/tmp/ctd23/feature_store/trainset \
	--input-csv=1 \
	--input-filename-pattern=event \
	--input-module-map=$HOME/reproduce_gnn_results/ctd23/data/ModuleMap.90k \
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
