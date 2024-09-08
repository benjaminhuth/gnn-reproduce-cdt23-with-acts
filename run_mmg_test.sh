#!/bin/bash

source $HOME/setup_lcg_cuda.sh
export LD_LIBRARY_PATH=$HOME/gnn/ModuleMapGraph/lib64:$LD_LIBRARY_PATH

export MMG_STACK_FACTOR=8

$PREFIX	$HOME/gnn/ModuleMapGraph/build/GPU/GraphBuilderTest.cu.exe \
		~/gnn/reproduce_cdt23/ctd23/data/ModuleMap.90k \
		~/gnn/reproduce_cdt23/tmp/ctd23/feature_store/trainset/ \
		event000000000-truth.csv "$@"
