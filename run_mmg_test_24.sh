#!/bin/bash

source $HOME/setup_lcg_cuda.sh
export LD_LIBRARY_PATH=$HOME/gnn/ModuleMapGraph/lib64:$LD_LIBRARY_PATH

export MMG_STACK_FACTOR=1

$PREFIX	$HOME/gnn/ModuleMapGraph/build/GPU/GraphBuilderTest.cu.exe \
		~/gnn/reproduce_cdt23/rel24/data/ModuleMap_rel24_ttbar_v5_89809evts \
		~/gnn/reproduce_cdt23/tmp/rel24/feature_store/trainset/ \
		event000000000-truth.csv "$@"
