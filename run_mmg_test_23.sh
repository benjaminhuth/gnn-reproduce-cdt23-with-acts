#!/bin/bash

export LD_LIBRARY_PATH=$HOME/software/ModuleMapGraph/lib64:$LD_LIBRARY_PATH

export MMG_STACK_FACTOR=8

$PREFIX $HOME/software/ModuleMapGraph/build/GPU/GraphBuilderTest.cu.exe \
    ~/reproduce_gnn_results/ctd23/data/ModuleMap.90k \
    ~/reproduce_gnn_results/tmp/ctd23/feature_store/trainset/ \
    event000000000-truth.csv "$@"
