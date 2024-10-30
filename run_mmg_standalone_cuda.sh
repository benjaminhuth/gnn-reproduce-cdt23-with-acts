#!/bin/bash

#source $HOME/setup_lcg_cuda.sh
export MMG_ROOT=$HOME/software/ModuleMapGraph
export LD_LIBRARY_PATH=$MMG_ROOT/lib64:$LD_LIBRARY_PATH

mkdir -p tmp/rel24/standalone

INPUT_DIR=tmp/rel24/feature_store/trainset

$PREFIX $MMG_ROOT/bin/GraphBuilder.cu.exe \
    --gpu-nb-blocks=512 \
    --input-dir=$INPUT_DIR \
    --input-filename-pattern="event" \
    --output-dir=tmp/rel24/standalone \
    --give-true-graph=0 \
    --input-module-map=rel24/data/ModuleMap_rel24_ttbar_v5_89809evts \
    --save-graph-on-disk-graphml=0 \
    --save-graph-on-disk-npz=0 \
    --save-graph-on-disk-pyg=0 \
    --save-graph-on-disk-csv=1 \
    --strip-hit-pair=0 \
    --extra-features=0 \
    --gpu-nb-events=1

