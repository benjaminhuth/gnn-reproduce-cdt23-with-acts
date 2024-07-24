#!/bin/bash

source $HOME/setup_lcg_cuda.sh
export LD_LIBRARY_PATH=$HOME/ModuleMapGraph/lib64:$LD_LIBRARY_PATH

mkdir -p tmp/rel24/standalone

INPUT_DIR=tmp/rel24/feature_store/trainset

$PREFIX $HOME/ModuleMapGraph/bin/GraphBuilder.exe \
    --input-dir=$INPUT_DIR \
    --input-filename-pattern="event" \
    --output-dir=tmp/rel24/standalone \
    --give-true-graph=0 \
    --input-module-map=rel24/data/ModuleMap_rel24_ttbar_v5_89809evts \
    --save-graph-on-disk-graphml=0 \
    --save-graph-on-disk-npz=0 \
    --save-graph-on-disk-pyg=0 \
    --save-graph-on-disk-csv=1 \
    --min-nhits=0 \
    --min-pt-cut=0 \
    --max-pt-cut=1000000 \
    --phi-slice=0 \
    --cut1-phi-slice=0 \
    --cut2-phi-slice=0 \
    --eta-region=0 \
    --cut1-eta=0 \
    --cut2-eta=0 \
    --strip-hit-pair=0 \
    --extra-features=0

