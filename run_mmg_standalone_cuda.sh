#!/bin/bash

#source $HOME/setup_lcg_cuda.sh

export MMG_ROOT=$HOME/software/ModuleMapGraph/install_master
#export MMG_ROOT=$HOME/software/ModuleMapGraph
export LD_LIBRARY_PATH=$MMG_ROOT/lib64:$LD_LIBRARY_PATH

#MM_FILE=rel24/data/ModuleMap_rel24_ttbar_v5_89809evts
MM_FILE=ctd23/data/ModuleMap.90k

VERSION=ctd23

echo "VERSION: $VERSION"

OUTPUT=tmp/$VERSION/standalone
INPUT_DIR=tmp/$VERSION/feature_store/trainset

mkdir -p $OUTPUT


$PREFIX $MMG_ROOT/bin/GraphBuilder.cu.exe --help

$PREFIX $MMG_ROOT/bin/GraphBuilder.cu.exe \
    -l0 \
    --gpu-nb-blocks=512 \
    --input-dir=$INPUT_DIR \
    --input-filename-pattern="event" \
    --output-dir=$OUTPUT \
    --give-true-graph=0 \
    --input-module-map=$MM_FILE \
    --save-graph-on-disk-graphml=0 \
    --save-graph-on-disk-npz=0 \
    --save-graph-on-disk-pyg=0 \
    --save-graph-on-disk-csv=1 \
    --strip-hit-pair=0 \
    --extra-features=0 \
    --events=1

