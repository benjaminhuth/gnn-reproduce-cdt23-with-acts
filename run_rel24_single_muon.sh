#!/bin/bash

source setup_acts.sh

DATA=rel24/data/Dump_GNN4Itk__1.root

mkdir -p tmp/rel24/single_muons

$PREFIX python3 scripts/module_map_pipeline.py "$@" \
    --modulemap rel24/data/ModuleMap_rel24_ttbar_v5_89809evts \
    --data $DATA \
    --gnn rel24/data/gnn.pt \
    --output tmp/rel24/single_muons \
    --no-phi-ovl-sps \
    --itk-material-map $ITK_ROOT/itk_geometry/itk_material.root \
    --itk-pixel-data $ITK_ROOT/itk_geometry/ITKPixels.db \
    --itk-strip-data $ITK_ROOT/itk_geometry/ITKStrips.db 
