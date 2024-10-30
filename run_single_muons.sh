#!/bin/bash

source setup_acts.sh

DATA=rel24/data/Dump_GNN4Itk__1.root

OUTDIR=tmp/rel24/single_muons
mkdir -p $OUTDIR

$PREFIX python3 scripts/module_map_pipeline.py "$@" \
    --modulemap rel24/data/ModuleMap_rel24_ttbar_v5_89809evts \
    --data $DATA \
    --gnn rel24/data/gnn.pt \
    --output $OUTDIR \
    --no-phi-ovl-sps \
    --itk-material-map $ITK_ROOT/itk_geometry/itk_material_map.root \
    --itk-pixel-data $ITK_ROOT/itk_geometry/athena_surfaces.json \
    --itk-strip-data $ITK_ROOT/itk_geometry/athena_transforms.csv

DIR=$PWD
(cd $OUTDIR && ls && python3 $DIR/scripts/plot_efficiency.py)
