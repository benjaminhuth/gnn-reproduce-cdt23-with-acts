#!/bin/bash

set -e

#export LD_PRELOAD=/lib64/libasan.so.6

source setup_acts.sh

DATA_PREFIX=rel24/data/ATLAS-P2-RUN4-03-00-00_v5_ttbar_uncorr_pu200/user.avallier.38040858.EXT0
DATA_POSTFIX=Dump_GNN4Itk.root

DATA=$(python3 -c "print(','.join([ f'$DATA_PREFIX.{n}.$DATA_POSTFIX' for n in ['_000124', '_000127', '_000074', '_000185'] ]))")
#DATA=$(python3 -c "print(','.join([ f'$DATA_PREFIX.{n}.$DATA_POSTFIX' for n in ['_000127', '_000074', '_000185'] ]))")

#DATA=rel24/data/user.avallier.38040855.EXT0._000005.Dump_GNN4Itk.root
#DATA=rel24/data/Dump_GNN4Itk.root

#GNN=rel24/data/gnn.pt
#GNN=rel24/data/gnn.onnx
GNN=rel24/data/gnn.engine

#OUTPUT_DIR=tmp/rel24/acts
OUTPUT_DIR=tmp/rel24/acts_test
mkdir -p $OUTPUT_DIR
rm -vfr $OUTPUT_DIR/*

export FRANKENSTEIN_ITK=1

ITK_FILE1=athena_surfaces.json
ITK_FILE2=athena_transforms.csv
#ITK_FILE1=ITKPixels.db
#ITK_FILE2=ITKStrips.db

$PREFIX python3 scripts/module_map_pipeline.py "$@" \
    --modulemap rel24/data/ModuleMap_rel24_ttbar_v5_89809evts \
    --data $DATA \
    --gnn $GNN \
    --output $OUTPUT_DIR \
    --itk-material-map $ITK_ROOT/itk_geometry/itk_material_map_json.root \
    --itk-pixel-data $ITK_ROOT/itk_geometry/$ITK_FILE1 \
    --itk-strip-data $ITK_ROOT/itk_geometry/$ITK_FILE2 \
    --no-phi-ovl-sps | tee $OUTPUT_DIR/logfile.log

unset LD_PRELOAD
DIR=$PWD
cd $OUTPUT_DIR
python3 $DIR/scripts/plot_efficiency.py *atlas*.root
python3 $DIR/scripts/plot_timing.py
