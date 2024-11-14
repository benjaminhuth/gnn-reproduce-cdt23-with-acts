#!/bin/bash

set -e

source setup_acts.sh

GNN=rel24/data/gnn.pt
#GNN=rel24/data/gnn.onnx
#GNN=rel24/data/gnn_best.engine

#OUTPUT_DIR=tmp/rel24/acts
OUTPUT_DIR=tmp/rel24/dump_graph_and_tracks
mkdir -p $OUTPUT_DIR
rm -vfr $OUTPUT_DIR/*

ITK_FILE1=athena_surfaces.json
ITK_FILE2=athena_transforms.csv
#ITK_FILE1=ITKPixels.db
#ITK_FILE2=ITKStrips.db

$PREFIX python3 scripts/dump_prototracks.py "$@" \
    --modulemap rel24/data/ModuleMap_rel24_ttbar_v5_89809evts \
    --input-file rel24/data/ATLAS-P2-RUN4-03-00-00_v5_ttbar_uncorr_pu200/user.avallier.38040858.EXT0._000074.Dump_GNN4Itk.root \
    --gnn $GNN \
    --output $OUTPUT_DIR \
    --exclude-phi-ovl-sps=True | tee $OUTPUT_DIR/logfile.log

$PREFIX python3 scripts/dump_constructed_graph.py "$@" \
    --modulemap rel24/data/ModuleMap_rel24_ttbar_v5_89809evts \
    --input-file rel24/data/ATLAS-P2-RUN4-03-00-00_v5_ttbar_uncorr_pu200/user.avallier.38040858.EXT0._000074.Dump_GNN4Itk.root \
    --gnn $GNN \
    --output $OUTPUT_DIR \
    --exclude-phi-ovl-sps=True | tee $OUTPUT_DIR/logfile.log
