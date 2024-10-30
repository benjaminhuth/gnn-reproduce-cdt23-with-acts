#!/bin/bash

set -e
source setup_acts.sh

DATA=rel24/data/user.avallier.38040858.EXT0._000124.Dump_GNN4Itk.root

#DATA=rel24/data/user.avallier.38040855.EXT0._000005.Dump_GNN4Itk.root
#DATA=rel24/data/Dump_GNN4Itk.root

#GNN=rel24/data/gnn.pt
#GNN=rel24/data/gnn.onnx
GNN=rel24/data/gnn_best.engine

rm -vfr tmp/rel24/acts/csv
rm -vf tmp/rel24/acts/*.root

ITK_FILE1=athena_surfaces.json
ITK_FILE2=athena_transforms.csv
#ITK_FILE1=ITKPixels.db
#ITK_FILE2=ITKStrips.db

$PREFIX python3 scripts/module_map_pipeline.py "$@" \
    --modulemap rel24/data/ModuleMap_rel24_ttbar_v5_89809evts \
    --data $DATA \
    --gnn $GNN \
    --output tmp/rel24/acts \
    --itk-material-map $ITK_ROOT/itk_geometry/itk_material_map_json.root \
    --itk-pixel-data $ITK_ROOT/itk_geometry/$ITK_FILE1 \
    --itk-strip-data $ITK_ROOT/itk_geometry/$ITK_FILE2 \
    --no-phi-ovl-sps | tee tmp/rel24/acts/logfile.log

DIR=$PWD
(cd tmp/rel24/acts/ && python3 $DIR/scripts/plot_efficiency.py)
