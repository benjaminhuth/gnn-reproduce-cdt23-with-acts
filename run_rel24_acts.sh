#!/bin/bash

source setup_acts.sh

DATA=rel24/data/user.avallier.38040858.EXT0._000124.Dump_GNN4Itk.root
#DATA=rel24/data/user.avallier.38040855.EXT0._000005.Dump_GNN4Itk.root
#DATA=rel24/data/Dump_GNN4Itk.root

$PREFIX python3 scripts/module_map_pipeline.py "$@" \
    --modulemap rel24/data/ModuleMap_rel24_ttbar_v5_89809evts \
    --data $DATA \
    --gnn rel24/data/gnn.pt \
    --output tmp/rel24/acts \
    --itk-material-map $ITK_ROOT/itk_geometry/itk_material_map.root \
    --itk-pixel-data $ITK_ROOT/itk_geometry/ITKPixels.db \
    --itk-strip-data $ITK_ROOT/itk_geometry/ITKStrips.db \
    --no-phi-ovl-sps
