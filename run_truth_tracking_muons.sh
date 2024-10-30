#!/bin/bash

source setup_acts.sh

mkdir -p tmp/rel24/truth_tracking

MMAP=$ITK_ROOT/itk_geometry/itk_material_map_json.root 
FILE1=$ITK_ROOT/itk_geometry/athena_surfaces.json 
FILE2=$ITK_ROOT/itk_geometry/athena_transforms.csv

#MMAP=$ITK_ROOT/itk_geometry/itk_material_map.root 
#FILE1=$ITK_ROOT/itk_geometry/ITKPixels.db 
#FILE2=$ITK_ROOT/itk_geometry/ITKStrips.db

python3 scripts/truth_tracking.py "$@" \
  -i rel24/data/Dump_GNN4Itk__1.root \
  -o tmp/rel24/truth_tracking \
  --material-map $MMAP \
  --itk-file1 $FILE1 \
  --itk-file2 $FILE2
