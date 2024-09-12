#!/bin/bash

source setup_acts.sh

mkdir -p tmp/ctd23/acts

# Take rel24 GNN since we now focus on ModuleMap performance
$PREFIX python3 scripts/metric_learning_pipeline.py \
    --metric-learning ctd23/data/metric_learning.pt \
    --filter ctd23/data/filter.pt \
    --data ctd23/data/GNN4Itk_v2__mc15_14TeV.600012.PhPy8EG_A14_ttbar_hdamp258p75_nonallhad.recon.RDO.e8185_s3770_s3773_r14431__J016.root \
    --gnn ctd23/data/gnn_metric_learning.pt \
    --output tmp/ctd23/acts \
    "$@"

#    --unique-modules-truth-graph \

