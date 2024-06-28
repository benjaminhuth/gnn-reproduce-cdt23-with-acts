#!/bin/bash

export ACTS_SEQUENCER_DISABLE_FPEMON=1


source $HOME/setup_lcg_cuda.sh
source $HOME/acts/build/python/setup.sh

$PREFIX python3 scripts/cdt23_reco_chain.py \
    --modulemap rel24/data/ModuleMap_rel24_ttbar_v5_89809evts \
    --data rel24/data/Dump_GNN4Itk.root \
    --gnn rel24/data/gnn.pt \
    "$@"
