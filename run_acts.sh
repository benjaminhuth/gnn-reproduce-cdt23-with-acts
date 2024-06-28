#!/bin/bash

export ACTS_SEQUENCER_DISABLE_FPEMON=1

source $HOME/CERN/setup_deps.sh
source $HOME/CERN/acts/build/python/setup.sh

$PREFIX python3 scripts/cdt23_reco_chain.py \
    --modulemap data/ModuleMap_rel24_ttbar_v5_89809evts \
    --data data/Dump_GNN4Itk.root \
    --gnn data/gnn.pt \
    "$@"
