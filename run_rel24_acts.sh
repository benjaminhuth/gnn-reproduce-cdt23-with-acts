#!/bin/bash

function try_source {
    if [[ -f $1 ]]; then
        source $1
    else
        echo "Could not source '$1'"
    fi
}


export ACTS_SEQUENCER_DISABLE_FPEMON=1

try_source $HOME/setup_lcg_cuda.sh
try_source $HOME/acts/build/python/setup.sh
try_source $HOME/CERN/acts/build/python/setup.sh

$PREFIX python3 scripts/cdt23_reco_chain.py \
    --modulemap rel24/data/ModuleMap_rel24_ttbar_v5_89809evts \
    --data rel24/data/Dump_GNN4Itk.root \
    --gnn rel24/data/gnn.pt \
    "$@"
