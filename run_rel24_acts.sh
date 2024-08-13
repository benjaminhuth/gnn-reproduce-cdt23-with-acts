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
try_source $HOME/gnn/acts/build/python/setup.sh
try_source $HOME/CERN/acts/build/python/setup.sh

DATA=rel24/data/user.avallier.38040855.EXT0._000005.Dump_GNN4Itk.root
#DATA=rel24/data/Dump_GNN4Itk.root

$PREFIX python3 scripts/module_map_pipeline.py "$@" \
    --modulemap rel24/data/ModuleMap_rel24_ttbar_v5_89809evts \
    --data $DATA \
    --gnn rel24/data/gnn.pt \
    --output tmp/rel24/acts \
    #--no-phi-ovl-sps \
