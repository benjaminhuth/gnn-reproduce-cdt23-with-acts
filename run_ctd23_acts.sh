#!/bin/bash

source setup_acts.sh

mkdir -p tmp/ctd23/acts


if [[ -v PERF ]]; then
    rm -f tmp/ctd23/perf.data
    PREFIX="perf record --call-graph=dwarf -F1000 --output=tmp/ctd23/perf.data "
fi

DATA="ctd23/data/GNN4Itk_v2__mc15_14TeV.600012.PhPy8EG_A14_ttbar_hdamp258p75_nonallhad.recon.RDO.e8185_s3770_s3773_r14431__J016.root"

# Take rel24 GNN since we now focus on ModuleMap performance
$PREFIX python3 scripts/module_map_pipeline.py \
    --modulemap ctd23/data/ModuleMap.90k \
    --data $DATA \
    --gnn ctd23/data/gnn.pt \
    --output tmp/ctd23/acts \
    "$@"

if [[ -v PERF ]]; then
    perf script -i tmp/ctd23/perf.data > tmp/ctd23/perf.out
fi
