#!/bin/bash

# Doesn't help
#export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:128"

source setup_acorn.sh

infer ctd23/config/data_reader.yaml

#infer ctd23/config/module_map.yaml
#evaluate ctd23/config/module_map.yaml

#infer ctd23/config/gnn_infer.yaml -c ctd23/data/GNN_IN2_epochs169.ckpt
#evaluate ctd23/config/gnn_eval.yaml -c ctd23/data/GNN_IN2_epochs169.ckpt
