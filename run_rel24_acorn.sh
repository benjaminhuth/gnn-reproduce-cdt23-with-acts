#!/bin/bash

source setup_acorn.sh

#infer rel24/config/data_reader.yaml

#infer rel24/config/module_map.yaml
#evaluate rel24/config/module_map.yaml

infer rel24/config/gnn.yaml -c "rel24/data/best_latent128_LN--val_loss=0.000409-epoch=77.ckpt"
evaluate rel24/config/gnn.yaml -c "rel24/data/best_latent128_LN--val_loss=0.000409-epoch=77.ckpt"
