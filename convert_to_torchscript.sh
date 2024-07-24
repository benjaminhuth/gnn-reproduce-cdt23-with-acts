#!/bin/bash

source setup_acorn.sh

MODEL=InteractionGNN2
STAGE=edge_classifier

function convert {
    DIR=$(dirname $3)

    echo "convert $3 -> $DIR/$4"
    python3 scripts/save_full_model.py \
        --model $1 \
        --stage $2 \
        --checkpoint $3

    mv "$2-$1.pt" "$DIR/$4"
    mv "hparams.json" "$DIR/$4.hparams.json"
}


#convert InteractionGNN2 edge_classifier "ctd23/data/GNN_IN2_epochs169.ckpt" gnn.pt
#convert InteractionGNN2 edge_classifier "rel24/data/best_latent128_LN--val_loss=0.000409-epoch=77.ckpt" gnn.pt

convert MetricLearning graph_construction "ctd23/data/best-11292882-f1_0.010190.ckpt" metric_learning.pt
