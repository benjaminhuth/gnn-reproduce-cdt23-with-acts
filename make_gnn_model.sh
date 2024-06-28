#!/bin/bash

export PYTHONPATH=$HOME/CERN/acorn:$PYTHONPATH
source ../.acorn-ve/bin/activate

MODEL=InteractionGNN2
STAGE=edge_classifier

function convert {
    echo "convert $1"
    python3 scripts/save_full_model.py \
        --model $MODEL \
        --stage $STAGE \
        --checkpoint $1

    DIR=$(dirname $1)
    mv "$STAGE-$MODEL.pt" "$DIR/gnn.pt"
    echo "---------------------------"
}


convert "ctd23/data/GNN_IN2_epochs169.ckpt"
convert "rel24/data/best_latent128_LN--val_loss=0.000409-epoch=77.ckpt"
