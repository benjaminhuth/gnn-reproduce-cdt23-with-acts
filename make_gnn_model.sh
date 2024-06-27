#!/bin/bash

export PYTHONPATH=$HOME/CERN/acorn:$PYTHONPATH
source ../.acorn-ve/bin/activate

MODEL=InteractionGNN2
STAGE=edge_classifier

python3 scripts/save_full_model.py \
    --model $MODEL \
    --stage $STAGE \
    --checkpoint data/GNN_IN2_epochs169.ckpt

mv "$STAGE-$MODEL.pt" data/gnn.pt
