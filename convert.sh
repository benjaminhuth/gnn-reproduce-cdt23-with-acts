#!/bin/bash

source setup_acorn.sh

export CUBLAS_WORKSPACE_CONFIG=:4096:8
#export TORCH_LOGS=+dynamo
#export TORCHDYNAMO_VERBOSE=1

SCRIPT_ARGS="$@"

function convert {
    DIR=$(dirname $3)

    echo "convert $3 -> $DIR/$4"
    python3 scripts/save_full_model.py $SCRIPT_ARGS \
        --model-name $1 \
        --stage $2 \
        --checkpoint $3


    mv "$2-$1.pt" "$DIR/$4.pt"
    mv "$2-$1.onnx" "$DIR/$4.onnx"
    mv "$2-$1.so" "$DIR/$4.so"
    mv "$2-$1_sigmoid.pt" "$DIR/$4_sigmoid.pt"
    mv "$2-$1_sigmoid.onnx" "$DIR/$4_sigmoid.onnx"
    mv "$2-$1_sigmoid.so" "$DIR/$4_sigmoid.so"

    mv "hparams.yaml" "$DIR/$4.hparams.yaml"
}


#convert InteractionGNN2 edge_classifier "ctd23/data/GNN_IN2_epochs169.ckpt" gnn.test
convert InteractionGNN2 edge_classifier "rel24/data/best_latent128_LN--val_loss=0.000409-epoch=77.ckpt" gnn

#convert MetricLearning graph_construction "ctd23/data/best-11292882-f1_0.010190.ckpt" metric_learning.pt
#convert Filter edge_classifier "./ctd23/data/best-11984324-auc=0.967753.ckpt" filter.pt
#convert InteractionGNN2WithPyG edge_classifier "./ctd23/data/best-21796495-val_loss=0.000755-epoch=91.ckpt" gnn_metric_learning.pt
