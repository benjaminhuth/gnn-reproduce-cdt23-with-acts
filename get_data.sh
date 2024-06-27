#!/bin/bash

mkdir -p data

MM_PATH=/eos/atlas/atlascerngroupdisk/perf-idtracking/GNN4ITk/ATLAS-P2-RUN4-03-00-00_Rel.24/module_maps/v5

MMT_FILE=$MM_PATH/ModuleMap_rel24_ttbar_v5_89809evts.triplets.root
MMD_FILE=$MM_PATH/ModuleMap_rel24_ttbar_v5_89809evts.doublets.root
DUMP_FILE=/eos/user/b/bhuth/data/Dump_GNN4Itk.root

CDT23_CKPT=/eos/user/s/scaillou/CTD_2023/model_store/gnn/GNN_IN2_epochs169.ckpt

if false; then
    sshpass -p "$CERN_PWD" scp bhuth@lxplus.cern.ch:$MMD_FILE ./data/
    sshpass -p "$CERN_PWD" scp bhuth@lxplus.cern.ch:$MMT_FILE ./data/
    sshpass -p "$CERN_PWD" scp bhuth@lxplus.cern.ch:$DUMP_FILE ./data/
    sshpass -p "$CERN_PWD" scp bhuth@lxplus.cern.ch:$CDT23_CKPT ./data/
else
    cp $MMD_FILE ./data/
    cp $MMT_FILE ./data/
    cp $DUMP_FILE ./data/
    cp $CDT23_CKPT ./data/
fi
