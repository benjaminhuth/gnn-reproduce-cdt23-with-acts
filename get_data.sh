#!/bin/bash

mkdir -p data

MM_PATH=/eos/atlas/atlascerngroupdisk/perf-idtracking/GNN4ITk/ATLAS-P2-RUN4-03-00-00_Rel.24/module_maps/v5

MMT_FILE=$MM_PATH/ModuleMap_rel24_ttbar_v5_89809evts.triplets.root
MMD_FILE=$MM_PATH/ModuleMap_rel24_ttbar_v5_89809evts.doublets.root
DUMP_FILE=/eos/user/p/pibutti/sw/run/gnn/Dump_GNN4Itk.root

CDT23_CKPT=/eos/user/s/scaillou/CTD_2023/model_store/gnn/GNN_IN2_epochs169.ckpt

sshpass -p "$CERN_PWD" scp bhuth@lxplus.cern.ch:$MMD_FILE ./data/
sshpass -p "$CERN_PWD" scp bhuth@lxplus.cern.ch:$MMT_FILE ./data/
sshpass -p "$CERN_PWD" scp bhuth@lxplus.cern.ch:$DUMP_FILE ./data/
sshpass -p "$CERN_PWD" scp bhuth@lxplus.cern.ch:$CDT23_CKPT ./data/
