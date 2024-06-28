#!/bin/bash

function copy_file {
    FILENAME=$(basename $1)

    if test -f "$2/$FILENAME"; then
        echo "'$2/$FILENAME' already present, skip"
        return
    else
        echo "'$2/$FILENAME' not there, download"
    fi

    {
        sshpass -p "$CERN_PWD" scp bhuth@lxplus.cern.ch:$1 $2
    } || {
        cp $1 $2
    }
}

# Rel 24
MM_PATH=/eos/atlas/atlascerngroupdisk/perf-idtracking/GNN4ITk/ATLAS-P2-RUN4-03-00-00_Rel.24/module_maps/v5

MMT_FILE=$MM_PATH/ModuleMap_rel24_ttbar_v5_89809evts.triplets.root
MMD_FILE=$MM_PATH/ModuleMap_rel24_ttbar_v5_89809evts.doublets.root
DUMP_FILE=/eos/user/b/bhuth/data/Dump_GNN4Itk.root
CKPT=/eos/user/s/scaillou/Rel24/model_store/edge_classifier/best_latent128_LN--val_loss=0.000409-epoch=77.ckpt

# CTD 23
CDT23_CKPT=/eos/user/s/scaillou/CTD_2023/model_store/gnn/GNN_IN2_epochs169.ckpt
CDT23_MM=/eos/user/s/scaillou/CTD_2023/model_store/module_map/MMtriplet_1GeV_3hits_noE__merged__sorted.txt
CDT23_MMT=/eos/atlas/atlascerngroupdisk/perf-idtracking/GNN4ITk/module_maps/MM_23/MMtriplet_1GeV_3hits_noE__merged__sorted_converted.root
CDT23_DATA=/eos/atlas/atlascerngroupdisk/perf-idtracking/GNN4ITk/GNN4Itk_v2__mc15_14TeV.600012.PhPy8EG_A14_ttbar_hdamp258p75_nonallhad.recon.RDO.e8185_s3770_s3773_r14431/GNN4Itk_v2__mc15_14TeV.600012.PhPy8EG_A14_ttbar_hdamp258p75_nonallhad.recon.RDO.e8185_s3770_s3773_r14431__J016.root

# Copy
mkdir -p rel24/data
mkdir -p ctd23/data

copy_file $MMD_FILE ./rel24/data
copy_file $MMT_FILE ./rel24/data
copy_file $DUMP_FILE ./rel24/data
copy_file $CKPT ./rel24/data

copy_file $CDT23_CKPT ./ctd23/data
copy_file $CDT23_MM ./ctd23/data
copy_file $CDT23_DATA ./ctd23/data
