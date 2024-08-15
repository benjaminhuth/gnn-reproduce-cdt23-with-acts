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
        cp $1 $2
    } || {
        kinit
        cp $1 $2
    }
}

# Rel 24
MM_PATH=/eos/atlas/atlascerngroupdisk/perf-idtracking/GNN4ITk/ATLAS-P2-RUN4-03-00-00_Rel.24/module_maps/v5

MMT_FILE=$MM_PATH/ModuleMap_rel24_ttbar_v5_89809evts.triplets.root
MMD_FILE=$MM_PATH/ModuleMap_rel24_ttbar_v5_89809evts.doublets.root
#DUMP_FILE=/eos/user/b/bhuth/data/Dump_GNN4Itk.root
#DUMP_FILE=/eos/atlas/atlascerngroupdisk/perf-idtracking/GNN4ITk/ATLAS-P2-RUN4-03-00-00_Rel.24/for_debug/user.avallier.37060564.EXT0._000001.Dump_GNN4Itk.root
#DUMP_FILE2=/eos/atlas/atlascerngroupdisk/perf-idtracking/GNN4ITk/ATLAS-P2-RUN4-03-00-00_Rel.24/dumps/v5/ttbar/pu200/user.avallier.mc21_14TeV.601229.PhPy8EG_A14_ttbar_hdamp258p75_SingleLep.DumpGNN4ITk_v5.e8481_s4149_r14697_EXT0/user.avallier.38040855.EXT0._000005.Dump_GNN4Itk.root
DUMP_FILE=/eos/user/b/bhuth/data/user.avallier.38040858.EXT0._000124.Dump_GNN4Itk.root
CKPT=/eos/user/s/scaillou/Rel24/model_store/edge_classifier/best_latent128_LN--val_loss=0.000409-epoch=77.ckpt

# CTD 23
CDT23_CKPT=/eos/user/s/scaillou/CTD_2023/model_store/gnn/GNN_IN2_epochs169.ckpt
#CDT23_MM=/eos/user/s/scaillou/CTD_2023/model_store/module_map/MMtriplet_1GeV_3hits_noE__merged__sorted.txt
#CDT23_MMT=/eos/atlas/atlascerngroupdisk/perf-idtracking/GNN4ITk/module_maps/MM_23/MMtriplet_1GeV_3hits_noE__merged__sorted_converted.root
CDT23_MMD=/eos/atlas/atlascerngroupdisk/perf-idtracking/GNN4ITk/ModuleMapGraph_CI/MM2023/ModuleMap.90k.doublets.root
CDT23_MMT=/eos/atlas/atlascerngroupdisk/perf-idtracking/GNN4ITk/ModuleMapGraph_CI/MM2023/ModuleMap.90k.triplets.root
CDT23_DATA=/eos/atlas/atlascerngroupdisk/perf-idtracking/GNN4ITk/GNN4Itk_v2__mc15_14TeV.600012.PhPy8EG_A14_ttbar_hdamp258p75_nonallhad.recon.RDO.e8185_s3770_s3773_r14431/GNN4Itk_v2__mc15_14TeV.600012.PhPy8EG_A14_ttbar_hdamp258p75_nonallhad.recon.RDO.e8185_s3770_s3773_r14431__J016.root
CDT23_MLM=/eos/user/b/bhuth/data/ctd23_models/best-11292882-f1_0.010190.ckpt
CDT23_FLT="/eos/user/b/bhuth/data/ctd23_models/best-11984324-auc=0.967753.ckpt"
CDT23_GNN_ML="/eos/user/b/bhuth/data/ctd23_models/best-21796495-val_loss=0.000755-epoch=91.ckpt"

# Copy
mkdir -p rel24/data
mkdir -p ctd23/data

copy_file $MMD_FILE ./rel24/data
copy_file $MMT_FILE ./rel24/data
copy_file $DUMP_FILE ./rel24/data
copy_file $CKPT ./rel24/data

copy_file $CDT23_CKPT ./ctd23/data
#copy_file $CDT23_MM ./ctd23/data
copy_file $CDT23_MMD ./ctd23/data
copy_file $CDT23_MMT ./ctd23/data
copy_file $CDT23_DATA ./ctd23/data
copy_file $CDT23_MLM ./ctd23/data
copy_file $CDT23_FLT ./ctd23/data
copy_file $CDT23_GNN_ML ./ctd23/data
