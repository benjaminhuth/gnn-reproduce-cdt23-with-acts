#!/bin/bash

mkdir -p data

scp bhuth@lxplus.cern.ch:/eos/user/s/scaillou/CTD_2023/model_store/module_map/MMtriplet_1GeV_3hits_noE__merged__sorted.txt data/
scp bhuth@lxplus.cern.ch:/eos/user/p/pibutti/sw/run/gnn/Dump_GNN4Itk.root data/
