#!/bin/bash

source setup_acorn.sh

#infer rel24/config/data_reader.yaml

infer rel24/config/module_map.yaml
#EVAL ctd23/config/module_map_eval.yaml

#INFER ctd23/config/gnn_infer.yaml
#EVAL ctd23/config/gnn_eval.yaml
