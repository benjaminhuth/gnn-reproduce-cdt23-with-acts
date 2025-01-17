#!/bin/bash

MAX_TRT=5
MAX_JOBS=8

DIR=benchmarks/$(date -Iminutes | cut -c-16)
mkdir -p $DIR

for ((JOBS=1; JOBS<=$MAX_JOBS; JOBS++)); do
  TRT=$(($JOBS<$MAX_TRT ? $JOBS : $MAX_TRT))
  echo "jobs: $JOBS, trt contexts: $TRT"
  ./run_rel24_acts.sh -n100 -j$JOBS --tensorrt-exec-contexts=$TRT --module-map-dynamic-alloc --timing-mode | tee $DIR/n100j${JOBS}c${TRT}.log
done
