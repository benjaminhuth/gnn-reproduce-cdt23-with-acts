#!/bin/bash


mkdir -p build
cd build

rm -rf *

cmake .. -G Ninja \
  -D CMAKE_PREFIX_PATH=/root/.miniconda3/envs/acorn-experimental/lib/python3.10/site-packages/torch \
  -D CMAKE_CUDA_ARCHITECTURES=80 \
  -D CMAKE_CUDA_COMPILER=/root/software/cuda-12.1/bin/nvcc
ninja -j1
