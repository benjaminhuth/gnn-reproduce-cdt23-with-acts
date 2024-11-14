#!/bin/bash

MIN_NODES=1
MIN_EDGES=1

MAX_NODES=1000000
MAX_EDGES=3000000

NODES=400000
EDGES=800000


python3 -c "import numpy as np; np.random.uniform(0,1,($NODES,12)).tofile('nodes.npz')"
python3 -c "import numpy as np; np.random.uniform(0,1,($EDGES,6)).tofile('edge_attr.npz')"
python3 -c "import numpy as np; np.random.randint(0,$NODES,(2,$EDGES)).tofile('edge_index.npz')"

#  --best \
trtexec \
  --directIO \
  --onnx=$1.onnx \
  --saveEngine=$1_best.engine \
  --minShapes=x:${MIN_NODES}x12,edge_attr:${MIN_EDGES}x6,edge_index:2x${MIN_EDGES} \
  --maxShapes=x:${MAX_NODES}x12,edge_attr:${MAX_EDGES}x6,edge_index:2x${MAX_EDGES} \
  --optShapes=x:${NODES}x12,edge_attr:${EDGES}x6,edge_index:2x${EDGES} \
  --shapes=x:${NODES}x12,edge_attr:${EDGES}x6,edge_index:2x${EDGES} \
  --loadInputs='x':'nodes.npz','edge_attr':'edge_attr.npz','edge_index':'edge_index.npz'

rm -f nodes.npz edge_attr.npz edge_index.npz edge_attr.npz
