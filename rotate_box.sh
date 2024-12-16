#!/bin/bash

NPROC=$1
THREADS=4
export JAX_ENABLE_X64=True 
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=true intra_op_parallelism_threads=$THREADS inter_op_parallelism_threads=$THREADS"
cd fdmfilaments
mpirun -x JAX_ENABLE_X64 \
       -x XLA_FLAGS \
       -n $NPROC \
       python rotate_box.py ${@:2}
cd ..
