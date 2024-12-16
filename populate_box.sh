#!/bin/bash

NPROC=$1
THREADS=4
export JAX_ENABLE_X64=True 
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=true intra_op_parallelism_threads=$THREADS inter_op_parallelism_threads=$THREADS"

mpirun -x JAX_ENABLE_X64 \
       -x XLA_FLAGS \
       -n $NPROC \
       --mca pml ob1 \
       --map-by node \
       --hostfile $2 \
       python ./fdmfilaments/generate_density.py ${@:3}
