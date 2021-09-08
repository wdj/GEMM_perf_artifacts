#!/bin/bash

# sbatch -N36 -A stf006 -t 30 runs.sh

launch_command="env OMP_NUM_THREADS=16 srun -n $(( 4 * $SLURM_NNODES )) --cpus-per-task=16 --ntasks-per-node=4 --gpus-per-task=1 --gpu-bind=closest --wait=0"

$launch_command ./wrapper.sh

