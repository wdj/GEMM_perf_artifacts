#!/bin/bash

#env CUDA_VISIBLE_DEVICES=0 ./runs.sh
#env CUDA_VISIBLE_DEVICES=1 ./runs.sh
#env CUDA_VISIBLE_DEVICES=2 ./runs.sh
#env CUDA_VISIBLE_DEVICES=3 ./runs.sh

# bsub -P stf006 -nnodes 18 -W 240 -J nodes18_gpuall ./runs.sh

#num_nodes=$(echo "$LSB_MCPU_HOSTS" | tr ' ' '\n' | grep h41 | sort -u | wc -l)

#launch_command="env OMP_NUM_THREADS=1 CUDA_DEVICE_ORDER=PCI_BUS_ID jsrun --nrs $(( 18 * 6 )) --rs_per_host 6 --cpu_per_rs 7 --bind packed:7 -g 1 --tasks_per_rs 1 -X 1 --smpiargs=\"-gpu\""

#launch_command="env OMP_NUM_THREADS=1 CUDA_DEVICE_ORDER=PCI_BUS_ID jsrun --nrs $(( $num_nodes * 6 )) --rs_per_host 6 --cpu_per_rs 7 --bind packed:7 -g 1 --tasks_per_rs 1 -X 1"

#for i in {0..7} ; do

  launch_command="env OMP_NUM_THREADS=1 CUDA_DEVICE_ORDER=PCI_BUS_ID"

  $launch_command ./wrapper.sh

#done

#wait

