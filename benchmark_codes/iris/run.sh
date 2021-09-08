#!/bin/bash

# Setup Intel runtime:  
module use /soft/modulefiles/
module load -s intel_compute_runtime cmake

# Setup the DPC++ environment:
module use /home/jyoung/gpfs_share/compilers/modulefiles/oneapi/2020.2.0.2997/
module load -s mkl compiler

# run test
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:$PWD

./hiplz_sycl_interop.exe $@

