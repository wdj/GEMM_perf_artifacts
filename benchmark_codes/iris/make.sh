#!/bin/bash

# Exit immediately on error.
set -eu -o pipefail

# Setup Intel runtime:  
module use /soft/modulefiles/
module load -s intel_compute_runtime cmake

# Setup the DPC++ environment
module use /home/jyoung/gpfs_share/compilers/modulefiles/oneapi/2020.2.0.2997/
module load -s mkl compiler

# Set the HIPLZ_INSTALL_PREFIX as HipLZ installation since the dynamic shared library that encapsulates HIP matrix muplication was pre-built and installed at ${HIPLZ_INSTALL_PREFIX}/lib
export HIPLZ_INSTALL_PREFIX=/home/jzhao/hipclworkspace/hipcl/install/

# Build libonemkl_gemm_wrapper.so
clang++ onemkl_gemm_wrapper.cpp -DMKL_ILP64 -lmkl_sycl -lmkl_intel_ilp64 -lmkl_sequential -lmkl_core -o libonemkl_gemm_wrapper.so -fsycl -lze_loader -shared -fPIC

# Build the HIPLZ part for invoking SyCL based library
# Setup the environment based on JLSE
export HIPCL_CLANG_PREFIX=/soft/compilers/clang-hipcl/8.0-20210506
export HIPLZ_BUILD=/home/jzhao/hipclworkspace/hipcl/build
export HIPLZ_INCLUDE=/home/jzhao/hipclworkspace/hipcl/include
export OPENCL_INSTALLATION=/soft/libraries/pocl/OpenCL-ICD-Loader/build-v2020.06.16

# Build hiplz_sycl_interop.o
${HIPCL_CLANG_PREFIX}/bin/clang++ -I${HIPLZ_INCLUDE} -g -fPIE -x hip --hip-device-lib-path=${HIPLZ_BUILD} --hip-device-lib=kernellib.bc --hip-llvm-pass-path=${HIPLZ_BUILD}/llvm_passes -pthread -std=c++14 -c hiplz_sycl_interop.cpp -o hiplz_sycl_interop.o

# Link
${HIPCL_CLANG_PREFIX}/bin/clang++ -g hiplz_sycl_interop.o -o hiplz_sycl_interop.exe -Wl,-rpath,${HIPLZ_INSTALL_PREFIX}/lib:${OPENCL_INSTALLATION}: ${HIPLZ_INSTALL_PREFIX}lib/libhiplz.so.0.9.0 libonemkl_gemm_wrapper.so -lOpenCL -lze_loader -pthread ${OPENCL_INSTALLATION}/libOpenCL.so.1.2

