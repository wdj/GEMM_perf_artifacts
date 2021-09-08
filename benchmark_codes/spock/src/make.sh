#!/bin/bash

module load rocm/4.2.0

hipcc \
-I ../includes \
-g -Wall -Wno-unused-function -fno-associative-math -Wno-error=unknown-pragmas -DTEST_PROCS_MAX=64 -Wno-constant-logical-operand \
-I/sw/spock/spack-envs/views/rocm-4.2.0/include \
-I/sw/spock/spack-envs/views/rocm-4.2.0/hip/include/hip \
-fno-gpu-rdc \
-Wno-unused-command-line-argument \
--amdgpu-target=gfx906,gfx908 \
-Wno-c99-designator \
-Wno-duplicate-decl-specifier \
-Wno-unused-variable \
 -O3 \
-fomit-frame-pointer \
-fno-math-errno \
-ffinite-math-only \
-fno-signed-zeros \
-fno-trapping-math \
-freciprocal-math \
-finline-functions \
-ftrapv \
-rdynamic \
run.cpp \
-o run \
-L/sw/spock/spack-envs/views/rocm-4.2.0/lib \
-lrocblas \
-lrocsparse \

