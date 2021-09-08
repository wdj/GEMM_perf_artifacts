// Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
// 
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
//  * Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
//  * Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimer in the
//    documentation and/or other materials provided with the distribution.
//  * Neither the name of NVIDIA CORPORATION nor the names of its
//    contributors may be used to endorse or promote products derived
//    from this software without specific prior written permission.
// 
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
// OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
// 
// For additional information on the license terms, see the CUDA EULA at
// https://docs.nvidia.com/cuda/eula/index.html

//-----------------------------------------------------------------------------

// Utilities and system includes
#include <assert.h>
#include <helper_string.h>  // helper for shared functions common to CUDA Samples

// CUDA runtime
//#include <cuda.h>
//#include <cuda_runtime.h>
//#include "device_launch_parameters.h"
//#include <cublas_v2.h>

#include "hip/hip_runtime.h"
#include "rocblas.h"
//#include "rocsparse.h"

// CUDA and CUBLAS functions
//#include <helper_functions.h>
//#include <helper_cuda.h>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <math.h>
#include <string>
#include <sstream>
#include "sys/time.h"

#define checkCudaErrors(s) (s);

const size_t g_block_size = 512;

//-----------------------------------------------------------------------------

// The following code is adapted from https://github.com/PrincetonVision/marvin/blob/master/tools/tensorIO_matlab/float2half.cpp
// The following is the relevant license:
//
// The MIT License (MIT)
// 
// Copyright (c) 2015 Princeton Vision Group
// 
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
// 
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
// 
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

typedef short __half;

__host__ __device__
__half __float2half(float f) {

    unsigned short ret;

    unsigned x = *((int*)(void*)(&f));
    unsigned u = (x & 0x7fffffff), remainder, shift, lsb, lsb_s1, lsb_m1;
    unsigned sign, exponent, mantissa;

    // Get rid of +NaN/-NaN case first.
    if (u > 0x7f800000) {
        ret = 0x7fffU;
        return ret;
    }

    sign = ((x >> 16) & 0x8000);

    // Get rid of +Inf/-Inf, +0/-0.
    if (u > 0x477fefff) {
        ret = sign | 0x7c00U;
        return ret;
    }
    if (u < 0x33000001) {
        ret = (sign | 0x0000);
        return ret;
    }

    exponent = ((u >> 23) & 0xff);
    mantissa = (u & 0x7fffff);

    if (exponent > 0x70) {
        shift = 13;
        exponent -= 0x70;
    } else {
        shift = 0x7e - exponent;
        exponent = 0;
        mantissa |= 0x800000;
    }
    lsb = (1 << shift);
    lsb_s1 = (lsb >> 1);
    lsb_m1 = (lsb - 1);

    // Round to nearest even.
    remainder = (mantissa & lsb_m1);
    mantissa >>= shift;
    if (remainder > lsb_s1 || (remainder == lsb_s1 && (mantissa & 0x1))) {
        ++mantissa;
        if (!(mantissa & 0x3ff)) {
            ++exponent;
            mantissa = 0;
        }
    }

    ret = (sign | (exponent << 10) | mantissa);

    return *(__half*)&ret;
}

//-----------------------------------------------------------------------------

typedef struct _matrixSize      // Optional Command-line multiplier for matrix sizes
{
  size_t uiWA, uiHA, uiWB, uiHB, uiWC, uiHC;
} sMatrixSize;

//! Compute reference data set matrix multiply on CPU
//! C = A * B
//! @param C          reference data, computed but preallocated
//! @param A          matrix A as provided to device
//! @param B          matrix B as provided to device
//! @param hA         height of matrix A
//! @param wB         width of matrix B

//-----------------------------------------------------------------------------
/// \brief Upper bound of random numbers from generator.

__host__ __device__
static size_t randomize_max() {

  const size_t im = 714025;
  return im;
}

//-----------------------------------------------------------------------------
/// \brief Random number genrator.

__host__ __device__
static size_t randomize(size_t i) {

  const size_t im = 714025;
  const size_t ia = 4096;
  const size_t ic = 150889;
  return (i * ia + ic) % im;
}

//-----------------------------------------------------------------------------

double walltime() {
  struct timeval tv;
  gettimeofday(&tv, NULL);
  double result = ((double)tv.tv_sec + (double)tv.tv_usec * 1.e-6);
  return result;
} 

//-----------------------------------------------------------------------------

int pci_bus_id() {
  hipDeviceProp_t deviceProp;
  // Assume only one GPU per rank.
  hipError_t error = hipGetDeviceProperties(&deviceProp, 0);
  const int pci_bus_id = error != hipSuccess ? 0 : deviceProp.pciBusID;
  return pci_bus_id;
}

//-----------------------------------------------------------------------------

int pci_domain_id() {
  hipDeviceProp_t deviceProp;
  // Assume only one GPU per rank.
  hipError_t error = hipGetDeviceProperties(&deviceProp, 0);
  const int pci_domain_id = error != hipSuccess ? 0 : deviceProp.pciDomainID;
  return pci_domain_id;
}

//-----------------------------------------------------------------------------

// Creates a random random float matrix entry.
__host__ __device__
float random_init(size_t i0, size_t i1, size_t dim0, size_t dim1, int randseed, int percent, int valscale) {

// ISSUE: i1 or i0?
  // i1 / dim1 < percent / 100
  if (i1 * 100 >= percent * dim1)
    return __float2half(0);

  const size_t i0_block = i0 % g_block_size;
  const size_t i1_block = i1 % g_block_size;

  const size_t v = i0_block / 2;
  const size_t f = i1_block;

#if 0
  const size_t uid = f + dim1 * v;
#else
  const size_t uid = (f+v+2) * (f+v+1) / 2 - f;
#endif

  size_t index = uid * randseed;
  // Randomize.
  index = randomize(index);
  index = randomize(index);
  // Calculate random number between 0 and 3.
  const float float_rand_value = index / (float)randomize_max();
  // Create 2-bit value - make extra sure less than 4.
  const int value = (int)((4. - 1e-5) * float_rand_value);

  // do this because sparse.
  const int undef = 2;

  const int veo = i0_block % 2;

  const size_t venc = veo ? (value==0 ? 0 : value==3 ? 2 : value==undef ? 0 : 1)
                          : (value==0 ? 2 : value==3 ? 0 : value==undef ? 0 : 1);

  const size_t venc_scaled = (venc << valscale) +
    (index & ((((size_t)1) << valscale) - 1));

  return (float)venc_scaled;
}

//-----------------------------------------------------------------------------

// Allocates a matrix with random float entries.
//__host__ __device__
void random_init(__half *data, size_t dim0, size_t dim1, int randseed, int percent, int valscale, bool is_b) {

  #pragma omp parallel for collapse(2)
  for (size_t i1 = 0; i1 < dim1; ++i1)
  for (size_t i0 = 0; i0 < dim0; ++i0) {
    data[i0 + dim0*i1] = __float2half(random_init(i0, i1, dim0, dim1, randseed, percent, valscale));
  }
}

//-----------------------------------------------------------------------------

// Allocates a matrix with random float entries.
__global__
void random_init_global(__half *data, size_t dim0, size_t dim1, int randseed, int percent, int valscale, bool is_b) {
  const size_t thread_0 = threadIdx.x + blockIdx.x * blockDim.x;
  const size_t thread_1 = blockIdx.y;

  const size_t i0 = thread_0;
  const size_t i1 = thread_1;

  if (i0 >= dim0 || i1 >= dim1)
    return;

  data[i0 + dim0*i1] = __float2half(random_init(i0, i1, dim0, dim1, randseed, percent, valscale));
}

//-----------------------------------------------------------------------------

struct Printer {
  double gFlopRate;
  float mSecPerMatMul;
  double flopsPerMatMul;
  size_t num_fails;
};

//-----------------------------------------------------------------------------

Printer matrixMultiply(int argc, char **argv, int devID,
  sMatrixSize &matrix_size, sMatrixSize &original_size,
  rocblas_handle &handle, int randseed, int percent, int valscale) {

  // Allocate host memory.

  const size_t size_A = matrix_size.uiWA * matrix_size.uiHA;
  const size_t mem_size_A = sizeof(__half) * size_A;
  __half *h_A = (__half *)malloc(mem_size_A);

  const size_t size_B = matrix_size.uiWB * matrix_size.uiHB;
  const size_t mem_size_B = sizeof(__half) * size_B;
  __half *h_B = (__half *)malloc(mem_size_B);

  const size_t size_C = matrix_size.uiWC * matrix_size.uiHC;
  const size_t mem_size_C = sizeof(float) * size_C;
  float *h_C = (float *) malloc(mem_size_C);

  // Initialize host memory.
  //random_init(h_A, matrix_size.uiWA, matrix_size.uiHA, randseed, percent, valscale);
  //random_init(h_B, matrix_size.uiWB, matrix_size.uiHB, randseed, percent, valscale);

  // Allocate device memory.

  __half *d_A, *d_B;
  float *d_C;

  checkCudaErrors(hipMalloc((void **) &d_A, mem_size_A));
  checkCudaErrors(hipMalloc((void **) &d_B, mem_size_B));
  checkCudaErrors(hipMalloc((void **) &d_C, mem_size_C));

  // Initialize device memory.

  //checkCudaErrors(hipMemcpy(d_A, h_A, mem_size_A, hipMemcpyHostToDevice));
  //checkCudaErrors(hipMemcpy(d_B, h_B, mem_size_B, hipMemcpyHostToDevice));

  const int threadblocksize = 256;

  random_init_global<<<dim3((matrix_size.uiWA+threadblocksize-1)/threadblocksize, matrix_size.uiHA, 1),
                       dim3(threadblocksize, 1, 1)>>>
    (d_A, matrix_size.uiWA, matrix_size.uiHA, randseed, percent, valscale, false);

  random_init_global<<<dim3((matrix_size.uiWB+threadblocksize-1)/threadblocksize, matrix_size.uiHB, 1),
                       dim3(threadblocksize, 1, 1)>>>
    (d_B, matrix_size.uiWB, matrix_size.uiHB, randseed, percent, valscale, true);

  checkCudaErrors(hipMemset((void *) d_C, 0, mem_size_C));

  // execute the kernel
  float msecPerMatMul = 0;
  double gFlopRate = 0;
  double flopsPerMatMul = 0;
  {
    const float alpha = 1.0f;
    const float beta  = 1.0f;
    hipEvent_t start, stop;

    // Allocate CUDA events that we'll use for timing
    checkCudaErrors(hipEventCreate(&start));
    checkCudaErrors(hipEventCreate(&stop));

    // Record the start event
    checkCudaErrors(hipEventRecord(start, NULL));

    //note cublas is column primary!
    //need to transpose the order
    checkCudaErrors(rocblas_gemm_ex(handle,
      //CUBLAS_OP_N,
      rocblas_operation_none,
      //CUBLAS_OP_T,
      rocblas_operation_transpose,
      matrix_size.uiWA, matrix_size.uiWB, matrix_size.uiHA,
      &alpha, d_B,
      //CUDA_R_16F,
      rocblas_datatype_f16_r,
      matrix_size.uiWB, d_A,
      //CUDA_R_16F,
      rocblas_datatype_f16_r,
      matrix_size.uiWA,
      &beta,
      d_C,
      //CUDA_R_32F,
      rocblas_datatype_f32_r,
      matrix_size.uiWB,
      d_C,
      //CUDA_R_32F,
      rocblas_datatype_f32_r,
      matrix_size.uiWB,
      //CUDA_R_32F,
      rocblas_datatype_f32_r,
      //CUBLAS_GEMM_ALGO4_TENSOR_OP));
      rocblas_gemm_algo_standard, 0, 0));

    // Record the stop event
    checkCudaErrors(hipEventRecord(stop, NULL));

    // Wait for the stop event to complete
    checkCudaErrors(hipEventSynchronize(stop));

    float msecTotal = 0.0f;
    checkCudaErrors(hipEventElapsedTime(&msecTotal, start, stop));

    // Compute and print the performance
    msecPerMatMul = msecTotal;
    flopsPerMatMul = 2.0 * (double)original_size.uiHC * (double)original_size.uiWC * (double)original_size.uiHB;
    gFlopRate = (flopsPerMatMul * 1.0e-9f) / (msecPerMatMul / 1000.0f);

    // copy result from device to host
    checkCudaErrors(hipEventDestroy(start));
    checkCudaErrors(hipEventDestroy(stop));
    checkCudaErrors(hipMemcpy(h_C, d_C, mem_size_C, hipMemcpyDeviceToHost));
  }

  // Allocate reference matrix.

  unsigned int size_R = g_block_size * g_block_size;
  unsigned int mem_size_R = sizeof(float) * size_R;
  float *h_R = (float *)malloc(mem_size_R);

  #pragma omp parallel for collapse(2)
  for (size_t i = 0; i < g_block_size; ++i) {
  for (size_t j = 0; j < g_block_size; ++j) {
    // NOTE: this test will fail if user requests percent < 100
    float sum = 0;
    for (size_t k = 0; k < g_block_size; ++k) {
      sum += (matrix_size.uiHA/g_block_size) *
        random_init(i, k, matrix_size.uiWA, matrix_size.uiHA, randseed, percent, valscale) *
        random_init(j, k, matrix_size.uiWB, matrix_size.uiHB, randseed, percent, valscale);
    }
    for (size_t k = 0; k < matrix_size.uiHA % g_block_size; ++k) {
      sum +=
        random_init(i, k, matrix_size.uiWA, matrix_size.uiHA, randseed, percent, valscale) *
        random_init(j, k, matrix_size.uiWB, matrix_size.uiHB, randseed, percent, valscale);
    }
    h_R[j + g_block_size * i] = sum;
  }
  }

  // Check accuracy of GPU GEMM.

  size_t num_fails = 0;

# pragma omp parallel for collapse(2) reduction(+:num_fails) schedule(dynamic,1000)
  for (size_t i = 0; i < matrix_size.uiWC; ++i) {
  for (size_t j = 0; j < matrix_size.uiHC; ++j) {
    num_fails += h_C[i + matrix_size.uiWC * j] !=
                 h_R[(i % g_block_size) + g_block_size * (j % g_block_size)];
  }
  }

  // clean up memory
  free(h_A);
  free(h_B);
  free(h_C);
  free(h_R);
  checkCudaErrors(hipFree(d_A));
  checkCudaErrors(hipFree(d_B));
  checkCudaErrors(hipFree(d_C));

  Printer result = {gFlopRate, msecPerMatMul, flopsPerMatMul, num_fails};
  return result;
}

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////

int main(int argc, char **argv) {

  int devID = 0;
  Printer matrix_result;
  sMatrixSize matrix_size;
  sMatrixSize original_size;
  rocblas_handle handle;

  checkCudaErrors(rocblas_create_handle(&handle));

  const int m = atoi(argv[1]);
  const int k = atoi(argv[2]);
  const int num_trials = argc > 3 ? atoi(argv[3]) : 1;
  const int randseed = argc > 4 ? atoi(argv[4]) : 1;
  const int percent = argc > 5 ? atoi(argv[5]) : 100;
  const int valscale = argc > 6 ? atoi(argv[6]) : 0;

    matrix_size.uiWA = matrix_size.uiHC = matrix_size.uiWB = matrix_size.uiWC = m;
    matrix_size.uiHB = matrix_size.uiHA = k;
    original_size = matrix_size;
    matrix_size.uiWC = matrix_size.uiHC = matrix_size.uiWB = matrix_size.uiWA = matrix_size.uiHC; // m
    matrix_size.uiHA = matrix_size.uiHB = k;
    for(int trial = 0; trial < num_trials; trial++){
      matrix_result = matrixMultiply(argc, argv, devID, matrix_size, original_size, handle, randseed, percent, valscale);
      std::cout << original_size.uiWA << " " << original_size.uiHA <<
        std::fixed <<
        std::setprecision(3) <<
        " TOps " << matrix_result.gFlopRate / 1e3 <<
        std::setprecision(6) <<
        " time " << matrix_result.mSecPerMatMul / 1e3 <<
        //" TOpcount " << matrix_result.flopsPerMatMul / 1e12 <<
        std::setprecision(0) <<
        " pci_bus_id " << pci_bus_id() << " pci_domain_id " << pci_domain_id() <<
        " randseed " << randseed << " percent_dense " << percent << " valscale " << valscale <<
        " num_fails " << matrix_result.num_fails <<
        " " << "\n" << std::flush;
    }

  checkCudaErrors(rocblas_destroy_handle(handle));

  return 0;
}

//-----------------------------------------------------------------------------

