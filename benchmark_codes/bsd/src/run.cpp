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

////////////////////////////////////////////////////////////////////////////
//
// Copyright 1993-2015 NVIDIA Corporation.  All rights reserved.
//
// Please refer to the NVIDIA end user license agreement (EULA) associated
// with this source code for terms and conditions that govern your use of
// this software. Any use, reproduction, disclosure, or distribution of
// this software and related documentation outside the terms of the EULA
// is strictly prohibited.
//
////////////////////////////////////////////////////////////////////////////

//
// Matrix multiplication: C = A * B.
// Host code.
//
// This sample implements matrix multiplication as described in Chapter 3
// of the programming guide and uses the CUBLAS library to demonstrate
// the best performance.

// SOME PRECAUTIONS:
// IF WE WANT TO CALCULATE ROW-MAJOR MATRIX MULTIPLY C = A * B,
// WE JUST NEED CALL CUBLAS API IN A REVERSE ORDER: cublasSegemm(B, A)!
// The reason is explained as follows:

// CUBLAS library uses column-major storage, but C/C++ use row-major storage.
// When passing the matrix pointer to CUBLAS, the memory layout alters from
// row-major to column-major, which is equivalent to an implicit transpose.

// In the case of row-major C/C++ matrix A, B, and a simple matrix multiplication
// C = A * B, we can't use the input order like cublasSgemm(A, B)  because of
// implicit transpose. The actual result of cublasSegemm(A, B) is A(T) * B(T).
// If col(A(T)) != row(B(T)), equal to row(A) != col(B), A(T) and B(T) are not
// multipliable. Moreover, even if A(T) and B(T) are multipliable, the result C
// is a column-based cublas matrix, which means C(T) in C/C++, we need extra
// transpose code to convert it to a row-based C/C++ matrix.

// To solve the problem, let's consider our desired result C, a row-major matrix.
// In cublas format, it is C(T) actually (because of the implicit transpose).
// C = A * B, so C(T) = (A * B) (T) = B(T) * A(T). Cublas matrice B(T) and A(T)
// happen to be C/C++ matrice B and A (still because of the implicit transpose)!
// We don't need extra transpose code, we only need alter the input order!
//
// CUBLAS provides high-performance matrix multiplication.
// See also:
// V. Volkov and J. Demmel, "Benchmarking GPUs to tune dense linear algebra,"
// in Proc. 2008 ACM/IEEE Conf. on Supercomputing (SC '08),
// Piscataway, NJ: IEEE Press, 2008, pp. Art. 31:1-11.
//

// Utilities and system includes
#include <assert.h>
#include <helper_string.h>  // helper for shared functions common to CUDA Samples

// CUDA runtime
#include <cuda.h>
#include <cuda_runtime.h>
#include "device_launch_parameters.h"
#include <cublas_v2.h>

// CUDA and CUBLAS functions
#include <helper_functions.h>
#include <helper_cuda.h>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <math.h>
#include <string>
#include <sstream>

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
  cudaDeviceProp deviceProp;
  // Assume only one GPU per rank.
  cudaError_t error = cudaGetDeviceProperties(&deviceProp, 0);
  const int pci_bus_id = error != cudaSuccess ? 0 : deviceProp.pciBusID;
  return pci_bus_id;
}

//-----------------------------------------------------------------------------

int pci_domain_id() {
  cudaDeviceProp deviceProp;
  // Assume only one GPU per rank.
  cudaError_t error = cudaGetDeviceProperties(&deviceProp, 0);
  const int pci_domain_id = error != cudaSuccess ? 0 : deviceProp.pciDomainID;
  return pci_domain_id;
}

//-----------------------------------------------------------------------------

// Allocates a matrix with random float entries.
__host__ __device__
float random_init(size_t i0, size_t i1, size_t dim0, size_t dim1,
  int randseed, int percent, int valscale, bool is_b) {

  int result = 0;

  if (i1 * 100 < percent * dim1) {

    const size_t v = i0 / 2;
    const size_t f = i1;

    //const size_t uid = f + dim1 * v;
    const size_t uid = (f+v+2) * (f+v+1) / 2 - f;

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

    const int venc = i0 % 2 ? (value==0 ? 0 : value==3 ? 2 : value==undef ? 0 : 1)
                            : (value==0 ? 2 : value==3 ? 0 : value==undef ? 0 : 1);

    const size_t venc_scaled = (venc << valscale) +
      (index & ((((size_t)1) << valscale) - 1));

    result = venc_scaled;

  }

  return (float)result;
}

//-----------------------------------------------------------------------------

// Allocates a matrix with random float entries.
void random_init(__half *data, size_t dim0, size_t dim1, int randseed, int percent, int valscale, bool is_b)
{
  #pragma omp parallel for collapse(2)
  for (size_t i1 = 0; i1 < dim1; ++i1)
  for (size_t i0 = 0; i0 < dim0; ++i0) {
    data[i0 + dim0*i1] = __float2half(random_init(i0, i1, dim0, dim1, randseed, percent, valscale, is_b));
  }
}

//-----------------------------------------------------------------------------

// Allocates a matrix with random float entries.
__global__
void random_init_global(__half *data, size_t dim0, size_t dim1, int randseed, int percent, int valscale, bool is_b)
{
  const size_t thread_0 = threadIdx.x + blockIdx.x * blockDim.x;
  const size_t thread_1 = blockIdx.y;

  const size_t i0 = thread_0;
  const size_t i1 = thread_1;

  if (i0 >= dim0 || i1 >= dim1)
    return;

  data[i0 + dim0*i1] = __float2half(random_init(i0, i1, dim0, dim1, randseed, percent, valscale, is_b));
}

//-----------------------------------------------------------------------------

struct Printer {
  double gFlopRate;
  float mSecPerMatMul;
  double flopsPerMatMul;
};

//-----------------------------------------------------------------------------

Printer matrixMultiply(int argc, char **argv, int devID,
  sMatrixSize &matrix_size, sMatrixSize &original_size,
  cublasHandle_t &handle, int randseed, int percent, int valscale, size_t num_fails) {

  // allocate host memory for matrices A and B
  const size_t size_A = matrix_size.uiWA * matrix_size.uiHA;
  const size_t mem_size_A = sizeof(__half) * size_A;
  __half *h_A = (__half *)malloc(mem_size_A);

  const size_t size_B = matrix_size.uiWB * matrix_size.uiHB;
  const size_t mem_size_B = sizeof(__half) * size_B;
  __half *h_B = (__half *)malloc(mem_size_B);

  const size_t size_C = matrix_size.uiWC * matrix_size.uiHC;
  const size_t mem_size_C = sizeof(float) * size_C;
  float *h_C = (float *) malloc(mem_size_C);

  // allocate device memory
  __half *d_A, *d_B;
  float *d_C;

  checkCudaErrors(cudaMalloc((void **) &d_A, mem_size_A));
  checkCudaErrors(cudaMalloc((void **) &d_B, mem_size_B));
  checkCudaErrors(cudaMalloc((void **) &d_C, mem_size_C));
  memset((void *) h_C, 0, mem_size_C);

  if (false) {

    // initialize host memory
    random_init(h_A, matrix_size.uiWA, matrix_size.uiHA, randseed, percent, valscale, false);
    random_init(h_B, matrix_size.uiWB, matrix_size.uiHB, randseed, percent, valscale, true);

    // initialize device memory

    checkCudaErrors(cudaMemset((void *) d_A, 0, mem_size_A));
    checkCudaErrors(cudaMemset((void *) d_B, 0, mem_size_B));
    checkCudaErrors(cudaMemcpy(d_A, h_A, mem_size_A, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_B, h_B, mem_size_B, cudaMemcpyHostToDevice));

  } else {

    const int threadblocksize = 256;

    random_init_global<<<dim3((matrix_size.uiWA+threadblocksize-1)/threadblocksize, matrix_size.uiHA, 1),
                         dim3(threadblocksize, 1, 1)>>>
      (d_A, matrix_size.uiWA, matrix_size.uiHA, randseed, percent, valscale, false);

    random_init_global<<<dim3((matrix_size.uiWB+threadblocksize-1)/threadblocksize, matrix_size.uiHB, 1),
                         dim3(threadblocksize, 1, 1)>>>
      (d_B, matrix_size.uiWB, matrix_size.uiHB, randseed, percent, valscale, true);

  }

  checkCudaErrors(cudaMemset((void *) d_C, 0, mem_size_C));

  // execute the kernel
  float msecPerMatMul = 0;
  double gFlopRate = 0;
  double flopsPerMatMul = 0;
  {
    const float alpha = 1.0f;
    const float beta  = 1.0f;
    cudaEvent_t start, stop;

    // Allocate CUDA events for timing
    checkCudaErrors(cudaEventCreate(&start));
    checkCudaErrors(cudaEventCreate(&stop));

    // Record the start event
    checkCudaErrors(cudaEventRecord(start, NULL));

    //note cublas is column primary!
    //need to transpose the order
    checkCudaErrors(cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_T,
      matrix_size.uiWA, matrix_size.uiWB, matrix_size.uiHA,
      &alpha, d_B, CUDA_R_16F, matrix_size.uiWB, d_A, CUDA_R_16F, matrix_size.uiWA,
      &beta, d_C, CUDA_R_32F, matrix_size.uiWB, CUDA_R_32F, CUBLAS_GEMM_ALGO4_TENSOR_OP));

    // Record the stop event
    checkCudaErrors(cudaEventRecord(stop, NULL));

    // Wait for the stop event to complete
    checkCudaErrors(cudaEventSynchronize(stop));

    float msecTotal = 0.0f;
    checkCudaErrors(cudaEventElapsedTime(&msecTotal, start, stop));

    // Compute and print the performance
    msecPerMatMul = msecTotal;
    flopsPerMatMul = 2.0 * (double)original_size.uiHC * (double)original_size.uiWC * (double)original_size.uiHB;
    gFlopRate = (flopsPerMatMul * 1.0e-9f) / (msecPerMatMul / 1000.0f);

    checkCudaErrors(cudaEventDestroy(start));
    checkCudaErrors(cudaEventDestroy(stop));

    // copy result from device to host
    checkCudaErrors(cudaMemcpy(h_C, d_C, mem_size_C, cudaMemcpyDeviceToHost));

    // check result

    num_fails = 0;

    for (size_t i : {(size_t)0, (matrix_size.uiWA-1)/2, matrix_size.uiWA-1})
    for (size_t j : {(size_t)0, (matrix_size.uiWB-1)/2, matrix_size.uiWB-1}) {
      float sum = 0;
      for (size_t k = 0; k < matrix_size.uiHA; ++k)
        sum +=
          random_init(i, k, matrix_size.uiWA, matrix_size.uiHA, randseed, percent, valscale, false) *
          random_init(j, k, matrix_size.uiWB, matrix_size.uiHB, randseed, percent, valscale, true);
        //printf("%.20f %.20f\n", h_C[i + matrix_size.uiWC * j], sum);
        num_fails += h_C[i + matrix_size.uiWC * j] != sum;
    }
    //printf("%zu\n", num_fails);
  }

  // clean up memory
  free(h_A);
  free(h_B);
  free(h_C);
  //free(reference);
  checkCudaErrors(cudaFree(d_A));
  checkCudaErrors(cudaFree(d_B));
  checkCudaErrors(cudaFree(d_C));

  Printer result = {gFlopRate, msecPerMatMul, flopsPerMatMul};
  return result;
}

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char **argv)
{
  int devID = 0; //, sizeMult = 5;
  Printer matrix_result;
  sMatrixSize matrix_size;
  sMatrixSize original_size;
//  std::ofstream myfile;
  cublasHandle_t handle;
  checkCudaErrors(cublasCreate(&handle));
  const int m = atoi(argv[1]);
  const int k = atoi(argv[2]);
  const int num_trials = argc > 3 ? atoi(argv[3]) : 1;
  const int randseed = argc > 4 ? atoi(argv[4]) : 1;
  const int percent = argc > 5 ? atoi(argv[5]) : 100;
  const int valscale = argc > 6 ? atoi(argv[6]) : 0;
  size_t num_fails = 0;
    matrix_size.uiWA = matrix_size.uiHC = matrix_size.uiWB = matrix_size.uiWC = m;
    matrix_size.uiHB = matrix_size.uiHA = k;
    original_size = matrix_size;
    matrix_size.uiWC = matrix_size.uiHC = matrix_size.uiWB = matrix_size.uiWA = matrix_size.uiHC; // m
    matrix_size.uiHA = matrix_size.uiHB = k;
    for(int trial = 0; trial < num_trials; trial++){
      matrix_result = matrixMultiply(argc, argv, devID, matrix_size, original_size, handle, randseed, percent, valscale, num_fails);
      //std::cout << original_size.uiWA << "," << original_size.uiHA << "," << original_size.uiHA * original_size.uiWA << "," << trial << "," << matrix_result.gFlopRate << "," << matrix_result.mSecPerMatMul << "," << matrix_result.flopsPerMatMul << "," << "\n";
      std::cout << original_size.uiWA << " " << original_size.uiHA <<
        std::fixed <<
        std::setprecision(3) <<
        " TOps " << matrix_result.gFlopRate / 1e3 <<
        std::setprecision(6) <<
        " time " << matrix_result.mSecPerMatMul / 1e3 <<
        //" TOpcount " << matrix_result.flopsPerMatMul / 1e12 <<
        std::setprecision(0) <<
        " pci_bus_id " << pci_bus_id() << " pci_domain_id " << pci_domain_id() <<
        " randseed " << randseed << " percent_dense " << percent <<
        " valscale " << valscale << " num_fails " << num_fails <<
        " " << "\n" << std::flush;
      //std::cout << m << std::endl;
    //} // for trial
  }
//  myfile.close();
  checkCudaErrors(cublasDestroy(handle));
  return 0;
}

//-----------------------------------------------------------------------------

