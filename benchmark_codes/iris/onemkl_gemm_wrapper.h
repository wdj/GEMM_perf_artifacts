#ifndef __ONEMKL_GEMM_WRAPPER__
#define __ONEMKL_GEMM_WRAPPER__

#include "cstdint"

// see /gpfs/jlse-fs0/users/jyoung/compilers/oneapi/2020.2.0/mkl/2021.2.0/include/oneapi

//typedef double In_t;
//typedef double Out_t;
//typedef double Scalar_t;

//typedef float In_t;
//typedef float Out_t;
//typedef float Scalar_t;

//typedef half In_t;
//typedef half Out_t;
//typedef half Scalar_t;

typedef half In_t;
typedef float Out_t;
typedef float Scalar_t;

//X typedef bfloat16 In_t;
//typedef float Out_t;
//typedef float Scalar_t;

//X typedef std::int8_t In_t;
//typedef std::int32_t Out_t;

enum {TRANSA = false};
enum {TRANSB = true};

extern "C" {
  // Run GEMM test via oneMKL
  int oneMKLGemmTest(unsigned long* nativeHandlers, In_t* A, In_t* B, Out_t* C,
     size_t m, size_t n, size_t k,
     size_t ldA, size_t ldB, size_t ldC, Scalar_t alpha, Scalar_t beta,
     bool is_using_buffers);
} // extern "C"extern "C" {

#endif // __ONEMKL_GEMM_WRAPPER__
