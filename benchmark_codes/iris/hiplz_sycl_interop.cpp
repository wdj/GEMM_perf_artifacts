//-----------------------------------------------------------------------------

#include "cstdlib"
#include <exception>
#include <iostream>
#include <vector>

#include "hip/hip_runtime.h"

#include "onemkl_gemm_wrapper.h"

using namespace std;

//-----------------------------------------------------------------------------

// see https://github.com/PrincetonVision/marvin/blob/master/tools/tensorIO_matlab/float2half.cpp

template<typename I_t, typename O_t>
struct Conv {
  static I_t o2i(O_t x) {return (I_t)x;}
  static O_t i2o(I_t x) {return (O_t)x;}
};

template<>
struct Conv<half, float> {
  typedef half I_t;
  typedef float O_t;

  static O_t i2o(I_t x2) {

    const unsigned short x = *(unsigned short*)&x2;

    unsigned sign = ((x >> 15) & 1);
    unsigned exponent = ((x >> 10) & 0x1f);
    unsigned mantissa = ((x & 0x3ff) << 13);
    if (exponent == 0x1f) {  /* NaN or Inf */
        mantissa = (mantissa ? (sign = 0, 0x7fffff) : 0);
        exponent = 0xff;
    } else if (!exponent) {  /* Denorm or Zero */
        if (mantissa) {
            unsigned int msb;
            exponent = 0x71;
            do {
                msb = (mantissa & 0x400000);
                mantissa <<= 1;  /* normalize */
                --exponent;
            } while (!msb);
            mantissa &= 0x7fffff;  /* 1.mantissa is implicit */
        }
    } else {
        exponent += 0x70;
    }
    int temp = ((sign << 31) | (exponent << 23) | mantissa);

    return *((float*)((void*)&temp));

  }

  static I_t o2i(O_t f) {
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

    return *(half*)&ret;
  }
};

//-----------------------------------------------------------------------------

void VerifyResult(size_t dim0, size_t dim1, vector<Out_t>& M, size_t ldM,
  vector<Out_t>& M_ref, size_t ldM_ref) {
  bool MismatchFound = false;

  for (size_t i=0; i < dim1; i++) {
    for (size_t j=0; j < dim0; j++) {

      const bool value_same = fabs(Conv<Out_t, float>::i2o(M[j+ldM*i]) -
                                   Conv<Out_t, float>::i2o(M_ref[j+ldM_ref*i])) < 1e-8;

      if (!value_same) {
        std::cout << "fail - incorrect element: [" << i << ", " << j <<
          "], expected: " << Conv<Out_t, float>::i2o(M_ref[j+ldM_ref*i]) <<
          " , but got: " << Conv<Out_t, float>::i2o(M[j+ldM*i]) << std::endl;
        MismatchFound = true;
      }
    }
  }

  if (!MismatchFound) {
    std::cout << "SUCCESS - The results are correct!" << std::endl;
    return;
  }
}

//-----------------------------------------------------------------------------

int main(int argc, char** argv) {

  //const int WIDTH = 32000; // FP16/FP32 32000 32000 32000 TOps 0.994601478 time 65.891718
  //const int WIDTH = 32000; // FP32 32000 32000 32000 TOps 1.103913430 time 59.366974
  //const int WIDTH = 22000; // FP64 22000 22000 22000 TOps 0.275101429 time 77.411448
  const int WIDTH = 22000;

  const bool have_args = argc == 4;

  const size_t m = have_args ? atoi(argv[1]) : WIDTH;
  const size_t k = have_args ? atoi(argv[2]) : WIDTH;
  const int num_trials = have_args ? atoi(argv[3]) : 1;

  const size_t n = m;
  const size_t ldA = TRANSA ? k : m;
  const size_t ldB = TRANSB ? n : k;
  const size_t ldC = m;
  const size_t dim_A = ldA*(TRANSA ? m : k);
  const size_t dim_B = ldB*(TRANSB ? k : n);
  const size_t dim_C = ldC*n;
  vector<In_t> A(dim_A, 0);
  vector<In_t> B(dim_B, 0);
  vector<Out_t> C(dim_C, 0);
  vector<Out_t> C_ref(dim_C, 0);
  const Scalar_t alpha = Conv<Scalar_t, float>::o2i(1);//1;
  const Scalar_t beta = Conv<Scalar_t, float>::o2i(1);//0;

  const bool is_using_buffers = false;
  const bool do_verify = 2 * m * n * k < 1000000;

  // initialize data on the host
  // prepare matrix data with ROW-major style
  // A(M, N)
  for (size_t i=0; i<k; i++)
    for (size_t j=0; j<m; j++)
      A[TRANSA ? i + ldA*j : j + ldA*i] = Conv<In_t, float>::o2i(1 + (j + m*i) % 4);
      //A[j + ldA*i] = Conv<In_t, Out_t>::o2i(0);

  // B(N, P)
  for (size_t i=0; i<n; i++)
    for (size_t j=0; j<k; j++)
      B[TRANSB ? i + ldB*j : j + ldB*i] = Conv<In_t, float>::o2i(1 + (j + k*i) % 4);
      //B[j + ldB*i] = Conv<In_t, Out_t>::o2i(0);

  // get CPU result for verification
  // Resultant matrix: C_ref = A*B
  if (do_verify) {
    for (size_t i=0; i<n; i++) {
      for (size_t j=0; j<m; j++) {
        C[j + ldC*i] = 0;
        C_ref[j + ldC*i] = C[j + ldC*i];
        for(size_t j2=0; j2<k; j2++) {
          C_ref[j + ldC*i] =
            Conv<Out_t,    float>::o2i(
            Conv<Out_t,    float>::i2o(C_ref[j + ldC*i]) +
            Conv<Scalar_t, float>::i2o(alpha) *
            Conv<In_t,     float>::i2o(A[TRANSA ? j2 + ldA*j : j + ldA*j2]) *
            Conv<In_t,     float>::i2o(B[TRANSB ? i + ldB*j2 : j2 + ldB*i]));
        }
      }
    }
  }

  //for (int i=0; i<dim_A; ++i) printf("%f ", Conv<In_t, float>::i2o(A[i])); printf("\n");
  //for (int i=0; i<dim_B; ++i) printf("%f ", Conv<In_t, float>::i2o(B[i])); printf("\n");
  //for (int i=0; i<dim_C; ++i) printf("%f ", Conv<Out_t, float>::i2o(C_ref[i])); printf("\n");

  In_t* d_A = nullptr;
  In_t* d_B = nullptr;
  Out_t* d_C = nullptr;

  for (int trial=0; trial<num_trials; ++trial) {

    if (!is_using_buffers) {
      hipMalloc(&d_A, dim_A*sizeof(*d_A));
      hipMalloc(&d_B, dim_B*sizeof(*d_B));
      hipMalloc(&d_C, dim_C*sizeof(*d_C));
      hipMemcpy(d_A, A.data(), dim_A*sizeof(*d_A), hipMemcpyHostToDevice);
      hipMemcpy(d_B, B.data(), dim_B*sizeof(*d_B), hipMemcpyHostToDevice);
      hipMemcpy(d_C, C.data(), dim_C*sizeof(*d_C), hipMemcpyHostToDevice);
    }

    // Create HipLZ stream
    hipStream_t stream = nullptr;
    hipStreamCreate(&stream);

    unsigned long nativeHandlers[4];
    int numItems = 0;
    hiplzStreamNativeInfo(stream, nativeHandlers, &numItems);

    // Invoke oneMKL GEMM

    //for (int i=0; i<num_trials; ++i)
    oneMKLGemmTest(nativeHandlers,
      is_using_buffers ? A.data() : d_A,
      is_using_buffers ? B.data() : d_B,
      is_using_buffers ? C.data() : d_C,
      m, n, k, ldA, ldB, ldC, alpha, beta, is_using_buffers);

    if (!is_using_buffers) {
      hipMemcpy(C.data(), d_C, dim_C*sizeof(*d_C), hipMemcpyDeviceToHost);
    }

    //for (int i=0; i<dim_C; ++i) printf("%f ", Conv<Out_t, float>::i2o(C[i])); printf("\n");

    // check results
    if (do_verify && 1 == num_trials) {
      std::cout << "Verify results between OneMKL & Serial: ";
      VerifyResult(m, n, C, ldC, C_ref, ldC);
    }

    if (!is_using_buffers) {
      hipFree(d_A);
      hipFree(d_B);
      hipFree(d_C);
    }

    hipStreamDestroy(stream);

  } // trial

  return 0;
}

//-----------------------------------------------------------------------------
