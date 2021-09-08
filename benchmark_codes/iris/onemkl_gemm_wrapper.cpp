//-----------------------------------------------------------------------------

#include <stdlib.h>
#include <vector>
#include <string.h>
#include <stdio.h>

#include "sys/time.h"

#include "level_zero/ze_api.h"
#include "CL/sycl/backend/level_zero.hpp"
#include "oneapi/mkl.hpp"

#include "onemkl_gemm_wrapper.h"

using namespace std;

//-----------------------------------------------------------------------------

double walltime() {

  struct timeval tv;
  gettimeofday(&tv, NULL);
  double result = ((double)tv.tv_sec + (double)tv.tv_usec * 1.e-6);

  return result;
}

//-----------------------------------------------------------------------------

int onemkl_gemm(sycl::queue& my_queue, In_t* A, In_t* B, Out_t* C,
  size_t m, size_t n, size_t k, size_t ldA, size_t ldB, size_t ldC, Scalar_t alpha, Scalar_t beta,
  bool is_using_buffers) {

  auto my_exception_handler = [](sycl::exception_list exceptions) {
    for (std::exception_ptr const& e : exceptions) {
      try {
        std::rethrow_exception(e);
      } catch (sycl::exception const& e) {
        std::cout << "Caught asynchronous SYCL exception:\n"
                  << e.what() << std::endl;
      } catch (std::exception const& e) {
        std::cout << "Caught asynchronous STL exception:\n"
                  << e.what() << std::endl;
      }
    }
  };

  const double t1 = walltime();

  if (is_using_buffers) {

    // create sycl buffers of matrix data for offloading between device and host 
    sycl::buffer<In_t, 1> A_buf(A, ldA*(TRANSA ? m : k));
    sycl::buffer<In_t, 1> B_buf(B, ldB*(TRANSB ? k : n));
    sycl::buffer<Out_t, 1> C_buf(C, ldC*n);

    // add oneapi::mkl::blas::gemm to execution queue and catch any synchronous exceptions  
    try {
      oneapi::mkl::blas::column_major::gemm(my_queue,
        TRANSA ? oneapi::mkl::transpose::trans : oneapi::mkl::transpose::nontrans,
        TRANSB ? oneapi::mkl::transpose::trans : oneapi::mkl::transpose::nontrans,
        m, n, k, alpha, A_buf, ldA, B_buf, ldB, beta, C_buf, ldC);
    } catch (sycl::exception const& e) {
      std::cout << "\t\tCaught synchronous SYCL exception during GEMM:\n"
                << e.what() << std::endl;
    } catch (std::exception const& e) {
      std::cout << "\t\tCaught synchronous STL exception during GEMM:\n"
                << e.what() << std::endl;
    }

  } else { 

    // add oneapi::mkl::blas::gemm to execution queue and catch any synchronous exceptions  
    try {
      oneapi::mkl::blas::column_major::gemm(my_queue,
        TRANSA ? oneapi::mkl::transpose::trans : oneapi::mkl::transpose::nontrans,
        TRANSB ? oneapi::mkl::transpose::trans : oneapi::mkl::transpose::nontrans,
        m, n, k, alpha, A, ldA, B, ldB, beta, C, ldC);
    } catch (sycl::exception const& e) {
      std::cout << "\t\tCaught synchronous SYCL exception during GEMM:\n"
                << e.what() << std::endl;
    } catch (std::exception const& e) {
      std::cout << "\t\tCaught synchronous STL exception during GEMM:\n"
                << e.what() << std::endl;
    }

  }
 
  // ensure any asynchronous exceptions caught are handled before proceeding 
  my_queue.wait_and_throw();

  const double t2 = walltime();

  printf("%zu %zu %zu TOps %.9f time %.6f  scalar sizes %zu %zu %zu  %s %s\n",
    m, n, k, 2.*m*n*k/(t2-t1)/1e12, t2-t1,
    sizeof(In_t), sizeof(Out_t), sizeof(Scalar_t),
    TRANSA ? "T" : "N", TRANSB ? "T" : "N");

  return 0;
}

//-----------------------------------------------------------------------------

// Run GEMM test via oneMKL  
extern "C" {

int oneMKLGemmTest(unsigned long* nativeHandlers, In_t* A, In_t* B, Out_t* C,
   size_t m, size_t n, size_t k, size_t ldA, size_t ldB, size_t ldC, Scalar_t alpha, Scalar_t beta,
   bool is_using_buffers) {
  // Extract the native information
  ze_driver_handle_t        hDriver  = (ze_driver_handle_t)nativeHandlers[0];
  ze_device_handle_t        hDevice  = (ze_device_handle_t)nativeHandlers[1];
  ze_context_handle_t       hContext = (ze_context_handle_t)nativeHandlers[2];
  ze_command_queue_handle_t hQueue   = (ze_command_queue_handle_t)nativeHandlers[3];
    
  sycl::platform sycl_platform = sycl::level_zero::make<sycl::platform>(hDriver);
 
  // make devices from converted platform and L0 device 
  sycl::device sycl_device =
    sycl::level_zero::make<sycl::device>(sycl_platform, hDevice);
  std::vector<sycl::device> devices;
  devices.push_back(sycl_device);
  sycl::context sycl_context =
    sycl::level_zero::make<sycl::context>(devices, hContext);
  sycl::queue queue = sycl::level_zero::make<sycl::queue>(sycl_context, hQueue);

  // Test the oneMKL
  const int result = onemkl_gemm(queue, A, B, C, m, n, k, ldA, ldB, ldC, alpha, beta,
    is_using_buffers);
  return result;
}

} // extern "C"

//-----------------------------------------------------------------------------
