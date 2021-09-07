
#include <iostream>
#include "string.h"

#include <cuda.h>
#include "cuda_runtime_api.h"

//===========================================================================//

void describe ( int device )
{

    cudaDeviceProp device_properties;
    ::memset( &device_properties, 0, sizeof(device_properties));

    std::cout << "***************************************"
              << "***************************************" << std::endl;

    std::cout << "Device number: " << device << std::endl;

    if ( cudaSuccess ==
          cudaGetDeviceProperties( &device_properties, device ) )
    {
        std::cout << "name: "
                  << "ASCII string identifying device:    "
                  << device_properties.name << std::endl;

        std::cout << "totalGlobalMem: "
                  << "Global memory available on device in bytes:    "
                  <<  device_properties.totalGlobalMem << std::endl;

        std::cout << "sharedMemPerBlock: "
                  << "Shared memory available per block in bytes:    "
                  <<  device_properties.sharedMemPerBlock << std::endl;

        std::cout << "regsPerBlock: "
                  << "32-bit registers available per block:    "
                  <<  device_properties.regsPerBlock << std::endl;

        std::cout << "warpSize: "
                  << "Warp size in threads:    "
                  <<  device_properties.warpSize << std::endl;

        std::cout << "memPitch: "
                  << "Maximum pitch in bytes allowed by memory copies:    "
                  <<  device_properties.memPitch << std::endl;

        std::cout << "maxThreadsPerBlock: "
                  << "Maximum number of threads per block:    "
                  <<  device_properties.maxThreadsPerBlock << std::endl;

        std::cout << "maxThreadsDim[3]: "
                  << "Maximum size of each dimension of a block:    "
                  <<  device_properties.maxThreadsDim[0] << " "
                  <<  device_properties.maxThreadsDim[1] << " "
                  <<  device_properties.maxThreadsDim[2] << std::endl;

        std::cout << "maxGridSize[3]: "
                  << "Maximum size of each dimension of a grid:    "
                  <<  device_properties.maxGridSize[0] << " "
                  <<  device_properties.maxGridSize[1] << " "
                  <<  device_properties.maxGridSize[2] << std::endl;

        std::cout << "clockRate: "
                  << "Clock frequency in kilohertz:    "
                  <<  device_properties.clockRate << std::endl;

        std::cout << "totalConstMem: "
                  << "Constant memory available on device in bytes:    "
                  <<  device_properties.totalConstMem << std::endl;

        std::cout << "major: "
                  << "Major compute capability:    "
                  <<  device_properties.major << std::endl;

        std::cout << "minor: "
                  << "Minor compute capability:    "
                  <<  device_properties.minor << std::endl;

        std::cout << "textureAlignment: "
                  << "Alignment requirement for textures:    "
                  <<  device_properties.textureAlignment << std::endl;

        std::cout << "deviceOverlap: "
                  << "Device can concurrently copy memory and execute a kernel:    "
                  <<  device_properties.deviceOverlap << std::endl;

        std::cout << "multiProcessorCount: "
                  << "Number of multiprocessors on device:    "
                  <<  device_properties.multiProcessorCount << std::endl;

        std::cout << "kernelExecTimeoutEnable: "
                  << "Specified whether there is a run time limit on kernels:    "
                  <<  device_properties.kernelExecTimeoutEnabled << std::endl;

        std::cout << "integrated: "
                  << "Device is integrated as opposed to discrete:    "
                  <<  device_properties.integrated << std::endl;

        std::cout << "canMapHostMemory: "
                  << "Device can map host memory with cudaHostAlloc/cudaHostGetDevicePointer:    "
                  <<  device_properties.canMapHostMemory << std::endl;

        std::cout << "computeMode: "
                  << "Compute mode (See ::cudaComputeMode):    "
                  <<  device_properties.computeMode << std::endl;

#define OUTPUT(NAME,DESC) \
    std::cout << #NAME << ": " << DESC << " " << device_properties.NAME << std::endl;

OUTPUT(surfaceAlignment,"the alignment requirements for surfaces.")
OUTPUT(concurrentKernels,"is 1 if the device supports executing multiple kernels within the same context  simultaneously, or 0 if not. It is not guaranteed that multiple kernels will be resident on the device concurrently so this feature should not be relied upon for correctness")
OUTPUT(ECCEnabled,"is 1 if the device has ECC support turned on, or 0 if not.")
OUTPUT(pciBusID,"the PCI bus identifier of the device")
OUTPUT(pciDeviceID,"the PCI device (sometimes called slot) identifier of the device")
OUTPUT(pciDomainID,"the PCI domain identifier of the device")
OUTPUT(tccDriver,"1 if the device is using a TCC driver or 0 if not")
OUTPUT(asyncEngineCount,"1 when the device can concurrently copy memory between host and device while executing a kernel. It is 2 when the device can concurrently copy memory between host and device in both directions and execute a kernel at the same time. It is 0 if neither of these is supported.")
OUTPUT(unifiedAddressing,"1 if the device shares a unified address space with the host and 0 otherwise")
OUTPUT(memoryClockRate,"the peak memory clock frequency in kilohertz")
OUTPUT(memoryBusWidth,"the memory bus width in bits")
OUTPUT(l2CacheSize,"L2 cache size in bytes")
OUTPUT(maxThreadsPerMultiProcessor,"the number of maximum resident threads per multiprocessor")

    }

    std::cout << "***************************************"
              << "***************************************" << std::endl;
}

//===========================================================================//

int get_count ()
{
    int num_devices = 0;

    ::cudaGetDeviceCount( &num_devices );

    return num_devices;
}

//===========================================================================//

void describe ()
{
    for ( int device=0; device < get_count(); ++device )
    {
        describe( device );
    }
}

//===========================================================================//

int main ()
{
    describe();
}

//===========================================================================//

