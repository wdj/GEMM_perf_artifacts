
#include <iostream>
#include "string.h"

#include "hip/hip_runtime_api.h"

//===========================================================================//

void describe ( int device )
{

    hipDeviceProp_t device_properties;
    ::memset( &device_properties, 0, sizeof(device_properties));

    std::cout << "***************************************"
              << "***************************************" << std::endl;

    std::cout << "Device number: " << device << std::endl;

    if ( hipSuccess ==
          hipGetDeviceProperties( &device_properties, device ) )
    {


    std::cout << device_properties.
    name
    << " name"
        << " Device name\n";
    std::cout << device_properties.
    totalGlobalMem
    << " totalGlobalMem"
        << " Size of global memory region (in bytes)\n";
    std::cout << device_properties.
    sharedMemPerBlock
    << " sharedMemPerBlock"
        << " Size of shared memory region (in bytes)\n";
    std::cout << device_properties.
    regsPerBlock
    << " regsPerBlock"
        << " Registers per block\n";
    std::cout << device_properties.
    warpSize
    << " warpSize"
        << " Warp size\n";
    std::cout << device_properties.
    maxThreadsPerBlock
    << " maxThreadsPerBlock"
        << " Max work items per work group or workgroup max size\n";
    std::cout << device_properties.
    maxThreadsDim[0]
    << " maxThreadsDim[0]"
    << device_properties.
    maxThreadsDim[1]
    << " maxThreadsDim[1]"
    << device_properties.
    maxThreadsDim[2]
    << " maxThreadsDim[2]"
        << " Max number of threads in each dimension (XYZ) of a block\n";
    std::cout << device_properties.
    maxGridSize[0]
    << " maxGridSize[0]"
    << device_properties.
    maxGridSize[1]
    << " maxGridSize[1]"
    << device_properties.
    maxGridSize[2]
    << " maxGridSize[2]"
        << " Max grid dimensions (XYZ)\n";
    std::cout << device_properties.
    clockRate
    << " clockRate"
        << " Max clock frequency of the multiProcessors in khz\n";
    std::cout << device_properties.
    memoryClockRate
    << " memoryClockRate"
        << " Max global memory clock frequency in khz\n";
    std::cout << device_properties.
    memoryBusWidth
    << " memoryBusWidth"
        << " Global memory bus width in bits\n";
    std::cout << device_properties.
    totalConstMem
    << " totalConstMem"
        << " Size of shared memory region (in bytes)\n";
    std::cout << device_properties.
    major
    << " major"
        << " Major compute capability.  On HCC, this is an approximation and features may differ from CUDA CC.  See the arch feature flags for portable ways to query feature caps\n";
    std::cout << device_properties.
    minor
    << " minor"
        << " Minor compute capability.  On HCC, this is an approximation and features may differ from CUDA CC.  See the arch feature flags for portable ways to query feature caps\n";
    std::cout << device_properties.
    multiProcessorCount
    << " multiProcessorCount"
        << " Number of multi-processors (compute units)\n";
    std::cout << device_properties.
    l2CacheSize
    << " l2CacheSize"
        << " L2 cache size\n";
    std::cout << device_properties.
    maxThreadsPerMultiProcessor
    << " maxThreadsPerMultiProcessor"
        << " Maximum resident threads per multi-processor\n";
    std::cout << device_properties.
    computeMode
    << " computeMode"
        << " Compute mode\n";
    std::cout << device_properties.
    clockInstructionRate
    << " clockInstructionRate"
        << " Frequency in khz of the timer used by the device-side \"clock*\" instructions.  New for HIP\n";
    //std::cout << device_properties.
    //arch
    //<< " arch"
    //    << " Architectural feature flags.  New for HIP\n";
    std::cout << device_properties.
    concurrentKernels
    << " concurrentKernels"
        << " Device can possibly execute multiple kernels concurrently\n";
    std::cout << device_properties.
    pciDomainID
    << " pciDomainID"
        << " PCI Domain ID\n";
    std::cout << device_properties.
    pciBusID
    << " pciBusID"
        << " PCI Bus ID\n";
    std::cout << device_properties.
    pciDeviceID
    << " pciDeviceID"
        << " PCI Device ID\n";
    std::cout << device_properties.
    maxSharedMemoryPerMultiProcessor
    << " maxSharedMemoryPerMultiProcessor"
        << " Maximum Shared Memory Per Multiprocessor\n";
    std::cout << device_properties.
    isMultiGpuBoard
    << " isMultiGpuBoard"
        << " 1 if device is on a multi-GPU board, 0 if not\n";
    std::cout << device_properties.
    canMapHostMemory
    << " canMapHostMemory"
        << " Check whether HIP can map host memory\n";
    std::cout << device_properties.
    gcnArch
    << " gcnArch"
        << " DEPRECATED: use gcnArchName instead\n";
    std::cout << device_properties.
    gcnArchName
    << " gcnArchNam"
        << " AMD GCN Arch Name\n";
    std::cout << device_properties.
    integrated
    << " integrated"
        << " APU vs dGPU\n";
    std::cout << device_properties.
    cooperativeLaunch
    << " cooperativeLaunch"
        << " HIP device supports cooperative launch\n";
    std::cout << device_properties.
    cooperativeMultiDeviceLaunch
    << " cooperativeMultiDeviceLaunch"
        << " HIP device supports cooperative launch on multiple devices\n";
    std::cout << device_properties.
    maxTexture1DLinear
    << " maxTexture1DLinear"
        << " Maximum size for 1D textures bound to linear memory\n";
    std::cout << device_properties.
    maxTexture1D
    << " maxTexture1D"
        << " Maximum number of elements in 1D images\n";
    std::cout << device_properties.
    maxTexture2D[0]
    << " maxTexture2D[0]"
    << device_properties.
    maxTexture2D[1]
    << " maxTexture2D[1]"
        << " Maximum dimensions (width, height) of 2D images, in image elements\n";
    std::cout << device_properties.
    maxTexture3D[0]
    << " maxTexture3D[0]"
    << device_properties.
    maxTexture3D[1]
    << " maxTexture3D[1]"
    << device_properties.
    maxTexture3D[2]
    << " maxTexture3D[2]"
        << " Maximum dimensions (width, height, depth) of 3D images, in image elements\n";
    //unsigned int* hdpMemFlushCntl
    //    << " Addres of HDP_MEM_COHERENCY_FLUSH_CNTL register\n";
    //unsigned int* hdpRegFlushCntl
    //    << " Addres of HDP_REG_COHERENCY_FLUSH_CNTL register\n";
    std::cout << device_properties.
    memPitch
    << " memPitch"
        << " Maximum pitch in bytes allowed by memory copies\n";
    std::cout << device_properties.
    textureAlignment
    << " textureAlignment"
        << " Alignment requirement for textures\n";
    std::cout << device_properties.
    texturePitchAlignment
    << " texturePitchAlignment"
        << " Pitch alignment requirement for texture references bound to pitched memory\n";
    std::cout << device_properties.
    kernelExecTimeoutEnabled
    << " kernelExecTimeoutEnabled"
        << " Run time limit for kernels executed on the device\n";
    std::cout << device_properties.
    ECCEnabled
    << " ECCEnabled"
        << " Device has ECC support enabled\n";
    std::cout << device_properties.
    tccDriver
    << " tccDriver"
        << " 1:If device is Tesla device using TCC driver, else 0\n";
    std::cout << device_properties.
    cooperativeMultiDeviceUnmatchedFunc
    << " cooperativeMultiDeviceUnmatchedFunc"
        << " HIP device supports cooperative launch on multiple devices with unmatched functions\n";
    std::cout << device_properties.
    cooperativeMultiDeviceUnmatchedGridDim
    << " cooperativeMultiDeviceUnmatchedGridDim"
        << " HIP device supports cooperative launch on multiple devices with unmatched grid dimensions\n";
    std::cout << device_properties.
    cooperativeMultiDeviceUnmatchedBlockDim
    << " cooperativeMultiDeviceUnmatchedBlockDim"
        << " HIP device supports cooperative launch on multiple devices with unmatched block dimensions\n";
    std::cout << device_properties.
    cooperativeMultiDeviceUnmatchedSharedMem
    << " cooperativeMultiDeviceUnmatchedSharedMem"
        << " HIP device supports cooperative launch on multiple devices with unmatched shared memories\n";
    std::cout << device_properties.
    isLargeBar
    << " isLargeBar"
        << " 1: if it is a large PCI bar device, else 0\n";
    std::cout << device_properties.
    asicRevision
    << " asicRevision"
        << " Revision of the GPU in this device\n";
    std::cout << device_properties.
    managedMemory
    << " managedMemory"
        << " Device supports allocating managed memory on this system\n";
    std::cout << device_properties.
    directManagedMemAccessFromHost
    << " directManagedMemAccessFromHost"
        << " Host can directly access managed memory on the device without migration\n";
    std::cout << device_properties.
    concurrentManagedAccess
    << " concurrentManagedAccess"
        << " Device can coherently access managed memory concurrently with the CPU\n";
    std::cout << device_properties.
    pageableMemoryAccess
    << " pageableMemoryAccess"
        << " Device supports coherently accessing pageable memory without calling hipHostRegister on it\n";
    std::cout << device_properties.
    pageableMemoryAccessUsesHostPageTables
    << " pageableMemoryAccessUsesHostPageTables"
        << " Device accesses pageable memory via the host's page tables\n";

    }

    std::cout << "***************************************"
              << "***************************************" << std::endl;
}

//===========================================================================//

int get_count ()
{
    int num_devices = 0;

    ::hipGetDeviceCount( &num_devices );

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

