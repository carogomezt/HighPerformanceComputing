#include <cuda.h>
#include <sstream>

void DisplayCUDAHeader(std::stringstream& stream) {
    const int kb = 1024;
    const int mb = kb * kb;

    stream << "CUDA version:   v" << CUDART_VERSION << std::endl << std::endl;    

    int devCount;
    cudaGetDeviceCount(&devCount);
    stream << "CUDA Devices: " << std::endl << std::endl;

    if (devCount == 0) stream << "NONE" << std::endl << std::endl;

    for(int i = 0; i < devCount; ++i) {
        cudaDeviceProp props;
        cudaGetDeviceProperties(&props, i);
        stream << i << ": " << props.name << ": " << props.major << "." << props.minor << std::endl;
        stream << "  Global memory:   " << props.totalGlobalMem / mb << "mb" << std::endl;
        stream << "  Shared memory:   " << props.sharedMemPerBlock / kb << "kb" << std::endl;
        stream << "  Constant memory: " << props.totalConstMem / kb << "kb" << std::endl;
        stream << "  Block registers: " << props.regsPerBlock << std::endl << std::endl;

        stream << "  Warp size:         " << props.warpSize << std::endl;
        stream << "  Threads per block: " << props.maxThreadsPerBlock << std::endl;
        stream << "  Max block dimensions: [ " << props.maxThreadsDim[0] << ", " << props.maxThreadsDim[1]  << ", " << props.maxThreadsDim[2] << " ]" << std::endl;
        stream << "  Max grid dimensions:  [ " << props.maxGridSize[0] << ", " << props.maxGridSize[1]  << ", " << props.maxGridSize[2] << " ]" << std::endl;
        stream << std::endl;
    }
}