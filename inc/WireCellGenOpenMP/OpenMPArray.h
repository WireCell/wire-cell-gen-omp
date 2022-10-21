/**
 * Similar like the WireCell::Array with Eigen backend,
 * this OpenMPArray provides interface for FFTs.
 */

#ifndef WIRECELL_OPENMPARRAY
#define WIRECELL_OPENMPARRAY

#include <string>
#include <typeinfo>

#if defined OPENMP_ENABLE_CUDA
    #include "WireCellGenOpenMP/OpenMPArray_cuda.h"
#elif defined OPENMP_ENABLE_HIP
    #include "WireCellGenOpenMP/OpenMPArray_hip.h"
#else
    #include "WireCellGenOpenMP/OpenMPArray_fftw.h"
#endif

#endif
