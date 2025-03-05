/**
 * Wrappers for FFTW based FFT
 */

#ifndef WIRECELL_OPENMPARRAY_FFTW 
#define WIRECELL_OPENMPARRAY_FFTW

#include <complex>
#include <cmath>
#include <cassert>
#include <iostream>   //FOR DEBUG
#include <omp.h>      //FOR DEBUG

#include <WireCellUtil/Array.h>

namespace WireCell 
{
  namespace OpenMPArray 
  {

    thread_local static Eigen::FFT<float> gEigenFFT;

    inline void dft_rc(std::complex<float>* out, const float* in, size_t N0, size_t N1, int dim = 0)
    {
      Eigen::Map<const Eigen::ArrayXXf> in_eigen(in, N0, N1);
      auto out_eigen = WireCell::Array::dft_rc(in_eigen, dim);
      memcpy((void*)out, (void*)out_eigen.data(), N0*N1*sizeof(std::complex<float>));
    }

    //FIXME: This should be optimized to be in-place, and test performance diff (both speed and memory)
    //As the out and in can be the same, I remove the const
    inline void dft_cc(std::complex<float>* out, std::complex<float> *in, size_t N0, size_t N1, int dim = 0)
    {
      Eigen::Map<Eigen::ArrayXXcf> in_eigen(in, N0, N1);
      auto out_eigen = WireCell::Array::dft_cc(in_eigen, dim);
      memcpy((void*)out, (void*)out_eigen.data(), N0*N1*sizeof(std::complex<float>));
    }

    //Can we do late evaluation for normalization? This takes several ms
    //FIXME: This should be optimized to be in-place like above
    //As the out and in could be the same, I remove const
    inline void idft_cc(std::complex<float>* out, std::complex<float> *in, size_t N0, size_t N1, int dim = 0)
    {
      Eigen::Map<Eigen::ArrayXXcf> in_eigen(in, N0, N1);
      auto out_eigen = WireCell::Array::idft_cc(in_eigen, dim);
      memcpy((void*)out, (void*)out_eigen.data(), N0*N1*sizeof(std::complex<float>));
    }

    inline void idft_cr(float* out, const std::complex<float> *in, size_t N0, size_t N1, int dim = 0)
    {
      Eigen::Map<const Eigen::ArrayXXcf> in_eigen(in, N0, N1);
      auto out_eigen = WireCell::Array::idft_cr(in_eigen, dim);
      memcpy((void*)out, (void*)out_eigen.data(), N0*N1*sizeof(float));
    }

//    inline void dft_rc_2d(std::complex<float>* out, const float* in, size_t N0, size_t N1)
//    {
//      cufftHandle plan;
//
//      assert( CUFFT_SUCCESS == cufftPlan2d(&plan, (int)N0, (int)N1, CUFFT_R2C)
//                            && "Error: dft_rc_2d, plan2d failed\n");
//#pragma omp target data use_device_ptr(in, out)
//      assert( CUFFT_SUCCESS == cufftExecR2C(plan, (cufftReal*)in, (cufftComplex*)out)
//                            && "Error: dft_rc_2d, execR2C failed\n");
//      cufftDestroy(plan);
//    }
//
//    inline void dft_rc_2d_test(std::complex<float>* out, const float* in, size_t N0, size_t N1)
//    {
////      constexpr int ISTRIDE = 1;   // distance between successive input elements in innermost dimension
////      constexpr int OSTRIDE = 1;   // distance between successive output elements in innermost dimension
////      const int IDIST = (N0*N1*ISTRIDE+3); // distance between first element of two consecutive signals in a batch of input data
////      const int ODIST = (N0*N1*OSTRIDE+5); // distance between first element of two consecutive signals in a batch of output data
////      
////      cufftHandle plan;
////      int n[2] = {int(N0), int(N1)};
////      int inembed[2] = {int(N0), int(N1)}; // pointer that indicates storage dimensions of input data
////      int onembed[2] = {int(N0), int(N1)}; // pointer that indicates storage dimensions of output data
////
////      auto error = cufftPlanMany(&plan, 2, n, inembed,ISTRIDE,IDIST, onembed,OSTRIDE,ODIST, CUFFT_C2C,1);
////      std::cout << "Error code is " << error << std::endl;
////      assert(0);
//
//      //working code starts!
////      constexpr int BATCH = 1;
////      constexpr int NRANK = 2;
////      constexpr int ISTRIDE = 1;   // distance between successive input elements in innermost dimension
////      constexpr int OSTRIDE = 1;   // distance between successive output elements in innermost dimension
////      const int IX = N0;
////      const int IY = N1;
////      const int OX = N0;
////      const int OY = N1;
////      const int IDIST = (IX*IY*ISTRIDE+3); // distance between first element of two consecutive signals in a batch of input data
////      const int ODIST = (OX*OY*OSTRIDE+5); // distance between first element of two consecutive signals in a batch of output data
////      
////      cufftHandle plan;
////      int isize = IDIST * BATCH;
////      int osize = ODIST * BATCH;
////      int n[NRANK] = {int(N0), int(N1)};
////      int inembed[NRANK] = {IX, IY}; // pointer that indicates storage dimensions of input data
////      int onembed[NRANK] = {OX, OY}; // pointer that indicates storage dimensions of output data
////
////      auto error = cufftPlanMany(&plan, NRANK, n, inembed,ISTRIDE,IDIST, onembed,OSTRIDE,ODIST, CUFFT_C2C,BATCH);
////      std::cout << "Error code is " << error << std::endl;
////      assert(0);
//
//      //working code ends!
//
//
//      cufftHandle plan;
//      int n[2] = {(int)N0, int(N1)};
//      int inembed[2] = {(int)N0, int(N1)};
//      int onembed[2] = {(int)N0, int(N1)};
//
//      assert( CUFFT_SUCCESS == cufftPlanMany(&plan, 2, n, inembed, 1, N0*N1, onembed, 1, N0*N1, CUFFT_R2C, 1)
//                            && "Error: dft_rc_2d_test, planmany failed\n");
//
////      auto error = cufftPlanMany(&plan, 2, n, inembed, 1, N0*N1, onembed, 1, N0*N1, CUFFT_R2C, 1);
////      if(error != CUFFT_SUCCESS)
////      {
////        std::cout << "Error code is " << error << std::endl;
////        assert(0);
////      }
////      assert(0 && "success!");
//#pragma omp target data use_device_ptr(in, out)
//      assert( CUFFT_SUCCESS == cufftExecR2C(plan, (cufftReal*)in, (cufftComplex*)out)
//                            && "Error: dft_rc_2d_test, execR2C failed\n");
//      cufftDestroy(plan);
//    }

  }  // namespace OpenMPArray
}  // namespace WireCell

#endif
