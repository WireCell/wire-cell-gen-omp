/**
 * Wrappers for cuFFT based FFT
 */

#ifndef WIRECELL_OPENMPARRAY_CUDA
#define WIRECELL_OPENMPARRAY_CUDA

#include <complex>
#include <cmath>
#include <cassert>
#include <iostream>   //FOR DEBUG

#include <cufft.h>

namespace WireCell 
{
  namespace OpenMPArray 
  {

    inline void dft_rc(std::complex<float>* out, const float* in, size_t N0, size_t N1, int dim = 0)
    {
      cufftHandle plan;

      if(dim == 0) 
      {
        int n[] = {(int)N1};
        int inembed[] = {(int)N1};
        int onembed[] = {(int)N1};
        assert( CUFFT_SUCCESS == cufftPlanMany(&plan, 1, n, inembed, (int)N0, 1, onembed, (int)N0, 1, CUFFT_R2C, (int)N0) 
                              && "Error: dim0, dft_rc, planmany failed\n");
#pragma omp target data use_device_ptr(in, out)
        assert( CUFFT_SUCCESS == cufftExecR2C(plan, (cufftReal*)in, (cufftComplex*)out)
                              && "Error: dim0, drf_rc, execR2C failed\n");
        cufftDestroy(plan);
      }

      else if(dim == 1) 
      {
        int n[] = {(int)N0};
        int inembed[] = {(int)N0};
        int onembed[] = {(int)N0};
        assert( CUFFT_SUCCESS == cufftPlanMany(&plan, 1, n, inembed, 1, (int)N0, onembed, 1, (int)N0, CUFFT_R2C, (int)N1)
                              && "Error: dim1, dft_rc, planmany failed\n");
#pragma omp target data use_device_ptr(in, out)
        assert( CUFFT_SUCCESS == cufftExecR2C(plan, (cufftReal*)in, (cufftComplex*)out)
                              && "Error: dim1, dft_rc, execR2C failed\n");
        cufftDestroy(plan);
      }
    }

    //FIXME: This should be optimized to be in-place, and test performance diff (both speed and memory)
    //As the out and in can be the same, I remove the const
    inline void dft_cc(std::complex<float>* out, std::complex<float> *in, size_t N0, size_t N1, int dim = 0)
    {
      cufftHandle plan;

      if(dim == 0) 
      {
        int n[] = {(int)N1};
        int inembed[] = {(int)N1};
        int onembed[] = {(int)N1};
        assert( CUFFT_SUCCESS == cufftPlanMany(&plan, 1, n, inembed, (int)N0, 1, onembed, (int)N0, 1, CUFFT_C2C, (int)N0)
                              && "Error: dim0, dft_cc, planmany failed\n");
#pragma omp target data use_device_ptr(in, out)
        assert( CUFFT_SUCCESS == cufftExecC2C(plan, (cufftComplex*)in, (cufftComplex*)out, CUFFT_FORWARD) 
                              && "Error: dim0, drf_cc, execR2C failed\n");
        cufftDestroy(plan);
      }

      else if(dim == 1) 
      {
        int n[] = {(int)N0};
        int inembed[] = {(int)N0};
        int onembed[] = {(int)N0};
        assert( CUFFT_SUCCESS == cufftPlanMany(&plan, 1, n, inembed, 1, (int)N0, onembed, 1, (int)N0, CUFFT_C2C, (int)N1) 
                              && "Error: dim1, dft_cc, planmany failed\n");
#pragma omp target data use_device_ptr(in, out)
        assert( CUFFT_SUCCESS == cufftExecC2C(plan, (cufftComplex*)in, (cufftComplex*)out, CUFFT_FORWARD) 
                              && "Error: dim1, drf_cc, execR2C failed\n");
        cufftDestroy(plan);
      }
    }

    //Can we do late evaluation for normalization? This takes several ms
    //FIXME: This should be optimized to be in-place like above
    //As the out and in could be the same, I remove const
    inline void idft_cc(std::complex<float>* out, std::complex<float> *in, size_t N0, size_t N1, int dim = 0)
    {
      cufftHandle plan;

      if(dim == 0) 
      {
        int n[] = {(int)N1};
        int inembed[] = {(int)N1};
        int onembed[] = {(int)N1};
        assert( CUFFT_SUCCESS ==  cufftPlanMany(&plan, 1, n, inembed, (int)N0, 1, onembed, (int)N0, 1, CUFFT_C2C, (int)N0)
                              && "Error: dim0, idft_cc, planmany failed\n");
#pragma omp target data use_device_ptr(in, out)
        assert( CUFFT_SUCCESS ==  cufftExecC2C(plan, (cufftComplex*)in, (cufftComplex*)out, CUFFT_INVERSE)
                              && "Error: dim0, idft_cc, execC2C failed\n");
        cufftDestroy(plan);
#pragma omp target teams distribute parallel for collapse(2)
        for(int i0=0; i0<N0; i0++)
        {
          for(int i1=0; i1<N1; i1++)
          {
            //FIXME
            out[i0 * N1 + i1] /= N1;
          }
        }
      }

      else if(dim == 1) 
      {
        int n[] = {(int)N0};
        int inembed[] = {(int)N0};
        int onembed[] = {(int)N0};
        assert( CUFFT_SUCCESS ==  cufftPlanMany(&plan, 1, n, inembed, 1, (int)N0, onembed, 1, (int)N0, CUFFT_C2C, (int)N1)
                              && "Error: dim1, idft_cc, planmany failed\n");
#pragma omp target data use_device_ptr(in, out)
        assert( CUFFT_SUCCESS ==  cufftExecC2C(plan, (cufftComplex*)in, (cufftComplex*)out, CUFFT_INVERSE)
                              && "Error: dim1, idft_cc, execC2C failed\n");
        cufftDestroy(plan);
#pragma omp target teams distribute parallel for collapse(2)
        for(int i0=0; i0<N0; i0++)
        {
          for(int i1=0; i1<N1; i1++)
          {
            //FIXME
            out[i0 * N1 + i1] /= N0;
          }
        }
      }
    }

    inline void idft_cr(float* out, const std::complex<float> *in, size_t N0, size_t N1, int dim = 0)
    {
      cufftHandle plan;

      if(dim == 0) 
      {
        int n[] = {(int)N1};
        int inembed[] = {(int)N1};
        int onembed[] = {(int)N1};
        assert( CUFFT_SUCCESS ==  cufftPlanMany(&plan, 1, n, inembed, (int)N0, 1, onembed, (int)N0, 1, CUFFT_C2R, (int)N0)
                              && "Error: dim0, idft_cr, planmany failed\n");
#pragma omp target data use_device_ptr(in, out)
        assert( CUFFT_SUCCESS ==  cufftExecC2R(plan, (cufftComplex*)in, (cufftReal*)out)
                              && "Error: dim0, idft_cr, execC2R failed\n");
        cufftDestroy(plan);
#pragma omp target teams distribute parallel for collapse(2)
        for(int i0=0; i0<N0; i0++)
        {
          for(int i1=0; i1<N1; i1++)
          {
            //FIXME
            out[i0 * N1 + i1] /= N1;
          }
        }
      }

      else if(dim == 1) 
      {
        int n[] = {(int)N0};
        int inembed[] = {(int)N0};
        int onembed[] = {(int)N0};
        assert( CUFFT_SUCCESS ==  cufftPlanMany(&plan, 1, n, inembed, 1, (int)N0, onembed, 1, (int)N0, CUFFT_C2R, (int)N1)
                              && "Error: dim1, idft_cr, planmany failed\n");
#pragma omp target data use_device_ptr(in, out)
        assert( CUFFT_SUCCESS ==  cufftExecC2R(plan, (cufftComplex*)in, (cufftReal*)out)
                              && "Error: dim1, idft_cr, execC2R failed\n");
        cufftDestroy(plan);
#pragma omp target teams distribute parallel for collapse(2)
        for(int i0=0; i0<N0; i0++)
        {
          for(int i1=0; i1<N1; i1++)
          {
            //FIXME
            out[i0 * N1 + i1] /= N0;
          }
        }
      }
    }

  }  // namespace OpenMPArray
}  // namespace WireCell

#endif
