/**
 * Wrappers for cuFFT based FFT
 */

#ifndef WIRECELL_OPENMPARRAY_CUDA
#define WIRECELL_OPENMPARRAY_CUDA

#include <complex>
#include <cmath>

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
        cufftPlanMany(&plan, 1, n, inembed, (int)N0, 1, onembed, (int)N0, 1, CUFFT_R2C, (int)N0);
        cufftExecR2C(plan, (cufftReal*)in, (cufftComplex*)out);
        cufftDestroy(plan);
      }

      else if(dim == 1) 
      {
        int n[] = {(int)N0};
        int inembed[] = {(int)N0};
        int onembed[] = {(int)N0};
        cufftPlanMany(&plan, 1, n, inembed, 1, (int)N0, onembed, 1, (int)N0, CUFFT_R2C, (int)N1);
        cufftExecR2C(plan, (cufftReal*)in, (cufftComplex*)out);
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
        cufftPlanMany(&plan, 1, n, inembed, (int)N0, 1, onembed, (int)N0, 1, CUFFT_C2C, (int)N0);
        cufftExecC2C(plan, (cufftComplex*)in, (cufftComplex*)out, CUFFT_FORWARD);
        cufftDestroy(plan);
      }

      else if(dim == 1) 
      {
        int n[] = {(int)N0};
        int inembed[] = {(int)N0};
        int onembed[] = {(int)N0};
        cufftPlanMany(&plan, 1, n, inembed, 1, (int)N0, onembed, 1, (int)N0, CUFFT_C2C, (int)N1);
        cufftExecC2C(plan, (cufftComplex*)in, (cufftComplex*)out, CUFFT_FORWARD);
        cufftDestroy(plan);
      }
    }

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
        cufftPlanMany(&plan, 1, n, inembed, (int)N0, 1, onembed, (int)N0, 1, CUFFT_C2C, (int)N0);
        cufftExecC2C(plan, (cufftComplex*)in, (cufftComplex*)out, CUFFT_INVERSE);
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
        cufftPlanMany(&plan, 1, n, inembed, 1, (int)N0, onembed, 1, (int)N0, CUFFT_C2C, (int)N1);
        cufftExecC2C(plan, (cufftComplex*)in, (cufftComplex*)out, CUFFT_INVERSE);
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
        cufftPlanMany(&plan, 1, n, inembed, (int)N0, 1, onembed, (int)N0, 1, CUFFT_C2R, (int)N0);
        cufftExecC2R(plan, (cufftComplex*)in, (cufftReal*)out);
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
        cufftPlanMany(&plan, 1, n, inembed, 1, (int)N0, onembed, 1, (int)N0, CUFFT_C2R, (int)N1);
        cufftExecC2R(plan, (cufftComplex*)in, (cufftReal*)out);
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
