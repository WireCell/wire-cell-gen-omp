/**
 * Wrappers for hipFFT based FFT
 */

#ifndef WIRECELL_OPENMPARRAY_HIP 
#define WIRECELL_OPENMPARRAY_HIP

#include <complex>
#include <cmath>
#include <cassert>
#include <iostream>   //FOR DEBUG
#include <omp.h>      //FOR DEBUG

#include <hipfft.h>

namespace WireCell 
{
  namespace OpenMPArray 
  {

    inline void dft_rc_2d(std::complex<float>* out, const float* in, size_t N0, size_t N1)
    {
      hipfftHandle plan;

      assert( HIPFFT_SUCCESS == hipfftPlan2d(&plan, (int)N0, (int)N1, HIPFFT_R2C)
                            && "Error: dft_rc_2d, plan2d failed\n");
#pragma omp target data use_device_ptr(in, out)
      assert( HIPFFT_SUCCESS == hipfftExecR2C(plan, (float*)in, (float2*)out)
                            && "Error: dft_rc_2d, execR2C failed\n");
      hipfftDestroy(plan);
    }

    inline void dft_rc_2d_test(std::complex<float>* out, const float* in, size_t N0, size_t N1)
    {
//      constexpr int ISTRIDE = 1;   // distance between successive input elements in innermost dimension
//      constexpr int OSTRIDE = 1;   // distance between successive output elements in innermost dimension
//      const int IDIST = (N0*N1*ISTRIDE+3); // distance between first element of two consecutive signals in a batch of input data
//      const int ODIST = (N0*N1*OSTRIDE+5); // distance between first element of two consecutive signals in a batch of output data
//      
//      cufftHandle plan;
//      int n[2] = {int(N0), int(N1)};
//      int inembed[2] = {int(N0), int(N1)}; // pointer that indicates storage dimensions of input data
//      int onembed[2] = {int(N0), int(N1)}; // pointer that indicates storage dimensions of output data
//
//      auto error = cufftPlanMany(&plan, 2, n, inembed,ISTRIDE,IDIST, onembed,OSTRIDE,ODIST, CUFFT_C2C,1);
//      std::cout << "Error code is " << error << std::endl;
//      assert(0);

      //working code starts!
//      constexpr int BATCH = 1;
//      constexpr int NRANK = 2;
//      constexpr int ISTRIDE = 1;   // distance between successive input elements in innermost dimension
//      constexpr int OSTRIDE = 1;   // distance between successive output elements in innermost dimension
//      const int IX = N0;
//      const int IY = N1;
//      const int OX = N0;
//      const int OY = N1;
//      const int IDIST = (IX*IY*ISTRIDE+3); // distance between first element of two consecutive signals in a batch of input data
//      const int ODIST = (OX*OY*OSTRIDE+5); // distance between first element of two consecutive signals in a batch of output data
//      
//      cufftHandle plan;
//      int isize = IDIST * BATCH;
//      int osize = ODIST * BATCH;
//      int n[NRANK] = {int(N0), int(N1)};
//      int inembed[NRANK] = {IX, IY}; // pointer that indicates storage dimensions of input data
//      int onembed[NRANK] = {OX, OY}; // pointer that indicates storage dimensions of output data
//
//      auto error = cufftPlanMany(&plan, NRANK, n, inembed,ISTRIDE,IDIST, onembed,OSTRIDE,ODIST, CUFFT_C2C,BATCH);
//      std::cout << "Error code is " << error << std::endl;
//      assert(0);

      //working code ends!


      hipfftHandle plan;
      int n[2] = {(int)N0, int(N1)};
      int inembed[2] = {(int)N0, int(N1)};
      int onembed[2] = {(int)N0, int(N1)};

      assert( HIPFFT_SUCCESS == hipfftPlanMany(&plan, 2, n, inembed, 1, N0*N1, onembed, 1, N0*N1, HIPFFT_R2C, 1)
                            && "Error: dft_rc_2d_test, planmany failed\n");

//      auto error = cufftPlanMany(&plan, 2, n, inembed, 1, N0*N1, onembed, 1, N0*N1, CUFFT_R2C, 1);
//      if(error != CUFFT_SUCCESS)
//      {
//        std::cout << "Error code is " << error << std::endl;
//        assert(0);
//      }
//      assert(0 && "success!");
#pragma omp target data use_device_ptr(in, out)
      assert( HIPFFT_SUCCESS == hipfftExecR2C(plan, (float*)in, (float2*)out)
                            && "Error: dft_rc_2d_test, execR2C failed\n");
      hipfftDestroy(plan);
    }

    inline void dft_rc(std::complex<float>* out, const float* in, size_t N0, size_t N1, int dim = 0)
    {
      hipfftHandle plan;

      if(dim == 0) 
      {
        int n[] = {(int)N1};
        int inembed[] = {(int)N1};
        int onembed[] = {(int)N1};
        assert( HIPFFT_SUCCESS == hipfftPlanMany(&plan, 1, n, inembed, (int)N0, 1, onembed, (int)N0, 1, HIPFFT_R2C, (int)N0) 
                              && "Error: dim0, dft_rc, planmany failed\n");
#pragma omp target data use_device_ptr(in, out)
        assert( HIPFFT_SUCCESS == hipfftExecR2C(plan, (float*)in, (float2*)out)
                              && "Error: dim0, drf_rc, execR2C failed\n");
        hipfftDestroy(plan);
      }

      else if(dim == 1) 
      {
        int n[] = {(int)N0};
        int inembed[] = {(int)N0};
        int onembed[] = {(int)N0};
        assert( HIPFFT_SUCCESS == hipfftPlanMany(&plan, 1, n, inembed, 1, (int)N0, onembed, 1, (int)N0, HIPFFT_R2C, (int)N1)
                              && "Error: dim1, dft_rc, planmany failed\n");
#pragma omp target data use_device_ptr(in, out)
        assert( HIPFFT_SUCCESS == hipfftExecR2C(plan, (float*)in, (float2*)out)
                              && "Error: dim1, dft_rc, execR2C failed\n");
        hipfftDestroy(plan);
      }
    }

    //FIXME: This should be optimized to be in-place, and test performance diff (both speed and memory)
    //As the out and in can be the same, I remove the const
    inline void dft_cc(std::complex<float>* out, std::complex<float> *in, size_t N0, size_t N1, int dim = 0)
    {
      hipfftHandle plan;

      if(dim == 0) 
      {
        int n[] = {(int)N1};
        int inembed[] = {(int)N1};
        int onembed[] = {(int)N1};
        assert( HIPFFT_SUCCESS == hipfftPlanMany(&plan, 1, n, inembed, (int)N0, 1, onembed, (int)N0, 1, HIPFFT_C2C, (int)N0)
                              && "Error: dim0, dft_cc, planmany failed\n");
#pragma omp target data use_device_ptr(in, out)
        assert( HIPFFT_SUCCESS == hipfftExecC2C(plan, (float2*)in, (float2*)out, HIPFFT_FORWARD) 
                              && "Error: dim0, drf_cc, execC2C failed\n");
        hipfftDestroy(plan);
      }

      else if(dim == 1) 
      {
        int n[] = {(int)N0};
        int inembed[] = {(int)N0};
        int onembed[] = {(int)N0};
        assert( HIPFFT_SUCCESS == hipfftPlanMany(&plan, 1, n, inembed, 1, (int)N0, onembed, 1, (int)N0, HIPFFT_C2C, (int)N1) 
                              && "Error: dim1, dft_cc, planmany failed\n");
#pragma omp target data use_device_ptr(in, out)
        assert( HIPFFT_SUCCESS == hipfftExecC2C(plan, (float2*)in, (float2*)out, HIPFFT_FORWARD) 
                              && "Error: dim1, drf_cc, execR2C failed\n");
        hipfftDestroy(plan);
      }
    }

    //Can we do late evaluation for normalization? This takes several ms
    //FIXME: This should be optimized to be in-place like above
    //As the out and in could be the same, I remove const
    inline void idft_cc(std::complex<float>* out, std::complex<float> *in, size_t N0, size_t N1, int dim = 0)
    {
      hipfftHandle plan;

      if(dim == 0) 
      {
        int n[] = {(int)N1};
        int inembed[] = {(int)N1};
        int onembed[] = {(int)N1};
        assert( HIPFFT_SUCCESS ==  hipfftPlanMany(&plan, 1, n, inembed, (int)N0, 1, onembed, (int)N0, 1, HIPFFT_C2C, (int)N0)
                              && "Error: dim0, idft_cc, planmany failed\n");
#pragma omp target data use_device_ptr(in, out)
        assert( HIPFFT_SUCCESS ==  hipfftExecC2C(plan, (float2*)in, (float2*)out, HIPFFT_BACKWARD)
                              && "Error: dim0, idft_cc, execC2C failed\n");
        hipfftDestroy(plan);
//#pragma omp target teams distribute parallel for simd collapse(2)
//        for(int i0=0; i0<N0; i0++)
//        {
//          for(int i1=0; i1<N1; i1++)
//          {
//            //FIXME
//            out[i0 * N1 + i1] /= N1;
//          }
//        }

#pragma omp target teams distribute parallel for simd
        for(int i=0; i<N0 * N1; i++)
        {
          //FIXME: correctness, can we do that in the final step?
          out[i] /= N1;
        }
      }

      else if(dim == 1) 
      {
        int n[] = {(int)N0};
        int inembed[] = {(int)N0};
        int onembed[] = {(int)N0};
        assert( HIPFFT_SUCCESS ==  hipfftPlanMany(&plan, 1, n, inembed, 1, (int)N0, onembed, 1, (int)N0, HIPFFT_C2C, (int)N1)
                              && "Error: dim1, idft_cc, planmany failed\n");
#pragma omp target data use_device_ptr(in, out)
        assert( HIPFFT_SUCCESS ==  hipfftExecC2C(plan, (float2*)in, (float2*)out, HIPFFT_BACKWARD)
                              && "Error: dim1, idft_cc, execC2C failed\n");
        hipfftDestroy(plan);
//#pragma omp target teams distribute parallel for simd collapse(2)
//        for(int i0=0; i0<N0; i0++)
//        {
//          for(int i1=0; i1<N1; i1++)
//          {
//            //FIXME
//            out[i0 * N1 + i1] /= N0;
//          }
//        }

#pragma omp target teams distribute parallel for simd
        for(int i=0; i<N0 * N1; i++)
        {
          //FIXME: correctness, can we do that in the final step?
          out[i] /= N0;
        }
      }
    }

    inline void idft_cr(float* out, const std::complex<float> *in, size_t N0, size_t N1, int dim = 0)
    {
      hipfftHandle plan;

      if(dim == 0) 
      {
        int n[] = {(int)N1};
        int inembed[] = {(int)N1};
        int onembed[] = {(int)N1};
        assert( HIPFFT_SUCCESS ==  hipfftPlanMany(&plan, 1, n, inembed, (int)N0, 1, onembed, (int)N0, 1, HIPFFT_C2R, (int)N0)
                              && "Error: dim0, idft_cr, planmany failed\n");
#pragma omp target data use_device_ptr(in, out)
        assert( HIPFFT_SUCCESS ==  hipfftExecC2R(plan, (float2*)in, (float*)out)
                              && "Error: dim0, idft_cr, execC2R failed\n");
        hipfftDestroy(plan);
//#pragma omp target teams distribute parallel for simd collapse(2)
//        for(int i0=0; i0<N0; i0++)
//        {
//          for(int i1=0; i1<N1; i1++)
//          {
//            //FIXME
//            out[i0 * N1 + i1] /= N1;
//          }
//        }
#pragma omp target teams distribute parallel for simd
        for(int i=0; i<N0 * N1; i++)
        {
          //FIXME: correctness, can we do that in the final step?
          out[i] /= N1;
        }

      }

      else if(dim == 1) 
      {
        int n[] = {(int)N0};
        int inembed[] = {(int)N0};
        int onembed[] = {(int)N0};
        assert( HIPFFT_SUCCESS ==  hipfftPlanMany(&plan, 1, n, inembed, 1, (int)N0, onembed, 1, (int)N0, HIPFFT_C2R, (int)N1)
                              && "Error: dim1, idft_cr, planmany failed\n");
#pragma omp target data use_device_ptr(in, out)
        assert( HIPFFT_SUCCESS ==  hipfftExecC2R(plan, (float2*)in, (float*)out)
                              && "Error: dim1, idft_cr, execC2R failed\n");
        hipfftDestroy(plan);
//#pragma omp target teams distribute parallel for simd collapse(2)
//        for(int i0=0; i0<N0; i0++)
//        {
//          for(int i1=0; i1<N1; i1++)
//          {
//            //FIXME
//            out[i0 * N1 + i1] /= N0;
//          }
//        }
#pragma omp target teams distribute parallel for simd
        for(int i=0; i<N0 * N1; i++)
        {
          //FIXME: correctness, can we do that in the final step?
          out[i] /= N0;
        }

      }
    }

  }  // namespace OpenMPArray
}  // namespace WireCell

#endif
