#include "WireCellGenOpenMP/BinnedDiffusion_transform.h"
#include "WireCellGenOpenMP/GaussianDiffusion.h"
#include "WireCellUtil/Units.h"

#include <iostream>             // debug
#include <omp.h>
#include <unordered_map>
#include <cmath>
#include <typeinfo>

#include "openmp_rng.h"

#define MAX_PATCH_SIZE 512 
#define P_BLOCK_SIZE  512
#define MAX_PATCHES  50000
#define MAX_NPSS_DEVICE 1000
#define MAX_NTSS_DEVICE 1000
#define FULL_MASK 0xffffffff
#define RANDOM_BLOCK_SIZE (1024*1024)
#define RANDOM_BLOCK_NUM 512
//#define MAX_RANDOM_LENGTH (RANDOM_BLOCK_NUM*RANDOM_BLOCK_SIZE)
#define MAX_RANDOM_LENGTH (MAX_PATCH_SIZE*MAX_PATCHES)
#define PI 3.14159265358979323846

#define MAX_P_SIZE 50 
#define MAX_T_SIZE 50 



using namespace std;

using namespace WireCell;

double g_get_charge_vec_time_part1 = 0.0;
double g_get_charge_vec_time_part2 = 0.0;
double g_get_charge_vec_time_part3 = 0.0;
double g_get_charge_vec_time_part4 = 0.0;
double g_get_charge_vec_time_part5 = 0.0;

extern double g_set_sampling_part1;
extern double g_set_sampling_part2;
extern double g_set_sampling_part3;
extern double g_set_sampling_part4;
extern double g_set_sampling_part5;

extern size_t g_total_sample_size;


GenOpenMP::BinnedDiffusion_transform::BinnedDiffusion_transform(const Pimpos& pimpos, const Binning& tbins,
                                      double nsigma, IRandom::pointer fluctuate,
                                      ImpactDataCalculationStrategy calcstrat)
    : m_pimpos(pimpos)
    , m_tbins(tbins)
    , m_nsigma(nsigma)
    , m_fluctuate(fluctuate)
    , m_calcstrat(calcstrat)
    , m_window(0,0)
    , m_outside_pitch(0)
    , m_outside_time(0)
{
}

bool GenOpenMP::BinnedDiffusion_transform::add(IDepo::pointer depo, double sigma_time, double sigma_pitch)
{

    const double center_time = depo->time();
    const double center_pitch = m_pimpos.distance(depo->pos());

    GenOpenMP::GausDesc time_desc(center_time, sigma_time);
    {
        double nmin_sigma = time_desc.distance(m_tbins.min());
        double nmax_sigma = time_desc.distance(m_tbins.max());

        double eff_nsigma = sigma_time>0?m_nsigma:0;
        if (nmin_sigma > eff_nsigma || nmax_sigma < -eff_nsigma) {
            // std::cerr << "BinnedDiffusion_transform: depo too far away in time sigma:"
            //           << " t_depo=" << center_time/units::ms << "ms not in:"
            //           << " t_bounds=[" << m_tbins.min()/units::ms << ","
            //           << m_tbins.max()/units::ms << "]ms"
            //           << " in Nsigma: [" << nmin_sigma << "," << nmax_sigma << "]\n";
            ++m_outside_time;
            return false;
        }
    }

    auto ibins = m_pimpos.impact_binning();

    GenOpenMP::GausDesc pitch_desc(center_pitch, sigma_pitch);
    {
        double nmin_sigma = pitch_desc.distance(ibins.min());
        double nmax_sigma = pitch_desc.distance(ibins.max());

        double eff_nsigma = sigma_pitch>0?m_nsigma:0;
        if (nmin_sigma > eff_nsigma || nmax_sigma < -eff_nsigma) {
            // std::cerr << "BinnedDiffusion_transform: depo too far away in pitch sigma: "
            //           << " p_depo=" << center_pitch/units::cm << "cm not in:"
            //           << " p_bounds=[" << ibins.min()/units::cm << ","
            //           << ibins.max()/units::cm << "]cm"
            //           << " in Nsigma:[" << nmin_sigma << "," << nmax_sigma << "]\n";
            ++m_outside_pitch;
            return false;
        }
    }

    // make GD and add to all covered impacts
    // int bin_beg = std::max(ibins.bin(center_pitch - sigma_pitch*m_nsigma), 0);
    // int bin_end = std::min(ibins.bin(center_pitch + sigma_pitch*m_nsigma)+1, ibins.nbins());
    // debug
    //int bin_center = ibins.bin(center_pitch);
    //cerr << "DEBUG center_pitch: "<<center_pitch/units::cm<<endl; 
    //cerr << "DEBUG bin_center: "<<bin_center<<endl;

    auto gd = std::make_shared<GaussianDiffusion>(depo, time_desc, pitch_desc);
    // for (int bin = bin_beg; bin < bin_end; ++bin) {
    //   //   if (bin == bin_beg)  m_diffs.insert(gd);
    //   this->add(gd, bin);
    // }
    m_diffs.push_back(gd);
    return true;
}

//FIXME: signature of first argument should be changed!!!
void GenOpenMP::BinnedDiffusion_transform::get_charge_matrix_openmp(float* out, size_t dim_p, size_t dim_t,
                                                                    std::vector<int>& vec_impact, const int start_pitch,
                                                                    const int start_tick)
{
  std::cout << "tw: get_charge_matrix_openmp\n";

  double wstart, wend ;
  wstart = omp_get_wtime();
  const auto ib = m_pimpos.impact_binning();

  // map between reduced impact # to array #

  std::map<int, int> map_redimp_vec;
  std::vector<std::unordered_map<long int, int> > vec_map_pair_pos;
  for(size_t i = 0; i != vec_impact.size(); i++) 
  {
    map_redimp_vec[vec_impact[i]] = int(i);
    std::unordered_map<long int, int> map_pair_pos;
    vec_map_pair_pos.push_back(map_pair_pos);
  }
  wend = omp_get_wtime();
  g_get_charge_vec_time_part1 = wend - wstart;
  cout << "get_charge_matrix_openmp(): part1 running time : " << g_get_charge_vec_time_part1 << endl;
  std::cout << "tw: is this step really necessary???\n";


  wstart = omp_get_wtime();
  const auto rb = m_pimpos.region_binning();
  // map between impact # to channel #
  std::map<int, int> map_imp_ch;
  // map between impact # to reduced impact #
  std::map<int, int> map_imp_redimp;

  std::cout << "tw: " << rb.nbins() << std::endl;
  for(int wireind = 0; wireind != rb.nbins(); wireind++) 
  {
    int wire_imp_no = m_pimpos.wire_impact(wireind);
    std::pair<int, int> imps_range = m_pimpos.wire_impacts(wireind);
    for(int imp_no = imps_range.first; imp_no != imps_range.second; imp_no++) 
    {
      map_imp_ch[imp_no] = wireind;
      map_imp_redimp[imp_no] = imp_no - wire_imp_no;
    }
  }

  int min_imp = 0;
  int max_imp = ib.nbins();
  int counter = 0;

  wend = omp_get_wtime();
  g_get_charge_vec_time_part2 = wend - wstart;
  cout << "get_charge_matrix_openmp(): part2 running time : " << g_get_charge_vec_time_part2 << endl;
  std::cout << "tw: is this step really necessary???\n";

  std::cout << "tw: I believe this is where things really get started. How did we get m_diffs?\n";
  wstart = omp_get_wtime();
  int npatches = m_diffs.size();
  GenOpenMP::GdData* gdata = (GenOpenMP::GdData*)malloc(sizeof(GenOpenMP::GdData) * npatches);

  //Can we do compute/data movement asynchronously? Necessary?
  int ii = 0;
  for (auto diff : m_diffs) 
  {
    gdata[ii].p_ct = diff->pitch_desc().center;
    gdata[ii].t_ct = diff->time_desc().center;
    gdata[ii].charge = diff->depo()->charge();
    gdata[ii].t_sigma = diff->time_desc().sigma;
    gdata[ii].p_sigma = diff->pitch_desc().sigma;
//    if(diff->pitch_desc().sigma == 0 || diff->time_desc().sigma == 0) 
//      std::cout<<"sigma-0 patch: " << ii << std::endl ;
    ii++;
  }

#pragma omp target enter data map(to:gdata[0:npatches])
  // make and device friendly Binning  and copy tbin pbin over.
  // tw: What are these two? What is m_tbins and ib? Where is rb?
  GenOpenMP::DBin tb, pb;
  tb.nbins = m_tbins.nbins();
  tb.minval = m_tbins.min();
  tb.binsize = m_tbins.binsize();
  pb.nbins = ib.nbins();
  pb.minval = ib.min();
  pb.binsize = ib.binsize();

  //FIXME: Do we need to copy that to device?
#pragma omp target enter data map(to:tb,pb)

  // perform set_sampling_pre tasks on gpu
  // FIXME Think about if we can use target_alloc to generate data so that we don't need to generate the host vesion!
  // FIXME Check if we use () instead of [] all around the file!!!
  unsigned int* np_vec  = (unsigned int*)malloc(sizeof(unsigned int) * npatches);
  unsigned int* nt_vec  = (unsigned int*)malloc(sizeof(unsigned int) * npatches);
  unsigned int* offsets = (unsigned int*)malloc(sizeof(unsigned int) * npatches * 2);
  unsigned long* patch_idx = (unsigned long*)malloc(sizeof(unsigned long*) * (npatches + 1));
  double* pvecs     = (double*)malloc(sizeof(double) * npatches * MAX_P_SIZE);
  double* tvecs     = (double*)malloc(sizeof(double) * npatches * MAX_T_SIZE);
  double* qweights  = (double*)malloc(sizeof(double) * npatches * MAX_P_SIZE);

#pragma omp target enter data map(alloc:np_vec[0:npatches],nt_vec[0:npatches],offsets[0:npatches*2])
#pragma omp target enter data map(alloc:pvecs[0:npatches*MAX_P_SIZE],tvecs[0:npatches*MAX_T_SIZE],qweights[0:npatches*MAX_P_SIZE])

  //FIXME: Do we need to combine np_vec and nt_vec together, just like offsets, or do we need to split offsets, like np_vec and nt_vec???
  // Kernel for calculate nt_vec and np_vec and offsets for t and p for each gd
  int nsigma = m_nsigma;
#pragma omp target teams distribute parallel for simd
  for(int i=0; i<npatches; i++)
  {
    double t_s = gdata[i].t_ct - gdata[i].t_sigma * nsigma;
    double t_e = gdata[i].t_ct + gdata[i].t_sigma * nsigma;
    int t_ofb = max(int((t_s - tb.minval) / tb.binsize), 0);
    int ntss = min((int((t_e - tb.minval) / tb.binsize)) + 1, tb.nbins) - t_ofb;

    double p_s = gdata[i].p_ct - gdata[i].p_sigma * nsigma;
    double p_e = gdata[i].p_ct + gdata[i].p_sigma * nsigma;
    int p_ofb = max(int((p_s - pb.minval) / pb.binsize), 0);
    int npss = min((int((p_e - pb.minval) / pb.binsize)) + 1, pb.nbins) - p_ofb;

    //FIXME: Can we do assignment directly? Will that harm cache? Will that improve register?
    
    nt_vec[i] = ntss;
    np_vec[i] = npss;
    offsets[i] = t_ofb;
    offsets[npatches + i] = p_ofb;
  }

  //FIXME I change the name np_d and nt_d to np_vec and nt_vec. Need to check if this is consistent all around the file
  //Calculate index for patch, temporary on cpu, can we improve that by writing an gpu version of scan? FIXME
#pragma omp target update from(np_vec[0:npatches],nt_vec[0:npatches])

  unsigned long result = 0;  // total patches points, openmp scan can not give the correct sum???

// Seems like this cost a very long time!!!!!

  patch_idx[0] = 0;
#pragma omp parallel for simd reduction(inscan,+:result)
  for(int i=0; i<npatches; i++)
  {
    result += (np_vec[i] * nt_vec[i]);
    #pragma omp scan inclusive(result)
    patch_idx[i+1] = result;
  }

  result = patch_idx[npatches];   //As openmp scan does not return the correct sum, we use inclusive scan start from idx 1
  std::cout << "result = " << result << std::endl;
#pragma omp target enter data map(to:patch_idx[0:npatches])

  // debug:
  std::cout << "total patch size: " << result << " WeightStrat: " << m_calcstrat << std::endl;
  
  // Allocate space for patches on device, we might also want to use target_alloc
  float* patch = (float*)malloc(sizeof(float) * result);
#pragma omp target enter data map(alloc:patch[0:result])

  //FIXME Should we save them in m_normals or create rd_normals and save them there?
  int size = (result+255) / 256 * 256;    //tw: This might not be necessary any more! 
  m_normals = (double*)malloc(sizeof(double) * size);
  unsigned long long seed = 2020;

#pragma omp target enter data map(alloc:m_normals[0:size])
#pragma omp target data use_device_ptr(m_normals)  
  omp_get_rng_normal_double(m_normals, size, 0.0, 1.0, seed);

  std::cout << "Create random numbers successfully!" << std::endl;

  // decide weight calculation
  int weightstrat = m_calcstrat;

  // each team resposible for 1 GD , kernel calculate pvecs and tvecs
  const double sqrt2 = sqrt(2.0);
  std::cout << " Start to compute pvecs and tvecs!" << std::endl;

  //Here I am trying to debug, so that the loop is divided into many loops. Need to put them together later!
#pragma omp target teams distribute
  for(int ip=0; ip<npatches; ip++)
  {
    double start_t = tb.minval + offsets[ip] * tb.binsize;
    double start_p = pb.minval + offsets[ip + npatches] * pb.binsize;
    int np = np_vec[ip];
    int nt = nt_vec[ip];

    if(np == 1)
      pvecs[ip * MAX_P_SIZE] = 1.0;
    else
    {
#pragma omp parallel for simd
      for(int ii=0; ii<np; ii++)
      {
        double step = pb.binsize;
        double factor = sqrt2 * gdata[ip].p_sigma;
        double x = (start_p + step * ii - gdata[ip].p_ct) / factor;
        double ef1 = 0.5 * erf(x);
        double ef2 = 0.5 * erf(x + step / factor);
        double val = ef2 - ef1;
        pvecs[ip * MAX_P_SIZE + ii] = val;
      }
    }
    
    if(nt == 1)
      tvecs[ip * MAX_T_SIZE] = 1.0;
    else
    {
#pragma omp parallel for simd
      for(int ii=0; ii<nt; ii++)
      {
        double step = tb.binsize;
        double factor = sqrt2 * gdata[ip].t_sigma;
        double x = (start_t + step * ii - gdata[ip].t_ct) / factor;
        double ef1 = 0.5 * erf(x);
        double ef2 = 0.5 * erf(x + step / factor);
        double val = ef2 - ef1;
        tvecs[ip * MAX_T_SIZE + ii] = val;
      }
    }
    
    if(weightstrat == 2)
    {
      if(gdata[ip].p_sigma == 0)
        qweights[ip * MAX_P_SIZE] = (start_p + pb.binsize - gdata[ip].p_ct) / pb.binsize;
      else
      {
#pragma omp parallel for simd
        for(int ii=0; ii<np; ii++)
        {
          double rel1 = (start_p + pb.binsize * ii - gdata[ip].p_ct) / gdata[ip].p_sigma;
          double rel2 = rel1 + pb.binsize / gdata[ip].p_sigma;
          double gaus1 = exp(-0.5 * rel1 * rel1);
          double gaus2 = exp(-0.5 * rel2 * rel2);
          double wt = -1.0 * gdata[ip].p_sigma / pb.binsize * (gaus2 - gaus1) / sqrt(2.0 * PI) / pvecs[ip * MAX_P_SIZE + ii]
                      + (gdata[ip].p_ct - (start_p + (ii + 1) * pb.binsize)) / pb.binsize;
          qweights[ip * MAX_P_SIZE + ii] = -wt;
        }
      }
    }
  }
  std::cout << "Compute pvecs, tvecs and qweights successfully!\n";

  wend = omp_get_wtime();
  //IMPORTANT: NOW WE KNOW nt_vec, np_vec and offsets are identical between omp and kokkos
  //IMPORTANT: NOW WE KNOW pvecs, tvecs and qweights are identical between omp and kokkos
  set_sampling_bat(npatches, nt_vec, np_vec, patch_idx, pvecs, tvecs, patch, m_normals, gdata);
  //IMPORTANT: NOW WE KNOW patch are identical between omp and kokkos

  wstart = omp_get_wtime();
  cout << "pr21 get_charge_matrix_openmp(): set_sampling_bat() no DtoH time " << wstart - wend << endl;
  std::cout << "tw: DEBUG: npatches: " << npatches << std::endl;
//  std::cout << "tw: DEBUG: np_vec: " << OpenMPArray::dump_1d_view(np_d,10000) << std::endl;
//  std::cout << "tw: DEBUG: nt_vec: " << OpenMPArray::dump_1d_view(nt_d,10000) << std::endl;
//  std::cout << "tw: DEBUG: offsets_d: " << OpenMPArray::dump_1d_view(offsets_d,10000) << std::endl;
//  std::cout << "tw: DEBUG: patch_idx: " << OpenMPArray::dump_1d_view(patch_idx,10000) << std::endl;
//  std::cout << "tw: DEBUG: patch_d: " << OpenMPArray::dump_1d_view(patch_d,10000) << std::endl;
//  std::cout << "tw: DEBUG: qweights_d: " << OpenMPArray::dump_1d_view(qweights_d,10000) << std::endl;

#pragma omp target teams distribute
  for(int ip=0; ip<npatches; ip++)
  {
    int np = np_vec[ip];
    int nt = nt_vec[ip];
    int p = offsets[npatches + ip] - start_pitch;
    int t = offsets[ip] - start_tick;
    int patch_size = np * nt;
#pragma omp parallel for simd
    for(int i=0; i<patch_size; i++)
    {
      auto idx = patch_idx[ip] + i;
      float charge = patch[idx];
      double weight = qweights[i % np + ip * MAX_P_SIZE];
      //FIXME: Now position space is continuous!!! (Like in Kokkos)
#pragma omp atomic update
      out[(p + i % np) + dim_p * (t + i / np)] += (float)(charge * weight);
#pragma omp atomic update
      out[(p + i % np + 1) + dim_p * (t + i / np)] += (float)(charge * (1.0 - weight));
    }
  }
  wend = omp_get_wtime();
  // std::cout << "yuhw: box_of_one: " << OpenMPArray::dump_2d_view(out,20) << std::endl;
  // std::cout << "yuhw: DEBUG: out: " << OpenMPArray::dump_2d_view(out,10000) << std::endl;
  g_get_charge_vec_time_part3 = wend - wstart;
  cout << "get_charge_matrix_openmp(): part3 running time : " << g_get_charge_vec_time_part3 << endl;
  cout << "get_charge_matrix_openmp(): set_sampling() running time : " << g_get_charge_vec_time_part4
       << ", counter : " << counter << endl;
  cout << "get_charge_matrix_openmp() : m_fluctuate : " << m_fluctuate << endl;

#pragma omp target exit data map(delete:gdata[0:npatches])
#pragma omp target exit data map(delete:tb,pb)
#pragma omp target exit data map(delete:np_vec[0:npatches],nt_vec[0:npatches],offsets[0:npatches*2])
#pragma omp target exit data map(delete:pvecs[0:npatches*MAX_P_SIZE],tvecs[0:npatches*MAX_T_SIZE],qweights[0:npatches*MAX_P_SIZE])
#pragma omp target exit data map(delete:patch_idx[0:npatches])
#pragma omp target exit data map(delete:patch[0:result])
#pragma omp target exit data map(delete:m_normals[0:size])
//#ifdef HAVE_CUDA_INC
//    cout << "get_charge_matrix_openmp() CUDA : set_sampling() part1 time : " << g_set_sampling_part1
//         << ", part2 (CUDA) time : " << g_set_sampling_part2 << endl;
//    cout << "GaussianDiffusion::sampling_CUDA() part3 time : " << g_set_sampling_part3
//         << ", part4 time : " << g_set_sampling_part4 << ", part5 time : " << g_set_sampling_part5 << endl;
//    cout << "GaussianDiffusion::sampling_CUDA() : g_total_sample_size : " << g_total_sample_size << endl;
//#else
//    cout << "set_sampling(): part1 time : " << g_set_sampling_part1
//         << ", part2 time : " << g_set_sampling_part2 << ", part3 time : " << g_set_sampling_part3 << endl;
//#endif
}

void GenOpenMP::BinnedDiffusion_transform::get_charge_matrix(std::vector<Eigen::SparseMatrix<float>* >& vec_spmatrix, std::vector<int>& vec_impact)
{
  const auto ib = m_pimpos.impact_binning();

  // map between reduced impact # to array # 
  std::map<int,int> map_redimp_vec;
  for (size_t i =0; i!= vec_impact.size(); i++){
    map_redimp_vec[vec_impact[i]] = int(i);
  }

  const auto rb = m_pimpos.region_binning();
  // map between impact # to channel #
  std::map<int, int> map_imp_ch;
  // map between impact # to reduced impact # 
  std::map<int, int> map_imp_redimp;

  //std::cout << ib.nbins() << " " << rb.nbins() << std::endl;
  for (int wireind=0;wireind!=rb.nbins();wireind++){
    int wire_imp_no = m_pimpos.wire_impact(wireind);
    std::pair<int,int> imps_range = m_pimpos.wire_impacts(wireind);
    for (int imp_no = imps_range.first; imp_no != imps_range.second; imp_no ++){
      map_imp_ch[imp_no] = wireind;
      map_imp_redimp[imp_no] = imp_no - wire_imp_no;
      
      //  std::cout << imp_no << " " << wireind << " " << wire_imp_no << " " << ib.center(imp_no) << " " << rb.center(wireind) << " " <<  ib.center(imp_no) - rb.center(wireind) << std::endl;
      // std::cout << imp_no << " " << map_imp_ch[imp_no] << " " << map_imp_redimp[imp_no] << std::endl;
    }
  }
  
  int min_imp = 0;
  int max_imp = ib.nbins();


   for (auto diff : m_diffs){
    //    std::cout << diff->depo()->time() << std::endl
    //diff->set_sampling(m_tbins, ib, m_nsigma, 0, m_calcstrat);
    diff->set_sampling(m_tbins, ib, m_nsigma, m_fluctuate, m_calcstrat);
    //counter ++;
    
    const auto patch = diff->patch();
    const auto qweight = diff->weights();

    const int poffset_bin = diff->poffset_bin();
    const int toffset_bin = diff->toffset_bin();

    const int np = patch.rows();
    const int nt = patch.cols();
    
    for (int pbin = 0; pbin != np; pbin++){
      int abs_pbin = pbin + poffset_bin;
      if (abs_pbin < min_imp || abs_pbin >= max_imp) continue;
      double weight = qweight[pbin];

      for (int tbin = 0; tbin!= nt; tbin++){
        int abs_tbin = tbin + toffset_bin;
        double charge = patch(pbin, tbin);

	      // std::cout << map_redimp_vec[map_imp_redimp[abs_pbin] ] << " " << map_redimp_vec[map_imp_redimp[abs_pbin]+1] << " " << abs_tbin << " " << map_imp_ch[abs_pbin] << std::endl;
	
	      vec_spmatrix.at(map_redimp_vec[map_imp_redimp[abs_pbin] ])->coeffRef(abs_tbin,map_imp_ch[abs_pbin]) += charge * weight; 
	      vec_spmatrix.at(map_redimp_vec[map_imp_redimp[abs_pbin]+1])->coeffRef(abs_tbin,map_imp_ch[abs_pbin]) += charge*(1-weight);
      }
    }

    

    
    diff->clear_sampling();
    // need to figure out wire #, time #, charge, and weight ...
   }

   for (auto it = vec_spmatrix.begin(); it!=vec_spmatrix.end(); it++){
     (*it)->makeCompressed();
   }
}

void GenOpenMP::BinnedDiffusion_transform::get_charge_matrix_openmp_noscan(float* out, size_t dim_p, size_t dim_t,
                                                                           std::vector<int>& vec_impact, const int start_pitch,
                                                                           const int start_tick)
{
  std::cout << "tw: get_charge_matrix_openmp_noscan\n";

  double wstart, wend ;
  wstart = omp_get_wtime();
  const auto ib = m_pimpos.impact_binning();

  // map between reduced impact # to array #

  std::map<int, int> map_redimp_vec;
  std::vector<std::unordered_map<long int, int> > vec_map_pair_pos;
  for(size_t i = 0; i != vec_impact.size(); i++) 
  {
    map_redimp_vec[vec_impact[i]] = int(i);
    std::unordered_map<long int, int> map_pair_pos;
    vec_map_pair_pos.push_back(map_pair_pos);
  }
  wend = omp_get_wtime();
  g_get_charge_vec_time_part1 = wend - wstart;
  cout << "get_charge_matrix_openmp(): part1 running time : " << g_get_charge_vec_time_part1 << endl;
  std::cout << "tw: is this step really necessary???\n";


  wstart = omp_get_wtime();
  const auto rb = m_pimpos.region_binning();
  // map between impact # to channel #
  std::map<int, int> map_imp_ch;
  // map between impact # to reduced impact #
  std::map<int, int> map_imp_redimp;

  std::cout << "tw: " << rb.nbins() << std::endl;
  for(int wireind = 0; wireind != rb.nbins(); wireind++) 
  {
    int wire_imp_no = m_pimpos.wire_impact(wireind);
    std::pair<int, int> imps_range = m_pimpos.wire_impacts(wireind);
    for(int imp_no = imps_range.first; imp_no != imps_range.second; imp_no++) 
    {
      map_imp_ch[imp_no] = wireind;
      map_imp_redimp[imp_no] = imp_no - wire_imp_no;
    }
  }

  int min_imp = 0;
  int max_imp = ib.nbins();
  int counter = 0;

  wend = omp_get_wtime();
  g_get_charge_vec_time_part2 = wend - wstart;
  cout << "get_charge_matrix_openmp(): part2 running time : " << g_get_charge_vec_time_part2 << endl;
  std::cout << "tw: is this step really necessary???\n";

  std::cout << "tw: I believe this is where things really get started. How did we get m_diffs?\n";
  wstart = omp_get_wtime();
  int npatches = m_diffs.size();
  GenOpenMP::GdData* gdata = (GenOpenMP::GdData*)malloc(sizeof(GenOpenMP::GdData) * npatches);

  //Can we do compute/data movement asynchronously? Necessary?
  int ii = 0;
  for (auto diff : m_diffs) 
  {
    gdata[ii].p_ct = diff->pitch_desc().center;
    gdata[ii].t_ct = diff->time_desc().center;
    gdata[ii].charge = diff->depo()->charge();
    gdata[ii].t_sigma = diff->time_desc().sigma;
    gdata[ii].p_sigma = diff->pitch_desc().sigma;
//    if(diff->pitch_desc().sigma == 0 || diff->time_desc().sigma == 0) 
//      std::cout<<"sigma-0 patch: " << ii << std::endl ;
    ii++;
  }

#pragma omp target enter data map(to:gdata[0:npatches])
  // make device friendly Binning  and copy tbin pbin over.
  // tw: What are these two? What is m_tbins and ib? Where is rb?
  GenOpenMP::DBin tb, pb;
  tb.nbins = m_tbins.nbins();
  tb.minval = m_tbins.min();
  tb.binsize = m_tbins.binsize();
  pb.nbins = ib.nbins();
  pb.minval = ib.min();
  pb.binsize = ib.binsize();

  //FIXME: Do we need to copy that to device?
#pragma omp target enter data map(to:tb,pb)

  // perform set_sampling_pre tasks on gpu
  // FIXME Think about if we can use target_alloc to generate data so that we don't need to generate the host vesion!
  // FIXME Check if we use () instead of [] all around the file!!!

  double t_temp = -omp_get_wtime();
  unsigned int* np_vec  = (unsigned int*)malloc(sizeof(unsigned int) * npatches);
  unsigned int* nt_vec  = (unsigned int*)malloc(sizeof(unsigned int) * npatches);
  unsigned int* offsets = (unsigned int*)malloc(sizeof(unsigned int) * npatches * 2);
  double* pvecs     = (double*)malloc(sizeof(double) * npatches * MAX_P_SIZE);
  double* tvecs     = (double*)malloc(sizeof(double) * npatches * MAX_T_SIZE);
  double* qweights  = (double*)malloc(sizeof(double) * npatches * MAX_P_SIZE);
  t_temp += omp_get_wtime();
  std::cout << "tw: Time for allocate np/t_vec, offsets, p/tvecs and qweights on host is " << t_temp * 1000.0 << " ms" << std::endl;

#pragma omp target enter data map(alloc:np_vec[0:npatches],nt_vec[0:npatches],offsets[0:npatches*2])
#pragma omp target enter data map(alloc:pvecs[0:npatches*MAX_P_SIZE],tvecs[0:npatches*MAX_T_SIZE],qweights[0:npatches*MAX_P_SIZE])

  //FIXME: Do we need to combine np_vec and nt_vec together, just like offsets, or do we need to split offsets, like np_vec and nt_vec???
  // Kernel for calculate nt_vec and np_vec and offsets for t and p for each gd
  int nsigma = m_nsigma;
#pragma omp target teams distribute parallel for simd
  for(int i=0; i<npatches; i++)
  {
    double t_s = gdata[i].t_ct - gdata[i].t_sigma * nsigma;
    double t_e = gdata[i].t_ct + gdata[i].t_sigma * nsigma;
    int t_ofb = max(int((t_s - tb.minval) / tb.binsize), 0);
    int ntss = min((int((t_e - tb.minval) / tb.binsize)) + 1, tb.nbins) - t_ofb;

    double p_s = gdata[i].p_ct - gdata[i].p_sigma * nsigma;
    double p_e = gdata[i].p_ct + gdata[i].p_sigma * nsigma;
    int p_ofb = max(int((p_s - pb.minval) / pb.binsize), 0);
    int npss = min((int((p_e - pb.minval) / pb.binsize)) + 1, pb.nbins) - p_ofb;

    //FIXME: Can we do assignment directly? Will that harm cache? Will that improve register?
    //Need to look at if this kernel is latency/register bound, and test if we can improve its register performance as above
    
    nt_vec[i] = ntss;
    np_vec[i] = npss;
    offsets[i] = t_ofb;
    offsets[npatches + i] = p_ofb;
  }

  //In this version, we do not calculate index for patch again, but allocate extra memory for each patches. 
  //This is equivalent to 
  //patch_idx[i] = i * MAX_PATCH_SIZE

  unsigned long result = MAX_PATCH_SIZE * npatches;

  // debug:
  std::cout << "total patch size: " << result << " WeightStrat: " << m_calcstrat << std::endl;
  
  //FIXME Should we save them in m_normals or create rd_normals and save them there?
  int size = (result+255) / 256 * 256;    //tw: This might not be necessary any more! 
  m_normals = (double*)malloc(sizeof(double) * size);
  unsigned long long seed = 2020;

#pragma omp target enter data map(alloc:m_normals[0:size])
#pragma omp target data use_device_ptr(m_normals)  
  omp_get_rng_normal_double(m_normals, size, 0.0, 1.0, seed);

  std::cout << "Create random numbers successfully!" << std::endl;

  // decide weight calculation
  int weightstrat = m_calcstrat;

  // each team resposible for 1 GD , kernel calculate pvecs and tvecs
  const double sqrt2 = sqrt(2.0);
  std::cout << " Start to compute pvecs and tvecs!" << std::endl;

  //FIXME: Can we improve this kernel by putting something in register or cache/shared memory?
  //It seems like it will generate three kernels, so putting them inside a single teams distribute is not very useful
  //Maybe the first and the last parallel for loop can be merged into a single kernel
  //In this kernel, the number of iters in inner loop is not pre-determinant. Shall we use num_threads to preset it?
#pragma omp target teams distribute
  for(int ip=0; ip<npatches; ip++)
  {
    double start_t = tb.minval + offsets[ip] * tb.binsize;
    double start_p = pb.minval + offsets[ip + npatches] * pb.binsize;
    int np = np_vec[ip];
    int nt = nt_vec[ip];

    if(np == 1)
      pvecs[ip * MAX_P_SIZE] = 1.0;
    else
    {
#pragma omp parallel for simd
      for(int ii=0; ii<np; ii++)
      {
        double step = pb.binsize;
        double factor = sqrt2 * gdata[ip].p_sigma;
        double x = (start_p + step * ii - gdata[ip].p_ct) / factor;
        double ef1 = 0.5 * erf(x);
        double ef2 = 0.5 * erf(x + step / factor);
        double val = ef2 - ef1;
        pvecs[ip * MAX_P_SIZE + ii] = val;
      }
    }
    
    if(nt == 1)
      tvecs[ip * MAX_T_SIZE] = 1.0;
    else
    {
#pragma omp parallel for simd
      for(int ii=0; ii<nt; ii++)
      {
        double step = tb.binsize;
        double factor = sqrt2 * gdata[ip].t_sigma;
        double x = (start_t + step * ii - gdata[ip].t_ct) / factor;
        double ef1 = 0.5 * erf(x);
        double ef2 = 0.5 * erf(x + step / factor);
        double val = ef2 - ef1;
        tvecs[ip * MAX_T_SIZE + ii] = val;
      }
    }
    
    if(weightstrat == 2)
    {
      if(gdata[ip].p_sigma == 0)
        qweights[ip * MAX_P_SIZE] = (start_p + pb.binsize - gdata[ip].p_ct) / pb.binsize;
      else
      {
#pragma omp parallel for simd
        for(int ii=0; ii<np; ii++)
        {
          double rel1 = (start_p + pb.binsize * ii - gdata[ip].p_ct) / gdata[ip].p_sigma;
          double rel2 = rel1 + pb.binsize / gdata[ip].p_sigma;
          double gaus1 = exp(-0.5 * rel1 * rel1);
          double gaus2 = exp(-0.5 * rel2 * rel2);
          double wt = -1.0 * gdata[ip].p_sigma / pb.binsize * (gaus2 - gaus1) / sqrt(2.0 * PI) / pvecs[ip * MAX_P_SIZE + ii]
                      + (gdata[ip].p_ct - (start_p + (ii + 1) * pb.binsize)) / pb.binsize;
          qweights[ip * MAX_P_SIZE + ii] = -wt;
        }
      }
    }
  }
  std::cout << "Compute pvecs, tvecs and qweights successfully!\n";

  // Allocate space for patches on device, we might also want to use target_alloc
  t_temp = -omp_get_wtime();
  float* patch = (float*)malloc(sizeof(float) * result);
  t_temp += omp_get_wtime();
  std::cout << "tw: Allocate space for patch on host takes " << t_temp * 1000 << " ms" << std::endl;

#pragma omp target enter data map(alloc:patch[0:result])

  wend = omp_get_wtime();
  set_sampling_bat_noscan(npatches, nt_vec, np_vec, pvecs, tvecs, patch, m_normals, gdata);
  wstart = omp_get_wtime();
  cout << "pr21 get_charge_matrix_openmp_noscan(): set_sampling_bat_noscan() no DtoH time " << wstart - wend << endl;
  std::cout << "tw: DEBUG: npatches: " << npatches << std::endl;
//  std::cout << "tw: DEBUG: np_vec: " << OpenMPArray::dump_1d_view(np_d,10000) << std::endl;
//  std::cout << "tw: DEBUG: nt_vec: " << OpenMPArray::dump_1d_view(nt_d,10000) << std::endl;
//  std::cout << "tw: DEBUG: offsets_d: " << OpenMPArray::dump_1d_view(offsets_d,10000) << std::endl;
//  std::cout << "tw: DEBUG: patch_idx: " << OpenMPArray::dump_1d_view(patch_idx,10000) << std::endl;
//  std::cout << "tw: DEBUG: patch_d: " << OpenMPArray::dump_1d_view(patch_d,10000) << std::endl;
//  std::cout << "tw: DEBUG: qweights_d: " << OpenMPArray::dump_1d_view(qweights_d,10000) << std::endl;

#pragma omp target teams distribute
  for(int ip=0; ip<npatches; ip++)
  {
    int np = np_vec[ip];
    int nt = nt_vec[ip];
    int p = offsets[npatches + ip] - start_pitch;
    int t = offsets[ip] - start_tick;
    int patch_size = np * nt;
    auto idx_st = ip * MAX_PATCH_SIZE;
#pragma omp parallel for simd
    for(int i=0; i<patch_size; i++)
    {
      auto idx = idx_st + i;
      float charge = patch[idx];
      double weight = qweights[i % np + ip * MAX_P_SIZE]; //As Chris says, % is expansive on GPU FIXME
      //FIXME: Now position space is continuous!!! (Like in Kokkos)
#pragma omp atomic update
      out[(p + i % np) + dim_p * (t + i / np)] += (float)(charge * weight);
#pragma omp atomic update
      out[(p + i % np + 1) + dim_p * (t + i / np)] += (float)(charge * (1.0 - weight));
    }
  }
  wend = omp_get_wtime();
  // std::cout << "yuhw: box_of_one: " << OpenMPArray::dump_2d_view(out,20) << std::endl;
  // std::cout << "yuhw: DEBUG: out: " << OpenMPArray::dump_2d_view(out,10000) << std::endl;
  g_get_charge_vec_time_part3 = wend - wstart;
  cout << "get_charge_matrix_openmp(): part3 running time : " << g_get_charge_vec_time_part3 << endl;
  cout << "get_charge_matrix_openmp(): set_sampling() running time : " << g_get_charge_vec_time_part4
       << ", counter : " << counter << endl;
  cout << "get_charge_matrix_openmp() : m_fluctuate : " << m_fluctuate << endl;

#pragma omp target exit data map(delete:gdata[0:npatches])
#pragma omp target exit data map(delete:tb,pb)
#pragma omp target exit data map(delete:np_vec[0:npatches],nt_vec[0:npatches],offsets[0:npatches*2])
#pragma omp target exit data map(delete:pvecs[0:npatches*MAX_P_SIZE],tvecs[0:npatches*MAX_T_SIZE],qweights[0:npatches*MAX_P_SIZE])
#pragma omp target exit data map(delete:patch[0:result])
#pragma omp target exit data map(delete:m_normals[0:size])
//#ifdef HAVE_CUDA_INC
//    cout << "get_charge_matrix_openmp() CUDA : set_sampling() part1 time : " << g_set_sampling_part1
//         << ", part2 (CUDA) time : " << g_set_sampling_part2 << endl;
//    cout << "GaussianDiffusion::sampling_CUDA() part3 time : " << g_set_sampling_part3
//         << ", part4 time : " << g_set_sampling_part4 << ", part5 time : " << g_set_sampling_part5 << endl;
//    cout << "GaussianDiffusion::sampling_CUDA() : g_total_sample_size : " << g_total_sample_size << endl;
//#else
//    cout << "set_sampling(): part1 time : " << g_set_sampling_part1
//         << ", part2 time : " << g_set_sampling_part2 << ", part3 time : " << g_set_sampling_part3 << endl;
//#endif
}

// a new function to generate the result for the entire frame ... 
void GenOpenMP::BinnedDiffusion_transform::get_charge_vec(std::vector<std::vector<std::tuple<int,int, double> > >& vec_vec_charge, std::vector<int>& vec_impact)
{
  assert(0 && "Error: The vec version is not implemented! Please make sure we are not calling this and are calling matrix version instead!\n");
}


//Note: This is a "device function", where all variables except npatches are on device. Make sure we have map them to device before calling this function!
void GenOpenMP::BinnedDiffusion_transform::set_sampling_bat(const unsigned long npatches,  
		const unsigned int* nt_d,
		const unsigned int* np_d, 
		const unsigned long* patch_idx , 
		const double* pvecs_d,
		const double* tvecs_d,
	        float*  patch_d,
	  const double* normals,
	  const GenOpenMP::GdData* gdata ) 
{
  std::cout << "Start set_sampling_bat" << std::endl;
  bool fl = false;
  if(m_fluctuate) fl = true;

  //FIXME: Do we want to optimize the code for host and device differently later? Now we just disable that!

#pragma omp target teams distribute
  for(int ip=0; ip<npatches; ip++)    
  {
    int np = np_d[ip];
    int nt = nt_d[ip];
    int patch_size = np*nt;
    unsigned long p0 = patch_idx[ip];

    double sum = 0.0;

#pragma omp parallel for reduction(+:sum)
    for(int ii=0; ii<patch_size; ii++)    
    {
      double v = pvecs_d[ip * MAX_P_SIZE + ii % np] * tvecs_d[ip * MAX_T_SIZE + ii / np];
      patch_d[ii + p0] = float(v);
      sum += v;
    }

    double charge = gdata[ip].charge;
    double charge_abs = abs(charge);
//    int charge_sign = charge < 0 ? -1 : 1;

#pragma omp parallel for  
    for(int ii=0; ii<patch_size; ii++)
    {
      patch_d[ii + p0] *= float(charge/sum);
    }

    //FIXME: we are debugging now, so turn rng off
//    if(fl)
    if(0)
    {
      int n = (int)charge_abs;
      sum = 0.0;

      //FIXME: type of patch_d is float, should we use double?
#pragma omp parallel for reduction(+:sum)
      for(int ii=0; ii<patch_size; ii++)
      {
        double p     = patch_d[ii + p0] / charge;
        double q     = 1.0 - p;
        double mu    = n * p;
        double sigma = sqrt(p * q * n) ;
        p            = normals[ii + p0] * sigma + mu ;
	      sum          += p ;
      }

#pragma omp parallel for  
      for(int ii=0; ii<patch_size; ii++)
      {
        patch_d[ii + p0] *= float(charge/sum);
      }
    }
  }
}

void GenOpenMP::BinnedDiffusion_transform::set_sampling_bat_noscan(const unsigned long npatches,  
		const unsigned int* nt_d,
		const unsigned int* np_d, 
		const double* pvecs_d,
		const double* tvecs_d,
	        float*  patch_d,
	  const double* normals,
	  const GenOpenMP::GdData* gdata ) 
{
  std::cout << "Start set_sampling_bat_noscan" << std::endl;
  bool fl = false;
  if(m_fluctuate) fl = true;

  //FIXME: Do we want to optimize the code for host and device differently later? Now we just disable that!

#pragma omp target teams distribute
  for(int ip=0; ip<npatches; ip++)    
  {
    int np = np_d[ip];
    int nt = nt_d[ip];
    int patch_size = np*nt;
    unsigned long p0 = ip * MAX_PATCH_SIZE;

    double sum = 0.0;

#pragma omp parallel for reduction(+:sum)
    for(int ii=0; ii<patch_size; ii++)    
    {
      double v = pvecs_d[ip * MAX_P_SIZE + ii % np] * tvecs_d[ip * MAX_T_SIZE + ii / np];
      patch_d[ii + p0] = float(v);
      sum += v;
    }

    double charge = gdata[ip].charge;
    double charge_abs = abs(charge);
//    int charge_sign = charge < 0 ? -1 : 1;

#pragma omp parallel for  
    for(int ii=0; ii<patch_size; ii++)
    {
      patch_d[ii + p0] *= float(charge/sum);
    }

    //FIXME: we are debugging now, so turn rng off
//    if(fl)
    if(0)
    {
      int n = (int)charge_abs;
      sum = 0.0;

      //FIXME: type of patch_d is float, should we use double?
#pragma omp parallel for reduction(+:sum)
      for(int ii=0; ii<patch_size; ii++)
      {
        double p     = patch_d[ii + p0] / charge;
        double q     = 1.0 - p;
        double mu    = n * p;
        double sigma = sqrt(p * q * n) ;
        p            = normals[ii + p0] * sigma + mu ;
	      sum          += p ;
      }

#pragma omp parallel for  
      for(int ii=0; ii<patch_size; ii++)
      {
        patch_d[ii + p0] *= float(charge/sum);
      }
    }
  }
}

static
std::pair<double,double> gausdesc_range(const std::vector<GenOpenMP::GausDesc> gds, double nsigma)
{
    int ncount = -1;
    double vmin=0, vmax=0;
    for (auto gd : gds) {
        ++ncount;

        const double lvmin = gd.center - gd.sigma*nsigma;
        const double lvmax = gd.center + gd.sigma*nsigma;
        if (!ncount) {
            vmin = lvmin;
            vmax = lvmax;
            continue;
        }
        vmin = std::min(vmin, lvmin);
        vmax = std::max(vmax, lvmax);
    }        
    return std::make_pair(vmin,vmax);
}

std::pair<double,double> GenOpenMP::BinnedDiffusion_transform::pitch_range(double nsigma) const
{
    std::vector<GenOpenMP::GausDesc> gds;
    for (auto diff : m_diffs) {
        gds.push_back(diff->pitch_desc());
    }
    return gausdesc_range(gds, nsigma);
}

std::pair<int,int> GenOpenMP::BinnedDiffusion_transform::impact_bin_range(double nsigma) const
{
    const auto ibins = m_pimpos.impact_binning();
    auto mm = pitch_range(nsigma);
    return std::make_pair(std::max(ibins.bin(mm.first), 0),
                          std::min(ibins.bin(mm.second)+1, ibins.nbins()));
}

std::pair<double,double> GenOpenMP::BinnedDiffusion_transform::time_range(double nsigma) const
{
    std::vector<GenOpenMP::GausDesc> gds;
    for (auto diff : m_diffs) {
        gds.push_back(diff->time_desc());
    }
    return gausdesc_range(gds, nsigma);
}

std::pair<int,int> GenOpenMP::BinnedDiffusion_transform::time_bin_range(double nsigma) const
{
    auto mm = time_range(nsigma);
    return std::make_pair(std::max(m_tbins.bin(mm.first),0),
                          std::min(m_tbins.bin(mm.second)+1, m_tbins.nbins()));
}
