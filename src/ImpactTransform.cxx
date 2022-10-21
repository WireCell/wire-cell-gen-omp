#include "WireCellGenOpenMP/ImpactTransform.h"
#include "WireCellUtil/Testing.h"
#include "WireCellUtil/FFTBestLength.h"

#include "WireCellGenOpenMP/OpenMPArray.h"

#include <iostream>  // debugging.
#include <omp.h>

double g_get_charge_vec_time = 0.0;
double g_get_charge_matrix_time = 0.0;
double g_fft_time = 0.0;

using namespace std;

using namespace WireCell;
GenOpenMP::ImpactTransform::ImpactTransform(IPlaneImpactResponse::pointer pir, BinnedDiffusion_transform& bd)
  : m_pir(pir)
  , m_bd(bd)
  , log(Log::logger("sim"))
  {}

bool GenOpenMP::ImpactTransform::transform_vector()
{
    double timer_transform = omp_get_wtime();
    // arrange the field response (210 in total, pitch_range/impact)
    // number of wires nwires ...
    m_num_group = std::round(m_pir->pitch() / m_pir->impact()) + 1;  // 11
    m_num_pad_wire = std::round((m_pir->nwires() - 1) / 2.);         // 10

    const auto pimpos = m_bd.pimpos();

    // std::cout << "yuhw: transform_vector: \n"
    // << " m_num_group: " << m_num_group
    // << " m_num_pad_wire: " << m_num_pad_wire
    // << " pitch: " << m_pir->pitch()
    // << " impact: " << m_pir->impact()
    // << std::endl;
    for (int i = 0; i != m_num_group; i++) {
        double rel_cen_imp_pos;
        if (i != m_num_group - 1) {
            rel_cen_imp_pos = -m_pir->pitch() / 2. + m_pir->impact() * i + 1e-9;
        }
        else {
            rel_cen_imp_pos = -m_pir->pitch() / 2. + m_pir->impact() * i - 1e-9;
        }
        m_vec_impact.push_back(std::round(rel_cen_imp_pos / m_pir->impact()));
        std::map<int, IImpactResponse::pointer> map_resp;  // already in freq domain

        for (int j = 0; j != m_pir->nwires(); j++) {
            map_resp[j - m_num_pad_wire] = m_pir->closest(rel_cen_imp_pos - (j - m_num_pad_wire) * m_pir->pitch());
            Waveform::compseq_t response_spectrum = map_resp[j - m_num_pad_wire]->spectrum();
            // std::cout
            // << " igroup: " << i
            // << " rel_cen_imp_pos: " << rel_cen_imp_pos
            // << " iwire: " << j
            // << " dist: " << rel_cen_imp_pos - (j - m_num_pad_wire) * m_pir->pitch()
            // << std::endl;
        }
        m_vec_map_resp.push_back(map_resp);

        std::vector<std::tuple<int, int, double> > vec_charge;  // ch, time, charge
        m_vec_vec_charge.push_back(vec_charge);
    }

    // now work on the charge part ...
    // trying to sampling ...
    double wstart = omp_get_wtime();
    m_bd.get_charge_vec(m_vec_vec_charge, m_vec_impact);
    double wend = omp_get_wtime();
    g_get_charge_vec_time += wend - wstart;
    log->debug("ImpactTransform::ImpactTransform() : get_charge_vec() Total running time : {}", g_get_charge_vec_time);

    std::pair<int, int> impact_range = m_bd.impact_bin_range(m_bd.get_nsigma());
    std::pair<int, int> time_range = m_bd.time_bin_range(m_bd.get_nsigma());

    int start_ch = std::floor(impact_range.first * 1.0 / (m_num_group - 1)) - 1;
    int end_ch = std::ceil(impact_range.second * 1.0 / (m_num_group - 1)) + 2;
    if ((end_ch - start_ch) % 2 == 1) end_ch += 1;
    int start_tick = time_range.first - 1;
    int end_tick = time_range.second + 2;
    if ((end_tick - start_tick) % 2 == 1) end_tick += 1;

    int npad_wire = 0;
    const size_t ntotal_wires = fft_best_length(end_ch - start_ch + 2 * m_num_pad_wire, 1);

    npad_wire = (ntotal_wires - end_ch + start_ch) / 2;
    m_start_ch = start_ch - npad_wire;
    m_end_ch = end_ch + npad_wire;

    int npad_time = m_pir->closest(0)->waveform_pad();
    const size_t ntotal_ticks = fft_best_length(end_tick - start_tick + npad_time);

    npad_time = ntotal_ticks - end_tick + start_tick;
    m_start_tick = start_tick;
    m_end_tick = end_tick + npad_time;

    wstart = omp_get_wtime();
    Array::array_xxc acc_data_f_w =
        Array::array_xxc::Zero(end_ch - start_ch + 2 * npad_wire, m_end_tick - m_start_tick);

    int num_double = (m_vec_vec_charge.size() - 1) / 2;

    // speed up version , first five
    for (int i = 0; i != num_double; i++) {
        Array::array_xxc c_data = Array::array_xxc::Zero(end_ch - start_ch + 2 * npad_wire, m_end_tick - m_start_tick);

        // fill normal order
        for (size_t j = 0; j != m_vec_vec_charge.at(i).size(); j++) {
            c_data(std::get<0>(m_vec_vec_charge.at(i).at(j)) + npad_wire - start_ch,
                   std::get<1>(m_vec_vec_charge.at(i).at(j)) - m_start_tick) +=
                std::get<2>(m_vec_vec_charge.at(i).at(j));
        }
        m_vec_vec_charge.at(i).clear();
        m_vec_vec_charge.at(i).shrink_to_fit();

        // fill reverse order
        int ii = num_double * 2 - i;
        for (size_t j = 0; j != m_vec_vec_charge.at(ii).size(); j++) {
            c_data(end_ch + npad_wire - 1 - std::get<0>(m_vec_vec_charge.at(ii).at(j)),
                   std::get<1>(m_vec_vec_charge.at(ii).at(j)) - m_start_tick) +=
                std::complex<float>(0, std::get<2>(m_vec_vec_charge.at(ii).at(j)));
        }
        m_vec_vec_charge.at(ii).clear();
        m_vec_vec_charge.at(ii).shrink_to_fit();

        // Do FFT on time
        c_data = Array::dft_cc(c_data, 0);
        // Do FFT on wire
        c_data = Array::dft_cc(c_data, 1);

        // std::cout << i << std::endl;
        {
            Array::array_xxc resp_f_w =
                Array::array_xxc::Zero(end_ch - start_ch + 2 * npad_wire, m_end_tick - m_start_tick);
            {
                Waveform::compseq_t rs1 = m_vec_map_resp.at(i)[0]->spectrum();
                // do a inverse FFT
                Waveform::realseq_t rs1_t = Waveform::idft(rs1);
                // pick the first xxx ticks
                Waveform::realseq_t rs1_reduced(m_end_tick - m_start_tick, 0);
                for (int icol = 0; icol != m_end_tick - m_start_tick; icol++) {
                    if (icol >= int(rs1_t.size())) break;
                    rs1_reduced.at(icol) = rs1_t[icol];
                }
                // do a FFT
                rs1 = Waveform::dft(rs1_reduced);

                for (int icol = 0; icol != m_end_tick - m_start_tick; icol++) {
                    resp_f_w(0, icol) = rs1[icol];
                }
            }

            for (int irow = 0; irow != m_num_pad_wire; irow++) {
                Waveform::compseq_t rs1 = m_vec_map_resp.at(i)[irow + 1]->spectrum();
                Waveform::realseq_t rs1_t = Waveform::idft(rs1);
                Waveform::realseq_t rs1_reduced(m_end_tick - m_start_tick, 0);
                for (int icol = 0; icol != m_end_tick - m_start_tick; icol++) {
                    if (icol >= int(rs1_t.size())) break;
                    rs1_reduced.at(icol) = rs1_t[icol];
                }
                rs1 = Waveform::dft(rs1_reduced);
                Waveform::compseq_t rs2 = m_vec_map_resp.at(i)[-irow - 1]->spectrum();
                Waveform::realseq_t rs2_t = Waveform::idft(rs2);
                Waveform::realseq_t rs2_reduced(m_end_tick - m_start_tick, 0);
                for (int icol = 0; icol != m_end_tick - m_start_tick; icol++) {
                    if (icol >= int(rs2_t.size())) break;
                    rs2_reduced.at(icol) = rs2_t[icol];
                }
                rs2 = Waveform::dft(rs2_reduced);
                for (int icol = 0; icol != m_end_tick - m_start_tick; icol++) {
                    resp_f_w(irow + 1, icol) = rs1[icol];
                    resp_f_w(end_ch - start_ch - 1 - irow + 2 * npad_wire, icol) = rs2[icol];
                }
            }
            // std::cout << " vector resp: " << i << " : " << Array::idft_cr(resp_f_w, 0) << std::endl;

            // Do FFT on wire for response // slight larger
            resp_f_w = Array::dft_cc(resp_f_w, 1);  // Now becomes the f and f in both time and wire domain ...
            // multiply them together
            c_data = c_data * resp_f_w;
        }

        // Do inverse FFT on wire
        c_data = Array::idft_cc(c_data, 1);

        // Add to wire result in frequency
        acc_data_f_w += c_data;
    }


    log->info("yuhw: start-end {}-{} {}-{}",m_start_ch, m_end_ch, m_start_tick, m_end_tick);

    // central region ...
    {
        int i = num_double;
        // fill response array in frequency domain

        Array::array_xxc data_f_w;
        {
            Array::array_xxf data_t_w =
                Array::array_xxf::Zero(end_ch - start_ch + 2 * npad_wire, m_end_tick - m_start_tick);
            // fill charge array in time-wire domain // slightly larger
            for (size_t j = 0; j != m_vec_vec_charge.at(i).size(); j++) {
                data_t_w(std::get<0>(m_vec_vec_charge.at(i).at(j)) + npad_wire - start_ch,
                         std::get<1>(m_vec_vec_charge.at(i).at(j)) - m_start_tick) +=
                    std::get<2>(m_vec_vec_charge.at(i).at(j));
            }
            // m_decon_data = data_t_w; // DEBUG tap data_t_w out
            m_vec_vec_charge.at(i).clear();
            m_vec_vec_charge.at(i).shrink_to_fit();

            // Do FFT on time
            data_f_w = Array::dft_rc(data_t_w, 0);
            // Do FFT on wire
            data_f_w = Array::dft_cc(data_f_w, 1);
        }

        {
            Array::array_xxc resp_f_w =
                Array::array_xxc::Zero(end_ch - start_ch + 2 * npad_wire, m_end_tick - m_start_tick);

            {
                Waveform::compseq_t rs1 = m_vec_map_resp.at(i)[0]->spectrum();

                // do a inverse FFT
                Waveform::realseq_t rs1_t = Waveform::idft(rs1);
                // pick the first xxx ticks
                Waveform::realseq_t rs1_reduced(m_end_tick - m_start_tick, 0);
                for (int icol = 0; icol != m_end_tick - m_start_tick; icol++) {
                    if (icol >= int(rs1_t.size())) break;
                    rs1_reduced.at(icol) = rs1_t[icol];
                }
                // do a FFT
                rs1 = Waveform::dft(rs1_reduced);

                for (int icol = 0; icol != m_end_tick - m_start_tick; icol++) {
                    resp_f_w(0, icol) = rs1[icol];
                }
            }
            for (int irow = 0; irow != m_num_pad_wire; irow++) {
                Waveform::compseq_t rs1 = m_vec_map_resp.at(i)[irow + 1]->spectrum();
                Waveform::realseq_t rs1_t = Waveform::idft(rs1);
                Waveform::realseq_t rs1_reduced(m_end_tick - m_start_tick, 0);
                for (int icol = 0; icol != m_end_tick - m_start_tick; icol++) {
                    if (icol >= int(rs1_t.size())) break;
                    rs1_reduced.at(icol) = rs1_t[icol];
                }
                rs1 = Waveform::dft(rs1_reduced);
                Waveform::compseq_t rs2 = m_vec_map_resp.at(i)[-irow - 1]->spectrum();
                Waveform::realseq_t rs2_t = Waveform::idft(rs2);
                Waveform::realseq_t rs2_reduced(m_end_tick - m_start_tick, 0);
                for (int icol = 0; icol != m_end_tick - m_start_tick; icol++) {
                    if (icol >= int(rs2_t.size())) break;
                    rs2_reduced.at(icol) = rs2_t[icol];
                }
                rs2 = Waveform::dft(rs2_reduced);
                for (int icol = 0; icol != m_end_tick - m_start_tick; icol++) {
                    resp_f_w(irow + 1, icol) = rs1[icol];
                    resp_f_w(end_ch - start_ch - 1 - irow + 2 * npad_wire, icol) = rs2[icol];
                }
            }
            // std::cout << " vector resp: " << i << " : " << Array::idft_cr(resp_f_w, 0) << std::endl;
            // Do FFT on wire for response // slight larger
            resp_f_w = Array::dft_cc(resp_f_w, 1);  // Now becomes the f and f in both time and wire domain ...
            // multiply them together
            data_f_w = data_f_w * resp_f_w;
        }

        // Do inverse FFT on wire
        data_f_w = Array::idft_cc(data_f_w, 1);

        // Add to wire result in frequency
        acc_data_f_w += data_f_w;
    }
    // m_decon_data = Array::idft_cr(acc_data_f_w, 0); // DEBUG only central
    acc_data_f_w = Array::idft_cc(acc_data_f_w, 0);  //.block(npad_wire,0,nwires,nsamples);
    Array::array_xxf real_m_decon_data = acc_data_f_w.real();
    Array::array_xxf img_m_decon_data = acc_data_f_w.imag().colwise().reverse();
    m_decon_data = real_m_decon_data + img_m_decon_data;

    double timer_fft = omp_get_wtime() - wstart;
    log->debug("ImpactTransform::transform_vector: FFT: {}", timer_fft);
    timer_transform = omp_get_wtime() - timer_transform;
    log->debug("ImpactTransform::transform_vector: Total: {}", timer_transform);
    
    log->debug("ImpactTransform::transform_vector: # of channels: {} # of ticks: {}", m_decon_data.rows(), m_decon_data.cols());
    log->debug("transform_vector: m_decon_data.sum(): {}", m_decon_data.sum());

    return true;
}

bool GenOpenMP::ImpactTransform::transform_matrix()
{
  std::cout << "TW_LOG_MESSAGE: Start doing transform_matrix()" << std::endl;

  double timer_transform = -omp_get_wtime();
  double td0(0.0), td1(0.0);
  double wstart, wend, t_temp, fft_time;

  wstart = omp_get_wtime();
  // arrange the field response (210 in total, pitch_range/impact)
  // number of wires nwires ...
  m_num_group = std::round(m_pir->pitch() / m_pir->impact()) + 1;  // 11
  m_num_pad_wire = std::round((m_pir->nwires() - 1) / 2.);         // 10

  std::pair<int, int> impact_range = m_bd.impact_bin_range(m_bd.get_nsigma());
  std::pair<int, int> time_range = m_bd.time_bin_range(m_bd.get_nsigma());

  int start_ch = std::floor(impact_range.first * 1.0 / (m_num_group - 1)) - 1;
  int end_ch = std::ceil(impact_range.second * 1.0 / (m_num_group - 1)) + 2;
  if ((end_ch - start_ch) % 2 == 1) end_ch += 1;
  int start_pitch = impact_range.first - 1;
  int end_pitch = impact_range.second + 2;
  if ((end_pitch - start_pitch) % 2 == 1) end_pitch += 1;
  int start_tick = time_range.first - 1;
  int end_tick = time_range.second + 2;
  if ((end_tick - start_tick) % 2 == 1) end_tick += 1;

  int npad_wire = 0;
  const size_t ntotal_wires = fft_best_length(end_ch - start_ch + 2 * m_num_pad_wire, 1);
  std::cout << "TW_LOG_MESSAGE: ntotal_wires = " << ntotal_wires << std::endl;
  npad_wire = (ntotal_wires - end_ch + start_ch) / 2;
  m_start_ch = start_ch - npad_wire;
  m_end_ch = end_ch + npad_wire;
  std::cout << "TW_LOG_MESSAGE: m_start_ch = " << m_start_ch << "   m_end_ch = " << m_end_ch << std::endl;

  int npad_time = m_pir->closest(0)->waveform_pad();
  const size_t ntotal_ticks = fft_best_length(end_tick - start_tick + npad_time);
  std::cout << "TW_LOG_MESSAGE: ntotal_ticks = " << ntotal_ticks << std::endl;
  npad_time = ntotal_ticks - end_tick + start_tick;
  m_start_tick = start_tick;
  m_end_tick = end_tick + npad_time;
  std::cout << "TW_LOG_MESSAGE: m_start_tick = "  << m_start_tick << "   m_end_tick = " << m_end_tick << std::endl;

  int npad_pitch = 0;
  const size_t ntotal_pitches = fft_best_length((end_ch - start_ch + 2 * npad_wire)*(m_num_group - 1), 1);
  std::cout << "TW_LOG_MESSAGE: m_num_group = " << m_num_group << "   ntotal_pitches = " << ntotal_pitches << std::endl;
  npad_pitch = (ntotal_pitches - end_pitch + start_pitch) / 2;
  start_pitch = start_pitch - npad_pitch;
  end_pitch = end_pitch + npad_pitch;
  std::cout << "TW_LOG_MESSAGE: start_pitch = " << start_pitch << "   end_pitch = " << end_pitch << std::endl;
  
  wend = omp_get_wtime();
  std::cout << "TW_TIMING_MESSAGE: transform matrix, part 1 (best length) takes " << (wend - wstart) * 1000.0 << " ms" << std::endl;
  td0 += wend - wstart;

  // now work on the charge part ...
  // trying to sampling ...
  wstart = omp_get_wtime();
  t_temp = -omp_get_wtime();

  size_t dim_p = end_pitch - start_pitch;
  size_t dim_t = m_end_tick - m_start_tick;
  std::cout << "TW_LOG_MESSAGE: dim_p = " << dim_p << "\t  dim_t = " << dim_t << std::endl;
  
  float *f_data = (float*)malloc(sizeof(float) * dim_p * dim_t);

  t_temp += omp_get_wtime();
  std::cout << "TW_TIMING_MESSAGE: time for malloc f_data on host is " << t_temp * 1000.0 << " ms" << std::endl;

  t_temp = -omp_get_wtime();

#pragma omp target enter data map(alloc: f_data[0:dim_p*dim_t])   

#pragma omp target teams distribute parallel for simd
  for(int i=0; i<dim_p*dim_t; i++)
    f_data[i] = 0.0;

  t_temp += omp_get_wtime();
  std::cout << "TW_TIMING_MESSAGE: time for alloc f_data on device is " << t_temp * 1000.0 << " ms" << std::endl;
  t_temp = -omp_get_wtime();

//FIXME: Notice f_data is not initialized in OpenMP code
//  m_bd.get_charge_matrix_openmp(f_data, dim_p, dim_t, m_vec_impact, start_pitch, m_start_tick);
  m_bd.get_charge_matrix_openmp_noscan(f_data, dim_p, dim_t, m_vec_impact, start_pitch, m_start_tick);

  t_temp += omp_get_wtime();
  std::cout << "TW_TIMING_MESSAGE: time for get_charge_matrix_openmp is " << t_temp * 1000.0 << " ms" << std::endl;
  td1 += t_temp;
  g_get_charge_matrix_time += t_temp;
  log->debug("ImpactTransform::ImpactTransform() : get_charge_matrix() Total_Time :  {}", g_get_charge_matrix_time);

  t_temp = -omp_get_wtime();
  fft_time = -omp_get_wtime();

  std::complex<float> *data_c = (std::complex<float>*)malloc(sizeof(std::complex<float>) * dim_p * dim_t);
#pragma omp target enter data map(alloc: data_c[0:dim_p*dim_t])    

#pragma omp target teams distribute parallel for simd
  for(int i=0; i<dim_p*dim_t; i++)
    data_c[i] = 0.0;

  size_t acc_dim_p = end_ch - start_ch + 2 * npad_wire;
  size_t acc_dim_t = m_end_tick - m_start_tick;
  std::cout << "TW_LOG_MESSAGE: acc_dim_p = " << acc_dim_p << "\t  acc_dim_t = " << acc_dim_t << std::endl;

  std::complex<float> *acc_data_f_w = (std::complex<float>*)malloc(sizeof(std::complex<float>) * acc_dim_p * acc_dim_t);
#pragma omp target enter data map(alloc: acc_data_f_w[0:acc_dim_p*acc_dim_t])    
  float *acc_data_t_w = (float*)malloc(sizeof(float) * acc_dim_p * acc_dim_t); 
#pragma omp target enter data map(alloc: acc_data_t_w[0:acc_dim_p*acc_dim_t])    

  t_temp += omp_get_wtime();
  std::cout << "TW_TIMING_MESSAGE: time for setup acc_data and data_c is " << t_temp * 1000.0 << " ms" << std::endl;
  t_temp = -omp_get_wtime();

  //FIXME: Can we combine the two dft into a single one? Will that improve time?

//tw: be careful: If uncomment these two lines, and comment out the next two lines, we switch the two statements about fft, and currently it gives the wrong answer!!!!
//  OpenMPArray::dft_rc(data_c, f_data, dim_p, dim_t, 1);
//  OpenMPArray::dft_cc(data_c, data_c, dim_p, dim_t, 0);

  OpenMPArray::dft_rc(data_c, f_data, dim_p, dim_t, 0);
  OpenMPArray::dft_cc(data_c, data_c, dim_p, dim_t, 1);

//  OpenMPArray::dft_rc_2d_test(data_c, f_data, dim_t, dim_p);   //Be careful about the relative order of two dimensions!!
  
  t_temp += omp_get_wtime();
  std::cout << "TW_TIMING_MESSAGE: time for first dft_rc/cc on data_c and f_data is " << t_temp * 1000.0 << " ms" << std::endl;
  t_temp = -omp_get_wtime();

#pragma omp target exit data map(delete:f_data[0:dim_p*dim_t])
  t_temp += omp_get_wtime();
  std::cout << "TW_TIMING_MESSAGE: time for deallocating f_data on GPU is " << t_temp * 1000.0 << " ms" << std::endl;

  wend = omp_get_wtime();
  std::cout << "TW_TIMING_MESSAGE: transform matrix, part 2 (2 dft and prep) takes " << (wend - wstart) * 1000.0 << " ms" << std::endl;

  wstart = omp_get_wtime();

  std::complex<float> *resp_f_w_k = (std::complex<float>*)malloc(sizeof(std::complex<float>) * dim_p * dim_t);
#pragma omp target enter data map(alloc: resp_f_w_k[0:dim_p*dim_t])         

#pragma omp target teams distribute parallel for simd
  for(int i=0; i<dim_p*dim_t; i++)
    resp_f_w_k[i] = 0.0;



//  This is the new version, where time size is set to be the smaller between m_start_tick - m_end_tick and sp_f.size()
//  Convolution with Field Response
  {
  // fine grained resp. matrix
    t_temp = -omp_get_wtime();

    std::cout << "TW_LOG_MESSAGE: Start doing transform_matrix(), dft on resp matrix" << std::endl;

    int nimpact = m_pir->nwires() * (m_num_group - 1);
    assert(dim_p >= nimpact && "Error: dim_p is smaller than nimpact!\n");

    double max_impact = (0.5 + m_num_pad_wire) * m_pir->pitch();
    int *idx_resp = (int*)malloc(sizeof(int) * nimpact);
    auto ip0 = m_pir->closest(-max_impact + 0.01 * m_pir->impact());
    int sp_size = ip0->spectrum().size();
    int fillsize = min(sp_size, int(dim_t));

    std::complex<float> *sp_fs = (std::complex<float>*)malloc(sizeof(std::complex<float>) * nimpact * sp_size);

    t_temp += omp_get_wtime(); 
    std::cout << "TW_TIMING_MESSAGE: time for dft on resp, step 1 (prep) is " << t_temp * 1000.0 << " ms" << std::endl;
    t_temp = -omp_get_wtime();

    for(int jimp = 0; jimp < nimpact; ++jimp) 
    {
        float impact = -max_impact + jimp * m_pir->impact();
        
        // to avoid exess of boundary.
        // central is < 0, after adding 0.01 * m_pir->impact(), it is > 0
        impact += impact < 0 ? 0.01 * m_pir->impact() : -0.01 * m_pir->impact();

        auto ip = m_pir->closest(impact);
        Waveform::compseq_t sp_f = ip->spectrum();

        memcpy((void*)&sp_fs[jimp*sp_size], (void*)&sp_f[0], sp_size*2*sizeof(float));

        int idx = impact < 0.1 * m_pir->impact() ? std::round(nimpact / 2.) - jimp
                             : dim_p - (jimp - std::round(nimpact / 2.));
        idx_resp[jimp] = idx;
    }


    t_temp += omp_get_wtime();
    std::cout << "TW_TIMING_MESSAGE: time for dft on resp, step 2 (prep resp) is " << t_temp * 1000.0 << " ms" << std::endl;
    t_temp = -omp_get_wtime();

    //be really careful: Now the memory layout is p-major instead of the earlier t-major
#pragma omp target enter data map(to: sp_fs[0:sp_size*nimpact])        
#pragma omp target enter data map(to: idx_resp[0:nimpact])



    t_temp += omp_get_wtime();
    std::cout << "TW_TIMING_MESSAGE: time for dft on resp, step 3 (data_cpy_h2d) is " << t_temp * 1000.0 << " ms" << std::endl;
    t_temp = -omp_get_wtime();

    float *sp_ts = (float*)malloc(sizeof(float) * nimpact * sp_size);
#pragma omp target enter data map(alloc: sp_ts[0:sp_size*nimpact])       

    t_temp += omp_get_wtime();
    std::cout << "TW_TIMING_MESSAGE: time for dft on resp, step 4 (prep for idft) is " << t_temp * 1000.0 << " ms" << std::endl;
    t_temp = -omp_get_wtime();

    OpenMPArray::idft_cr(sp_ts, sp_fs, sp_size, nimpact, 1);



    t_temp += omp_get_wtime();
    std::cout << "TW_TIMING_MESSAGE: time for dft on resp, step 5 (idft) is " << t_temp * 1000.0 << " ms" << std::endl;
    t_temp = -omp_get_wtime();

    std::complex<float> *resp_redu = (std::complex<float>*)malloc(sizeof(std::complex<float>) * nimpact * fillsize);
#pragma omp target enter data map(alloc: resp_redu[0:fillsize*nimpact])        

#pragma omp target teams distribute parallel for simd
    for(int i=0; i<fillsize*nimpact; i++)
      resp_redu[i] = 0.0;







    t_temp += omp_get_wtime();
    std::cout << "TW_TIMING_MESSAGE: time for dft on resp, step 6 (prep for dft) is " << t_temp * 1000.0 << " ms" << std::endl;
    t_temp = -omp_get_wtime();

    if(fillsize < sp_size)
    {
      assert(0 && "Error: tw: I have not implemented this branch! Need to think about what is the best way to do this!\n");
    }
    else
    {
      OpenMPArray::dft_rc(resp_redu, sp_ts, fillsize, nimpact, 1);
 
      t_temp += omp_get_wtime();
      std::cout << "TW_TIMING_MESSAGE: time for dft on resp, step 7 (dft) is " << t_temp * 1000.0 << " ms" << std::endl;
      t_temp = -omp_get_wtime();

      std::cout << "TW_LOG_MESSAGE: nimpact = " << nimpact << "\t  fillsize = " << fillsize << std::endl;

#pragma omp target teams distribute parallel for simd collapse(2)
      for(int i1=0; i1<nimpact; i1++)
      {
        for(int i0=0; i0<fillsize; i0++)
        {
          resp_f_w_k[idx_resp[i1] + i0 * dim_p] = resp_redu[i0 + i1 * fillsize];
        }
      }

      t_temp += omp_get_wtime();
      std::cout << "TW_TIMING_MESSAGE: time for dft on resp, step 8 (pseudo-transpose) is " << t_temp * 1000.0 << " ms" << std::endl;
      t_temp = -omp_get_wtime();
    }
 
    OpenMPArray::dft_cc(resp_f_w_k, resp_f_w_k, dim_p, dim_t, 1);

    t_temp += omp_get_wtime();
    std::cout << "TW_TIMING_MESSAGE: time for dft on resp, step 9 (final dft) is " << t_temp * 1000.0 << " ms" << std::endl;
    t_temp = -omp_get_wtime();

    wend = omp_get_wtime();
    std::cout << "TW_TIMING_MESSAGE: time for dft on resp total takes " << (wend - wstart) * 1000.0 << " ms" << std::endl;

#pragma omp target teams distribute parallel for simd
    for(int i=0; i<dim_t * dim_p; i++)
    {
      data_c[i] *= resp_f_w_k[i];
    }

    t_temp += omp_get_wtime();
    std::cout << "TW_TIMING_MESSAGE: time for resp(f) * data(f) is " << t_temp * 1000.0 << " ms" << std::endl;
    t_temp = -omp_get_wtime();

    // transfer wire to time domain
    OpenMPArray::idft_cc(data_c, data_c, dim_p, dim_t, 1);

    t_temp += omp_get_wtime();
    std::cout << "TW_TIMING_MESSAGE: time for data(f) idft is " << t_temp * 1000.0 << " ms" << std::endl;
    t_temp = -omp_get_wtime();

    // extract M(channel) from M(impact)
#pragma omp target teams distribute parallel for simd collapse(2)        
    for(int i1=0; i1<acc_dim_t; i1++)
    {
      for(int i0=0; i0<acc_dim_p; i0++)
      {
        //FIXME: correctness check, also notice memory access is not coalescing
        acc_data_f_w[i0 + (acc_dim_p * i1)] = data_c[(i0 + 1) * 10 + (dim_p * i1)];
      }
    }
    
    t_temp += omp_get_wtime();
    std::cout << "TW_TIMING_MESSAGE: time for getting acc_data_f_w by sampling data_c is " << t_temp * 1000.0 << " ms" << std::endl;
  }

  t_temp = -omp_get_wtime();

  OpenMPArray::idft_cr(acc_data_t_w, acc_data_f_w, acc_dim_p, acc_dim_t, 0);

  t_temp += omp_get_wtime();
  std::cout << "TW_TIMING_MESSAGE: time for idft on acc_data_t_w is " << t_temp * 1000.0 << " ms" << std::endl;
  t_temp = -omp_get_wtime();

#pragma omp target exit data map(from:acc_data_t_w[0:acc_dim_p*acc_dim_t])
//    //tw: DEBUG start here!
//    std::cout << "************************" << std::endl;
//    for(int ip=0; ip<acc_dim_p; ip++)
//    {
//      for(int it=0; it<acc_dim_t; it++)
//      {
//        if(abs(acc_data_t_w[ip + it * acc_dim_p]) > 1e-8)
//          std::cout << "tw: DEBUG: ip = " << ip << "  it = " << it << "  acc_data_t_w_eigen data = " << acc_data_t_w[ip + it * acc_dim_p] << std::endl;
//      }
//    }
//    std::cout << "************************" << std::endl;
//    //tw: DEBUG end here!
    
  Eigen::Map<Eigen::ArrayXXf> acc_data_t_w_eigen(acc_data_t_w, acc_dim_p, acc_dim_t);
  m_decon_data = acc_data_t_w_eigen; // FIXME: reduce this copy

  fft_time += omp_get_wtime();
  g_fft_time += fft_time;

  t_temp += omp_get_wtime();
  std::cout << "TW_TIMING_MESSAGE: time for copying back to cpu and then to Eigen is " << t_temp * 1000.0 << " ms" << std::endl;
  t_temp = -omp_get_wtime();
 
#pragma omp target exit data map(delete: data_c[0:dim_p*dim_t])
#pragma omp target exit data map(delete: resp_f_w_k[0:dim_p*dim_t])    
#pragma omp target exit data map(delete: acc_data_f_w[0:acc_dim_p*acc_dim_t])

  t_temp += omp_get_wtime();
  std::cout << "TW_TIMING_MESSAGE: time for deallocating on device is " << t_temp * 1000.0 << " ms" << std::endl;
  t_temp = -omp_get_wtime();

  timer_transform += omp_get_wtime();
  log->debug("ImpactTransform::transform_matrix: FFT: {}", fft_time);
  log->debug("ImpactTransform::transform_matrix: Total: {}", timer_transform * 1000.0);

  log->debug("ImpactTransform::transform_matrix: # of channels: {} # of ticks: {}", m_decon_data.rows(), m_decon_data.cols());
  log->debug("ImpactTransform::transform_matrix: m_decon_data.sum(): {}", m_decon_data.sum());

  std::cout<<"Tranform_matrix_p0_Time: "<<td0 <<std::endl;
  std::cout<<"get_charge_matrix_Time: "<<td1 <<std::endl;
  std::cout<<"FFTs_Time: "<< fft_time <<std::endl;

  log->debug("ImpactTransform::transform_matrix: Total_FFT_Time: {}", g_fft_time);
  return true;
}

//bool GenOpenMP::ImpactTransform::transform_matrix_old()
//{
//    double timer_transform = omp_get_wtime();
//    // arrange the field response (210 in total, pitch_range/impact)
//    // number of wires nwires ...
//    m_num_group = std::round(m_pir->pitch() / m_pir->impact()) + 1;  // 11
//    m_num_pad_wire = std::round((m_pir->nwires() - 1) / 2.);         // 10
//
//    std::pair<int, int> impact_range = m_bd.impact_bin_range(m_bd.get_nsigma());
//    std::pair<int, int> time_range = m_bd.time_bin_range(m_bd.get_nsigma());
//
//    int start_ch = std::floor(impact_range.first * 1.0 / (m_num_group - 1)) - 1;
//    int end_ch = std::ceil(impact_range.second * 1.0 / (m_num_group - 1)) + 2;
//    if ((end_ch - start_ch) % 2 == 1) end_ch += 1;
//    int start_pitch = impact_range.first - 1;
//    int end_pitch = impact_range.second + 2;
//    if ((end_pitch - start_pitch) % 2 == 1) end_pitch += 1;
//    int start_tick = time_range.first - 1;
//    int end_tick = time_range.second + 2;
//    if ((end_tick - start_tick) % 2 == 1) end_tick += 1;
//
//    int npad_wire = 0;
//    const size_t ntotal_wires = fft_best_length(end_ch - start_ch + 2 * m_num_pad_wire, 1);
//    std::cout << "ntotal_wires = " << ntotal_wires << std::endl;
//    npad_wire = (ntotal_wires - end_ch + start_ch) / 2;
//    m_start_ch = start_ch - npad_wire;
//    m_end_ch = end_ch + npad_wire;
//    std::cout << "m_start_ch = " << m_start_ch << "   m_end_ch = " << m_end_ch << std::endl;
//
//    int npad_time = m_pir->closest(0)->waveform_pad();
//    const size_t ntotal_ticks = fft_best_length(end_tick - start_tick + npad_time);
//    std::cout << "ntotal_ticks = " << ntotal_ticks << std::endl;
//    npad_time = ntotal_ticks - end_tick + start_tick;
//    m_start_tick = start_tick;
//    m_end_tick = end_tick + npad_time;
//    std::cout << "m_start_tick = "  << m_start_tick << "   m_end_tick = " << m_end_tick << std::endl;
//
//    int npad_pitch = 0;
//    const size_t ntotal_pitches = fft_best_length((end_ch - start_ch + 2 * npad_wire)*(m_num_group - 1), 1);
//    std::cout << "ntotal_pitches = " << ntotal_pitches << std::endl;
//    npad_pitch = (ntotal_pitches - end_pitch + start_pitch) / 2;
//    start_pitch = start_pitch - npad_pitch;
//    end_pitch = end_pitch + npad_pitch;
//    std::cout << "start_pitch = " << start_pitch << "   end_pitch = " << end_pitch << std::endl;
//    
//    // std::cout << "yuhw: "
//    // << " channel: { " << start_ch << ", " << end_ch << " } "
//    // << " pitch: { " << start_pitch << ", " << end_pitch << " }"
//    // << " tick: { " << m_start_tick << ", " << m_end_tick << " }\n";
//
//    // now work on the charge part ...
//    // trying to sampling ...
//    double wstart = omp_get_wtime();
//
//
////---------------------------------------OpenMP code starts here------------------------------------------
////format is ////    Kokkos code
////              OpenMP code
//
//
//////    KokkosArray::array_xxf f_data = KokkosArray::Zero<KokkosArray::array_xxf>(end_pitch - start_pitch, m_end_tick - m_start_tick);;
//    size_t dim_p = end_pitch - start_pitch;
//    size_t dim_t = m_end_tick - m_start_tick;
//    double t_temp = -omp_get_wtime();
//    float *f_data = (float*)malloc(sizeof(float) * dim_p * dim_t);
//    t_temp += omp_get_wtime();
//    std::cout << "Time for malloc f_data on host is " << t_temp << std::endl;
//    std::cout << "dim_p = " << dim_p << "\t dim_t = " << dim_t << std::endl;
//#pragma omp target enter data map(alloc: f_data[0:dim_p*dim_t])    
//
//    double t_get_charge_matrix = -omp_get_wtime();
//////    m_bd.get_charge_matrix_openmp(f_data, m_vec_impact, start_pitch, m_start_tick);
////  Notice f_data is not initialized in OpenMP code FIXME
////    m_bd.get_charge_matrix_openmp(f_data, dim_p, dim_t, m_vec_impact, start_pitch, m_start_tick);
//    m_bd.get_charge_matrix_openmp_noscan(f_data, dim_p, dim_t, m_vec_impact, start_pitch, m_start_tick);
//    t_get_charge_matrix += omp_get_wtime();
//
//    log->debug("ImpactTransform::ImpactTransform() : get_charge_matrix() Total running time : {}", t_get_charge_matrix);
//
//
//    std::complex<float> *data_c = (std::complex<float>*)malloc(sizeof(std::complex<float>) * dim_p * dim_t);
//#pragma omp target enter data map(alloc: data_c[0:dim_p*dim_t])    
//
//    size_t acc_dim_p = end_ch - start_ch + 2 * npad_wire;
//    std::cout << "acc_dim_p = " << acc_dim_p << std::endl;
//    size_t acc_dim_t = m_end_tick - m_start_tick;
//    std::cout << "acc_dim_t = " << acc_dim_t << std::endl;
//
//    std::complex<float> *acc_data_f_w = (std::complex<float>*)malloc(sizeof(std::complex<float>) * acc_dim_p * acc_dim_t);
//#pragma omp target enter data map(alloc: acc_data_f_w[0:acc_dim_p*acc_dim_t])    
//    std::cout << "Malloc on host for acc_data_f_w finished" << std::endl;
//    float *acc_data_t_w = (float*)malloc(sizeof(float) * acc_dim_p * acc_dim_t); 
//#pragma omp target enter data map(alloc: acc_data_t_w[0:acc_dim_p*acc_dim_t])    
//    std::cout << "Malloc on host for acc_data_t_w finished" << std::endl;
//
//    //FIXME: Can we combine the two dft into a single one? Will that improve time?
//    t_temp = -omp_get_wtime();
//
////tw: be careful: If uncomment these two lines, and comment out the next two lines, we switch the two statements about fft, and currently it gives the wrong answer!!!!
////    OpenMPArray::dft_rc(data_c, f_data, dim_p, dim_t, 1);
////    OpenMPArray::dft_cc(data_c, data_c, dim_p, dim_t, 0);
//
//    OpenMPArray::dft_rc(data_c, f_data, dim_p, dim_t, 0);
//    OpenMPArray::dft_cc(data_c, data_c, dim_p, dim_t, 1);
//
////    OpenMPArray::dft_rc_2d_test(data_c, f_data, dim_t, dim_p);   //Be careful about the relative order of two dimensions!!
//    t_temp += omp_get_wtime();
//    std::cout << "tw: dft_rc and dft_cc on data_c and f_data takes " << t_temp * 1000 << " ms" << std::endl;
//#pragma omp target exit data map(delete:f_data[0:dim_p*dim_t])
//
//    //tw: test dft API result starts here. This block of code should be placed at a different file for unit test
////    {
////      //horizonal direction: n0
////      //vertical direction: n1
////      int n0 = 8;
////      int n1 = 4;
////      float *test_in = (float*)malloc(sizeof(float) * n0 * n1);
////      for(int i=0; i<n1; i++)
////      {
////        for(int j=0; j<n0; j++)
////        {
//////          test_in[j + i * n0] = j + i * n0;
////          test_in[j + i * n0] = (i+1) * (i+1) + j + (j+1) * (j+1) * i;
////        }
////      }
////      for(int i=0; i<n0 * n1; i++)
////      {
////        std::cout << test_in[i] << "    ";
////        if(i % n0 == n0 - 1)
////          std::cout << "\n";
////      }
////#pragma omp target enter data map(to: test_in[0:n0*n1])
////      std::complex<float> *test_out = (std::complex<float>*)malloc(sizeof(std::complex<float>) * n0 * n1);
////#pragma omp target enter data map(alloc: test_out[0:n0*n1])
//////      OpenMPArray::dft_rc(test_out, test_in, n0, n1, 0);
//////      OpenMPArray::dft_cc(test_out, test_out, n0, n1, 1);
////      OpenMPArray::dft_rc_2d_test(test_out, test_in, n1, n0);      
////#pragma omp target exit data map(from: test_out[0:n0*n1])
////      for(int i=0; i<n0 * n1; i++)
////      {
////        std::cout << test_out[i] << "    ";
////        if(i % n0 == n0 - 1)
////          std::cout << "\n";
////      }
////    }
//    //tw: test dft API result ends here
//
//
//
//
////    //tw: DEBUG start here!
////    std::cout << "dim_p = " << dim_p << "  dim_t = " << dim_t << std::endl;
////#pragma omp target update from(data_c[0:dim_p*dim_t])
////    for(int ip=0; ip<dim_p; ip+=10)
////    {
////      for(int it=0; it<dim_t; it+=10)
////      {
////        std::cout << "tw: DEBUG: ip = " << ip << "  it = " << it << "  data = " << data_c[ip + it * dim_p] << std::endl;
////      }
////    }
////    //tw: DEBUG end here!
//
//
//
//    wstart = omp_get_wtime();
//////    KokkosArray::array_xxc acc_data_f_w = KokkosArray::Zero<KokkosArray::array_xxc>(end_ch - start_ch + 2 * npad_wire, m_end_tick - m_start_tick);
//////    KokkosArray::array_xxf acc_data_t_w = KokkosArray::Zero<KokkosArray::array_xxf>(end_ch - start_ch + 2 * npad_wire, m_end_tick - m_start_tick);
//
//    log->info("tw: pitch   {} {} {} {}",start_pitch, end_pitch, dim_p, dim_t);
//    log->info("tw: start-end {}-{} {}-{} {} {}",m_start_ch, m_end_ch, m_start_tick, m_end_tick, acc_dim_p, acc_dim_t);
//    log->info("tw: m_num_group {}",m_num_group);
//
////    This is the new version, where time size is set to be the smaller between m_start_tick - m_end_tick and sp_f.size()
//    //  Convolution with Field Response
//    {
//        // fine grained resp. matrix
//////        KokkosArray::array_xxc resp_f_w_k =
//////            KokkosArray::Zero<KokkosArray::array_xxc>(end_pitch - start_pitch, m_end_tick - m_start_tick);
//        std::complex<float> *resp_f_w_k = (std::complex<float>*)malloc(sizeof(std::complex<float>) * dim_p * dim_t);
//        std::cout << "Start to do conv" << std::endl;
//
//
//////        auto resp_f_w_k_h = Kokkos::create_mirror_view(resp_f_w_k);
//
//        int nimpact = m_pir->nwires() * (m_num_group - 1);
//        assert(dim_p >= nimpact);
//        double max_impact = (0.5 + m_num_pad_wire) * m_pir->pitch();
//
//        // TODO this loop could be parallerized by seperating it into 2 parts
//        // 1, extract the m_pir into a Kokkos::Array first
//        // 2, do various operation in a parallerized fasion
////        std::cout << "nimpact = " << nimpact << std::endl;
////        std::cout << "DEBUG: m_end_tick = " << m_end_tick << "  m_start_tick = " << m_start_tick << std::endl;
//        t_temp = -omp_get_wtime(); 
//        for (int jimp = 0; jimp < nimpact; ++jimp) 
//        {
//            float impact = -max_impact + jimp * m_pir->impact();
////            std::cout << "tw: jimp " << jimp << ", " << impact;
//            // to avoid exess of boundary.
//            // central is < 0, after adding 0.01 * m_pir->impact(), it is > 0
//            impact += impact < 0 ? 0.01 * m_pir->impact() : -0.01 * m_pir->impact();
////            std::cout << ", " << impact;
//
//            auto ip = m_pir->closest(impact);
//            Waveform::compseq_t sp_f = ip->spectrum();
//            Waveform::realseq_t sp_t = Waveform::idft(sp_f);
//            int sp_size = sp_f.size();
//            int min_t_size = min(sp_size, m_end_tick - m_start_tick);
//            Waveform::realseq_t sp_t_reduced(min_t_size, 0);
//            for (int icol = 0; icol != min_t_size; icol++) 
//            {
//              sp_t_reduced[icol] = sp_t[icol];
//            }
////            if(jimp % 10 == 0)
////            {
////              std::cout << "jimp = " << jimp << "  DEBUG: sp_t_reduced.size() = " << sp_t_reduced.size() << std::endl;
////              for(int i=0; i<sp_t_reduced.size(); i++)
////              {
////                std::cout << "jimp = " << jimp << "  DEBUG: sp_t_reduced: i = " << i << "  data = " << sp_t_reduced[i] << std::endl;
////              }
////            }
//
//            auto sp_f_reduced = Waveform::dft(sp_t_reduced);
////            if(jimp == 0)
////            {
////              int sz = sp_f_reduced.size();
////              for(int ii=1; ii<sz; ii++)
////                std::cout << "ii = " << ii << "   sp_f_reduced = " << sp_f_reduced[ii] << "   mirror = " << sp_f_reduced[sz-ii] << std::endl;
////            }
//
//            // 0 and right on the left; left on the right
//            // int idx = impact < 0 ? end_pitch - start_pitch - (std::round(nimpact / 2.) - jimp)
//            //                      : jimp - std::round(nimpact / 2.);
//            // 0 and left and the left; right on the right
//            int idx = impact < 0.1 * m_pir->impact() ? std::round(nimpact / 2.) - jimp
//                                 : dim_p - (jimp - std::round(nimpact / 2.));
////            std::cout << ", " << idx << std::endl;
//            assert(idx >= 0 && idx < dim_p);
//            for (size_t i = 0; i < sp_f_reduced.size(); ++i) 
//            {
//              //FIXME
//                resp_f_w_k[idx + (dim_p * i)] = sp_f_reduced[i];
//            }
//        }
//        t_temp += omp_get_wtime(); 
//        std::cout << "FFT and iFFT on host for resp_f_w_k takes " << t_temp * 1000.0 << " ms" << std::endl;
//
////    //tw: DEBUG start here!
////        std::cout << "dim_p = " << dim_p << "  dim_t = " << dim_t << std::endl;
////    for(int ip=0; ip<dim_p; ip+=10)
////    {
////      for(int it=0; it<dim_t; it+=10)
////      {
////        std::cout << "tw: DEBUG: ip = " << ip << "  it = " << it << "  data = " << resp_f_w_k[ip + it * dim_p] << std::endl;
////      }
////    }
////    //tw: DEBUG end here!
//
//        //can we make that asynchronous?
//////        Kokkos::deep_copy(resp_f_w_k, resp_f_w_k_h);
////Why this H2D moves 763,248,640 byte of data???
//#pragma omp target enter data map(to: resp_f_w_k[0:dim_p*dim_t])    
//////        resp_f_w_k = KokkosArray::dft_cc(resp_f_w_k, 1);
//        OpenMPArray::dft_cc(resp_f_w_k, resp_f_w_k, dim_p, dim_t, 1);
//////        // auto tmp = KokkosArray::idft_cr(resp_f_w_k, 0);
//////        // resp_f_w_k = KokkosArray::dft_cc(resp_f_w_k, 1);
//////        // std::cout << "resp_f_w_k: " << KokkosArray::dump_2d_view(tmp, 10000);
//
////          This two are done earlier after get_charge_matrix        
//////        auto data_c = KokkosArray::dft_rc(f_data, 0);
//////        data_c = KokkosArray::dft_cc(data_c, 1);
//
//        // S(f) * R(f)
//////        Kokkos::parallel_for(
//////            Kokkos::MDRangePolicy<Kokkos::Rank<2, Kokkos::Iterate::Left>>({0, 0}, {data_c.extent(0), data_c.extent(1)}),
//////            KOKKOS_LAMBDA(const KokkosArray::Index& i0, const KokkosArray::Index& i1) {
//////                data_c(i0, i1) *= resp_f_w_k(i0, i1);
//////                // data_c(i0, i1) *= 1.0;
//////            });
////
//// tw: !debug, remove comment of this code block
//        t_temp = -omp_get_wtime();
//#pragma omp target teams distribute parallel for simd
//        for(int i=0; i<dim_t * dim_p; i++)
//        {
//          data_c[i] *= resp_f_w_k[i];
//        }
//
////    //tw: DEBUG start here!
////#pragma omp target update from(data_c[0:dim_t*dim_p])
////    for(int ip=0; ip<dim_p; ip+=10)
////    {
////      for(int it=0; it<dim_t; it+=10)
////      {
////        std::cout << "tw: DEBUG: ip = " << ip << "  it = " << it << "  data = " << data_c[ip + it * dim_p] << std::endl;
////      }
////    }
////    //tw: DEBUG end here!
//
//        t_temp += omp_get_wtime();
//        std::cout << "Time for data_c *= resp_f_w_k = " << t_temp * 1000 << " ms" << std::endl;
//        std::cout << "tw: finish S*R, first step" << std::endl;
//        // transfer wire to time domain
//////        data_c = KokkosArray::idft_cc(data_c, 1);
//        OpenMPArray::idft_cc(data_c, data_c, dim_p, dim_t, 1);
//
//        // extract M(channel) from M(impact)
//////        Kokkos::parallel_for(
//////            Kokkos::MDRangePolicy<Kokkos::Rank<2, Kokkos::Iterate::Left>>({0, 0}, {acc_data_f_w.extent(0), acc_data_f_w.extent(1)}),
//////            KOKKOS_LAMBDA(const KokkosArray::Index& i0, const KokkosArray::Index& i1) {
//////                acc_data_f_w(i0, i1) = data_c((i0 + 1) * 10, i1);
//////            });
//#pragma omp target teams distribute parallel for simd collapse(2)        
//        for(int i1=0; i1<acc_dim_t; i1++)
//        {
//          for(int i0=0; i0<acc_dim_p; i0++)
//          {
//            //FIXME: correctness check, also notice memory access is not coalescing
//            acc_data_f_w[i0 + (acc_dim_p * i1)] = data_c[(i0 + 1) * 10 + (dim_p * i1)];
//          }
//        }
//        std::cout << "tw: finish S*R, second step" << std::endl;
//#pragma omp target exit data map(delete: resp_f_w_k[0:dim_p*dim_t])    
//    }
//#pragma omp target exit data map(delete: data_c[0:dim_p*dim_t])
//
////    This is the old version, where time size is fixed to m_start_tick - m_end_tick    
////    {
////        // fine grained resp. matrix
////////        KokkosArray::array_xxc resp_f_w_k =
////////            KokkosArray::Zero<KokkosArray::array_xxc>(end_pitch - start_pitch, m_end_tick - m_start_tick);
////        std::complex<float> *resp_f_w_k = (std::complex<float>*)malloc(sizeof(std::complex<float>) * dim_p * dim_t);
////        std::cout << "Start to do conv" << std::endl;
////
////
////////        auto resp_f_w_k_h = Kokkos::create_mirror_view(resp_f_w_k);
////
////        int nimpact = m_pir->nwires() * (m_num_group - 1);
////        assert(dim_p >= nimpact);
////        double max_impact = (0.5 + m_num_pad_wire) * m_pir->pitch();
////
////        // TODO this loop could be parallerized by seperating it into 2 parts
////        // 1, extract the m_pir into a Kokkos::Array first
////        // 2, do various operation in a parallerized fasion
////        std::cout << "nimpact = " << nimpact << std::endl;
////        std::cout << "DEBUG: m_end_tick = " << m_end_tick << "  m_start_tick = " << m_start_tick << std::endl;
////        for (int jimp = 0; jimp < nimpact; ++jimp) 
////        {
////            float impact = -max_impact + jimp * m_pir->impact();
//////            std::cout << "tw: jimp " << jimp << ", " << impact;
////            // to avoid exess of boundary.
////            // central is < 0, after adding 0.01 * m_pir->impact(), it is > 0
////            impact += impact < 0 ? 0.01 * m_pir->impact() : -0.01 * m_pir->impact();
//////            std::cout << ", " << impact;
////
////            auto ip = m_pir->closest(impact);
////            Waveform::compseq_t sp_f = ip->spectrum();
////            Waveform::realseq_t sp_t = Waveform::idft(sp_f);
////            Waveform::realseq_t sp_t_reduced(m_end_tick - m_start_tick, 0);
////            for (int icol = 0; icol != m_end_tick - m_start_tick; icol++) 
////            {
////                if (icol >= int(sp_t.size())) 
////                  break;
////                sp_t_reduced.at(icol) = sp_t[icol];
////            }
////            if(jimp % 10 == 0)
////            {
////              std::cout << "jimp = " << jimp << "  DEBUG: sp_t_reduced.size() = " << sp_t_reduced.size() << std::endl;
////              for(int i=0; i<sp_t_reduced.size(); i++)
////              {
////                std::cout << "jimp = " << jimp << "  DEBUG: sp_t_reduced: i = " << i << "  data = " << sp_t_reduced[i] << std::endl;
////              }
////            }
////
////            auto sp_f_reduced = Waveform::dft(sp_t_reduced);
////
////            // 0 and right on the left; left on the right
////            // int idx = impact < 0 ? end_pitch - start_pitch - (std::round(nimpact / 2.) - jimp)
////            //                      : jimp - std::round(nimpact / 2.);
////            // 0 and left and the left; right on the right
////            int idx = impact < 0.1 * m_pir->impact() ? std::round(nimpact / 2.) - jimp
////                                 : dim_p - (jimp - std::round(nimpact / 2.));
//////            std::cout << ", " << idx << std::endl;
////            assert(idx >= 0 && idx < dim_p);
////            for (size_t i = 0; i < sp_f_reduced.size(); ++i) 
////            {
////              //FIXME
////                resp_f_w_k[idx + (dim_p * i)] = sp_f_reduced[i];
////            }
////        }
//
//////    acc_data_t_w = KokkosArray::idft_cr(acc_data_f_w, 0);
//
//
//    OpenMPArray::idft_cr(acc_data_t_w, acc_data_f_w, acc_dim_p, acc_dim_t, 0);
//#pragma omp target exit data map(delete: acc_data_f_w[0:acc_dim_p*acc_dim_t])
//
//////    auto acc_data_t_w_h = Kokkos::create_mirror_view(acc_data_t_w);
//////    // std::cout << "yuhw: acc_data_t_w_h: " << KokkosArray::dump_2d_view(acc_data_t_w,10) << std::endl;
//////    Kokkos::deep_copy(acc_data_t_w_h, acc_data_t_w);
//#pragma omp target exit data map(from:acc_data_t_w[0:acc_dim_p*acc_dim_t])
//
////    //tw: DEBUG start here!
////    std::cout << "************************" << std::endl;
////    for(int ip=0; ip<acc_dim_p; ip++)
////    {
////      for(int it=0; it<acc_dim_t; it++)
////      {
////        if(abs(acc_data_t_w[ip + it * acc_dim_p]) > 1e-8)
////          std::cout << "tw: DEBUG: ip = " << ip << "  it = " << it << "  acc_data_t_w_eigen data = " << acc_data_t_w[ip + it * acc_dim_p] << std::endl;
////      }
////    }
////    std::cout << "************************" << std::endl;
////    //tw: DEBUG end here!
//    
//    Eigen::Map<Eigen::ArrayXXf> acc_data_t_w_eigen(acc_data_t_w, acc_dim_p, acc_dim_t);
//    m_decon_data = acc_data_t_w_eigen; // FIXME: reduce this copy
//    
//    double timer_fft = omp_get_wtime() - wstart;
//    log->debug("ImpactTransform::transform_matrix: FFT: {}", timer_fft);
//    timer_transform = omp_get_wtime() - timer_transform;
//    log->debug("ImpactTransform::transform_matrix: Total: {}", timer_transform);
//
//    log->debug("ImpactTransform::transform_matrix: # of channels: {} # of ticks: {}", m_decon_data.rows(), m_decon_data.cols());
//    log->debug("ImpactTransform::transform_matrix: m_decon_data.sum(): {}", m_decon_data.sum());
//    return true;
//}

GenOpenMP::ImpactTransform::~ImpactTransform() {}

Waveform::realseq_t GenOpenMP::ImpactTransform::waveform(int iwire) const
{
    const int nsamples = m_bd.tbins().nbins();
    if (iwire < m_start_ch || iwire >= m_end_ch) {
        return Waveform::realseq_t(nsamples, 0.0);
    }
    else {
        Waveform::realseq_t wf(nsamples, 0.0);
        for (int i = 0; i != nsamples; i++) {
            if (i >= m_start_tick && i < m_end_tick) {
                wf.at(i) = m_decon_data(iwire - m_start_ch, i - m_start_tick);
            }
            else {
                // wf.at(i) = 1e-25;
            }
            // std::cout << m_decon_data(iwire-m_start_ch,i-m_start_tick) << std::endl;
        }

        // if (m_pir->closest(0)->long_aux_waveform().size() > 0) {
        //     // now convolute with the long-range response ...
        //     const size_t nlength = fft_best_length(nsamples + m_pir->closest(0)->long_aux_waveform_pad());

        //     // nlength = nsamples;

        //     //   std::cout << nlength << " " << nsamples + m_pir->closest(0)->long_aux_waveform_pad() << std::endl;

        //     wf.resize(nlength, 0);
        //     Waveform::realseq_t long_resp = m_pir->closest(0)->long_aux_waveform();
        //     long_resp.resize(nlength, 0);
        //     Waveform::compseq_t spec = Waveform::dft(wf);
        //     Waveform::compseq_t long_spec = Waveform::dft(long_resp);
        //     for (size_t i = 0; i != nlength; i++) {
        //         spec.at(i) *= long_spec.at(i);
        //     }
        //     wf = Waveform::idft(spec);
        //     wf.resize(nsamples, 0);
        // }

        return wf;
    }
}

//  tw: test dft API result starts here. This block of code should be placed at a different file for unit test
//    {
//      //horizonal direction: n0
//      //vertical direction: n1
//      int n0 = 8;
//      int n1 = 4;
//      float *test_in = (float*)malloc(sizeof(float) * n0 * n1);
//      for(int i=0; i<n1; i++)
//      {
//        for(int j=0; j<n0; j++)
//        {
////          test_in[j + i * n0] = j + i * n0;
//          test_in[j + i * n0] = (i+1) * (i+1) + j + (j+1) * (j+1) * i;
//        }
//      }
//      for(int i=0; i<n0 * n1; i++)
//      {
//        std::cout << test_in[i] << "    ";
//        if(i % n0 == n0 - 1)
//          std::cout << "\n";
//      }
//#pragma omp target enter data map(to: test_in[0:n0*n1])
//      std::complex<float> *test_out = (std::complex<float>*)malloc(sizeof(std::complex<float>) * n0 * n1);
//#pragma omp target enter data map(alloc: test_out[0:n0*n1])
////      OpenMPArray::dft_rc(test_out, test_in, n0, n1, 0);
////      OpenMPArray::dft_cc(test_out, test_out, n0, n1, 1);
//      OpenMPArray::dft_rc_2d_test(test_out, test_in, n1, n0);      
//#pragma omp target exit data map(from: test_out[0:n0*n1])
//      for(int i=0; i<n0 * n1; i++)
//      {
//        std::cout << test_out[i] << "    ";
//        if(i % n0 == n0 - 1)
//          std::cout << "\n";
//      }
//    }
//  tw: test dft API result ends here

//  ===========================================================================================================
//    This is the old version, where time size is fixed to m_start_tick - m_end_tick    
//    {
//        // fine grained resp. matrix
//////        KokkosArray::array_xxc resp_f_w_k =
//////            KokkosArray::Zero<KokkosArray::array_xxc>(end_pitch - start_pitch, m_end_tick - m_start_tick);
//        std::complex<float> *resp_f_w_k = (std::complex<float>*)malloc(sizeof(std::complex<float>) * dim_p * dim_t);
//        std::cout << "Start to do conv" << std::endl;
//
//
//////        auto resp_f_w_k_h = Kokkos::create_mirror_view(resp_f_w_k);
//
//        int nimpact = m_pir->nwires() * (m_num_group - 1);
//        assert(dim_p >= nimpact);
//        double max_impact = (0.5 + m_num_pad_wire) * m_pir->pitch();
//
//        // TODO this loop could be parallerized by seperating it into 2 parts
//        // 1, extract the m_pir into a Kokkos::Array first
//        // 2, do various operation in a parallerized fasion
//        std::cout << "nimpact = " << nimpact << std::endl;
//        std::cout << "DEBUG: m_end_tick = " << m_end_tick << "  m_start_tick = " << m_start_tick << std::endl;
//        for (int jimp = 0; jimp < nimpact; ++jimp) 
//        {
//            float impact = -max_impact + jimp * m_pir->impact();
////            std::cout << "tw: jimp " << jimp << ", " << impact;
//            // to avoid exess of boundary.
//            // central is < 0, after adding 0.01 * m_pir->impact(), it is > 0
//            impact += impact < 0 ? 0.01 * m_pir->impact() : -0.01 * m_pir->impact();
////            std::cout << ", " << impact;
//
//            auto ip = m_pir->closest(impact);
//            Waveform::compseq_t sp_f = ip->spectrum();
//            Waveform::realseq_t sp_t = Waveform::idft(sp_f);
//            Waveform::realseq_t sp_t_reduced(m_end_tick - m_start_tick, 0);
//            for (int icol = 0; icol != m_end_tick - m_start_tick; icol++) 
//            {
//                if (icol >= int(sp_t.size())) 
//                  break;
//                sp_t_reduced.at(icol) = sp_t[icol];
//            }
//            if(jimp % 10 == 0)
//            {
//              std::cout << "jimp = " << jimp << "  DEBUG: sp_t_reduced.size() = " << sp_t_reduced.size() << std::endl;
//              for(int i=0; i<sp_t_reduced.size(); i++)
//              {
//                std::cout << "jimp = " << jimp << "  DEBUG: sp_t_reduced: i = " << i << "  data = " << sp_t_reduced[i] << std::endl;
//              }
//            }
//
//            auto sp_f_reduced = Waveform::dft(sp_t_reduced);
//
//            // 0 and right on the left; left on the right
//            // int idx = impact < 0 ? end_pitch - start_pitch - (std::round(nimpact / 2.) - jimp)
//            //                      : jimp - std::round(nimpact / 2.);
//            // 0 and left and the left; right on the right
//            int idx = impact < 0.1 * m_pir->impact() ? std::round(nimpact / 2.) - jimp
//                                 : dim_p - (jimp - std::round(nimpact / 2.));
////            std::cout << ", " << idx << std::endl;
//            assert(idx >= 0 && idx < dim_p);
//            for (size_t i = 0; i < sp_f_reduced.size(); ++i) 
//            {
//              //FIXME
//                resp_f_w_k[idx + (dim_p * i)] = sp_f_reduced[i];
//            }
//        }

