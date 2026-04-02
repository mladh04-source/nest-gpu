
/*
 *  aeif_cond_alpha_alt_neuron_nestml.h
 *
 *  This file is part of NEST GPU.
 *
 *  Copyright (C) 2021 The NEST Initiative
 *
 *  NEST GPU is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 2 of the License, or
 *  (at your option) any later version.
 *
 *  NEST GPU is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with NEST GPU.  If not, see <http://www.gnu.org/licenses/>.
 *
 */
#ifndef AEIF_COND_ALPHA_ALT_NEURON_NESTML_H
#define AEIF_COND_ALPHA_ALT_NEURON_NESTML_H

#include <iostream>
#include <string>
#include "cuda_error.h"
#include "rk5.h"
#include "node_group.h"
#include "base_neuron.h"
#include "neuron_models.h"

#define MIN(a,b) (((a)<(b))?(a):(b))

extern __constant__ float NESTGPUTimeResolution;

namespace aeif_cond_alpha_alt_neuron_nestml_ns
{
enum ScalVarIndexes {
  i_V_m,
  i_w,
  i_refr_t,
  i_g_exc,
  i_g_exc__d,
  i_g_inh,
  i_g_inh__d,
  N_SCAL_VAR
};

enum ScalParamIndexes {
  i_C_m,
  i_refr_T,
  i_V_reset,
  i_g_L,
  i_E_L,
  i_a,
  i_b,
  i_Delta_T,
  i_tau_w,
  i_V_th,
  i_V_peak,
  i_tau_syn_exc,
  i_tau_syn_inh,
  i_E_exc,
  i_E_inh,
  i_I_e,
  i___h,
  i_I_stim,
  N_SCAL_PARAM
};

enum PortVarIndexes {
  i_exc_spikes,
  i_inh_spikes,
  N_PORT_VAR
};

const std::string aeif_cond_alpha_alt_neuron_nestml_scal_var_name[N_SCAL_VAR] = {
  "V_m",
  "w",
  "refr_t",
  "g_exc",
  "g_exc__d",
  "g_inh",
  "g_inh__d",
};

const std::string aeif_cond_alpha_alt_neuron_nestml_scal_param_name[N_SCAL_PARAM] = {
  "C_m",
  "refr_T",
  "V_reset",
  "g_L",
  "E_L",
  "a",
  "b",
  "Delta_T",
  "tau_w",
  "V_th",
  "V_peak",
  "tau_syn_exc",
  "tau_syn_inh",
  "E_exc",
  "E_inh",
  "I_e",
  "__h",
  "I_stim",
};

const std::string aeif_cond_alpha_alt_neuron_nestml_port_var_name[N_PORT_VAR] = {
  "exc_spikes",
  "inh_spikes",
};

enum GroupParamIndexes {
  i_h_min_rel = 0,  // Min. step in ODE integr. relative to time resolution
  i_h0_rel,         // Starting step in ODE integr. relative to time resolution
  N_GROUP_PARAM
};

const std::string aeif_cond_alpha_alt_neuron_nestml_group_param_name[N_GROUP_PARAM] = {
  "h_min_rel",
  "h0_rel"
};

}; //namespace

struct aeif_cond_alpha_alt_neuron_nestml_rk5
{
  int i_node_0_;
};

template<int NVAR, int NPARAM>
__device__
void Derivatives(double x, float *y, float *dydx, float *param,
		 aeif_cond_alpha_alt_neuron_nestml_rk5 data_struct);

template<int NVAR, int NPARAM>
__device__
void ExternalUpdate(double x, float *y, float *param, bool end_time_step,
		    aeif_cond_alpha_alt_neuron_nestml_rk5 data_struct);

__device__
void NodeInit(int n_var, int n_param, double x, float *y,
	      float *param, aeif_cond_alpha_alt_neuron_nestml_rk5 data_struct);

__device__
void NodeCalibrate(int n_var, int n_param, double x, float *y,
		   float *param, aeif_cond_alpha_alt_neuron_nestml_rk5 data_struct);

class aeif_cond_alpha_alt_neuron_nestml : public BaseNeuron
{
 public:
  RungeKutta5<aeif_cond_alpha_alt_neuron_nestml_rk5> rk5_;
  float h_min_;
  float h_;
  aeif_cond_alpha_alt_neuron_nestml_rk5 rk5_data_struct_;

  int Init(int i_node_0, int n_neuron, int n_port, int i_group);

  int Calibrate(double, float time_resolution);

  int Update(long long it, double t1);

  int GetX(int i_neuron, int n_node, double *x) {
    return rk5_.GetX(i_neuron, n_node, x);
  }
  
  int GetY(int i_var, int i_neuron, int n_node, float *y) {
    return rk5_.GetY(i_var, i_neuron, n_node, y);
  }

  template<int N_PORT>
    int UpdateNR(long long it, double t1);

};


#endif