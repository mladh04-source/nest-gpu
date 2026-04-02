/*
 *  iaf_psc_exp_neuron_nestml.h
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

#ifndef IAF_PSC_EXP_NEURON_NESTML_H
#define IAF_PSC_EXP_NEURON_NESTML_H

#include <iostream>
#include <string>

#include "cuda_error.h"
#include "node_group.h"
#include "base_neuron.h"
#include "neuron_models.h"
#include "iaf_psc_exp_odeint_solver.h"

#define MIN(a,b) (((a)<(b))?(a):(b))

extern __constant__ float NESTGPUTimeResolution;

namespace iaf_psc_exp_neuron_nestml_ns
{
enum ScalVarIndexes {
  i_V_m,
  i_refr_t,
  i_I_syn_exc,
  i_I_syn_inh,
  N_SCAL_VAR
};

enum ScalParamIndexes {
  i_C_m,
  i_tau_m,
  i_tau_syn_inh,
  i_tau_syn_exc,
  i_refr_T,
  i_E_L,
  i_V_reset,
  i_V_th,
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

const std::string iaf_psc_exp_neuron_nestml_scal_var_name[N_SCAL_VAR] = {
  "V_m",
  "refr_t",
  "I_syn_exc",
  "I_syn_inh",
};

const std::string iaf_psc_exp_neuron_nestml_scal_param_name[N_SCAL_PARAM] = {
  "C_m",
  "tau_m",
  "tau_syn_inh",
  "tau_syn_exc",
  "refr_T",
  "E_L",
  "V_reset",
  "V_th",
  "I_e",
  "__h",
  "I_stim",
};

const std::string iaf_psc_exp_neuron_nestml_port_var_name[N_PORT_VAR] = {
  "exc_spikes",
  "inh_spikes",
};

} // namespace iaf_psc_exp_neuron_nestml_ns

class iaf_psc_exp_neuron_nestml : public BaseNeuron
{
public:
  ~iaf_psc_exp_neuron_nestml();

  // Host-driven numeric solver operating on NEST GPU device arrays.
  IafPscExpOdeintSolver* odeint_solver_ = nullptr;

  int Init(int i_node_0, int n_neuron, int n_port, int i_group,
           unsigned long long* seed = nullptr);

  int Calibrate(double time_min, float time_resolution);

  int Update(long long it, double t1);

  int Free();
};

#endif // IAF_PSC_EXP_NEURON_NESTML_H