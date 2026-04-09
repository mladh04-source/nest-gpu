

/*
 *  aeif_cond_alpha_alt_neuron_nestml.cu
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

#include <config.h>
#include <cmath>
#include <iostream>

#include "aeif_cond_alpha_alt_neuron_nestml.h"
#include "spike_buffer.h"

using namespace aeif_cond_alpha_alt_neuron_nestml_ns;

/*
 * POST-INTEGRATION KERNEL
 *
 * Handles:
 * onReceive: apply buffered spikes to conductance derivatives
 *  onCondition: threshold check, reset, adaptation jump, spike emission
 *
 * ODE integration itself is handled by AeifCondAlphaAltOdeintSolver.
 */
__global__ void aeif_cond_alpha_alt_neuron_nestml_PostUpdate(
    int n_node,
    int i_node_0,
    float* var_arr,
    float* param_arr,
    int n_var,
    int n_param)
{
  const int i_neuron = threadIdx.x + blockIdx.x * blockDim.x;
  if (i_neuron >= n_node)
    return;

  float* var   = var_arr   + n_var   * i_neuron;
  float* param = param_arr + n_param * i_neuron;

 
  // onReceive (spike input ports)

  if (var[N_SCAL_VAR + i_exc_spikes] != 0.0f)
  {
    var[i_g_exc__d] += (0.001f * var[N_SCAL_VAR + i_exc_spikes])
                       * (M_E / param[i_tau_syn_exc]) * 1.0f * 1000.0f;
    var[N_SCAL_VAR + i_exc_spikes] = 0.0f;
  }

  if (var[N_SCAL_VAR + i_inh_spikes] != 0.0f)
  {
    var[i_g_inh__d] += (0.001f * var[N_SCAL_VAR + i_inh_spikes])
                       * (M_E / param[i_tau_syn_inh]) * 1.0f * 1000.0f;
    var[N_SCAL_VAR + i_inh_spikes] = 0.0f;
  }


  // onCondition (threshold / spike / reset)

  if (var[i_refr_t] <= 0.0f && var[i_V_m] >= param[i_V_peak])
  {
    var[i_refr_t] = param[i_refr_T];
    var[i_V_m]    = param[i_V_reset];
    var[i_w]     += param[i_b];
    PushSpike(i_node_0 + i_neuron, 1.0f);
  }

  // Reset input ports
  var[N_SCAL_VAR + i_exc_spikes] = 0.0f;
  var[N_SCAL_VAR + i_inh_spikes] = 0.0f;
}



// Class methods

aeif_cond_alpha_alt_neuron_nestml::~aeif_cond_alpha_alt_neuron_nestml()
{
  Free();
}

int aeif_cond_alpha_alt_neuron_nestml::Init(int i_node_0, int n_node, int /*n_port*/,
                                            int i_group, unsigned long long* seed)
{
  BaseNeuron::Init(i_node_0, n_node, 2 /*n_port*/, i_group, seed);
  node_type_ = i_aeif_cond_alpha_alt_neuron_nestml_model;

  // State variables
  n_scal_var_ = N_SCAL_VAR;
  n_port_var_ = N_PORT_VAR;
  n_var_      = n_scal_var_ + n_port_var_;

  // Parameters
  n_scal_param_ = N_SCAL_PARAM;
  n_param_      = n_scal_param_;

  AllocParamArr();
  AllocVarArr();

  scal_var_name_   = aeif_cond_alpha_alt_neuron_nestml_scal_var_name;
  scal_param_name_ = aeif_cond_alpha_alt_neuron_nestml_scal_param_name;
  port_var_name_   = aeif_cond_alpha_alt_neuron_nestml_port_var_name;


  // Parameters (defaults)

  SetScalParam(0, n_node, "C_m",         281.0);   // pF
  SetScalParam(0, n_node, "refr_T",      2.0);     // ms
  SetScalParam(0, n_node, "V_reset",    -60.0);    // mV
  SetScalParam(0, n_node, "g_L",         30.0);    // nS
  SetScalParam(0, n_node, "E_L",        -70.6);    // mV
  SetScalParam(0, n_node, "a",            4.0);    // nS
  SetScalParam(0, n_node, "b",           80.5);    // pA
  SetScalParam(0, n_node, "Delta_T",      2.0);    // mV
  SetScalParam(0, n_node, "tau_w",      144.0);    // ms
  SetScalParam(0, n_node, "V_th",       -50.4);    // mV
  SetScalParam(0, n_node, "V_peak",       0.0);    // mV
  SetScalParam(0, n_node, "tau_syn_exc",  0.2);    // ms
  SetScalParam(0, n_node, "tau_syn_inh",  2.0);    // ms
  SetScalParam(0, n_node, "E_exc",        0.0);    // mV
  SetScalParam(0, n_node, "E_inh",      -85.0);    // mV
  SetScalParam(0, n_node, "I_e",          0.0);    // pA
  SetScalParam(0, n_node, "I_stim",       0.0);    // pA


  // State variables

  SetScalVar(0, n_node, "V_m",   *GetScalParam(0, n_node, "E_L"));
  SetScalVar(0, n_node, "w",      0.0);
  SetScalVar(0, n_node, "refr_t", 0.0);
  SetScalVar(0, n_node, "g_exc",  0.0);
  SetScalVar(0, n_node, "g_exc__d", 0.0);
  SetScalVar(0, n_node, "g_inh",  0.0);
  SetScalVar(0, n_node, "g_inh__d", 0.0);


  // Host-driven numeric solver using NEST GPU arrays

  odeint_solver_ = new AeifCondAlphaAltOdeintSolver(
      n_node_, var_arr_, n_var_, param_arr_, n_param_);


  // Input handling

  float input_weight = 1.0f;
  gpuErrchk(cudaMalloc(&port_weight_arr_, sizeof(float)));
  gpuErrchk(cudaMemcpy(port_weight_arr_, &input_weight,
                       sizeof(float), cudaMemcpyHostToDevice));
  port_weight_arr_step_  = 0;
  port_weight_port_step_ = 0;

  port_input_arr_       = GetVarArr() + n_scal_var_ + GetPortVarIdx("exc_spikes");
  port_input_arr_step_  = n_var_;
  port_input_port_step_ = 1;

  return 0;
}

int aeif_cond_alpha_alt_neuron_nestml::Calibrate(double /*time_min*/, float /*time_resolution*/)
{
  // ODEINT path does not need RK5/analytic calibration.
  return 0;
}

int aeif_cond_alpha_alt_neuron_nestml::Update(long long /*it*/, double t1)
{
  float dt = 0.0f;
  gpuErrchk(cudaMemcpyFromSymbol(&dt, NESTGPUTimeResolution, sizeof(float)));

  const float t0 = static_cast<float>(t1) - dt;

  odeint_solver_->Step(t0, dt);

  aeif_cond_alpha_alt_neuron_nestml_PostUpdate<<<(n_node_ + 1023) / 1024, 1024>>>(
      n_node_, i_node_0_, var_arr_, param_arr_, n_var_, n_param_);

  return 0;
}

int aeif_cond_alpha_alt_neuron_nestml::Free()
{
  delete odeint_solver_;
  odeint_solver_ = nullptr;

  FreeVarArr();
  FreeParamArr();
  return 0;
}