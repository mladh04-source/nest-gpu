
/*
 *  iaf_psc_exp_neuron_nestml.cu
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
#include "iaf_psc_exp_neuron_nestml.h"
#include "spike_buffer.h"

using namespace iaf_psc_exp_neuron_nestml_ns;

__global__ void iaf_psc_exp_neuron_nestml_Calibrate(int n_node, float *param_arr,
				      int n_param, float h)
{
  int i_neuron = threadIdx.x + blockIdx.x * blockDim.x;
  if (i_neuron < n_node) {
    float *param = param_arr + n_param*i_neuron;
    param[i___h] = h; // as ms    


      param[i___P__I_syn_exc__I_syn_exc] = exp((-param[i___h]) / param[i_tau_syn_exc]); // as real


      param[i___P__I_syn_inh__I_syn_inh] = exp((-param[i___h]) / param[i_tau_syn_inh]); // as real


      param[i___P__V_m__I_syn_exc] = param[i_tau_m] * param[i_tau_syn_exc] * ((-exp(param[i___h] / param[i_tau_m])) + exp(param[i___h] / param[i_tau_syn_exc])) * exp((-param[i___h]) * (param[i_tau_m] + param[i_tau_syn_exc]) / (param[i_tau_m] * param[i_tau_syn_exc])) / (param[i_C_m] * (param[i_tau_m] - param[i_tau_syn_exc])); // as real


      param[i___P__V_m__I_syn_inh] = param[i_tau_m] * param[i_tau_syn_inh] * (exp(param[i___h] / param[i_tau_m]) - exp(param[i___h] / param[i_tau_syn_inh])) * exp((-param[i___h]) * (param[i_tau_m] + param[i_tau_syn_inh]) / (param[i_tau_m] * param[i_tau_syn_inh])) / (param[i_C_m] * (param[i_tau_m] - param[i_tau_syn_inh])); // as real


      param[i___P__V_m__V_m] = exp((-param[i___h]) / param[i_tau_m]); // as real


      param[i___P__refr_t__refr_t] = 1; // as real
  }
}


__global__ void iaf_psc_exp_neuron_nestml_Update(int n_node, int i_node_0, float *var_arr,
				   float *param_arr, int n_var, int n_param)
{
  int i_neuron = threadIdx.x + blockIdx.x * blockDim.x;
  if (i_neuron < n_node) {
    float *var = var_arr + n_var*i_neuron;
    float *param = param_arr + n_param*i_neuron;
    /**
     * subthreshold updates of the convolution variables
     *
     * step 1: regardless of whether and how integrate_odes() will be called, update variables due to convolutions
    **/

    /**
     * Begin NESTML generated code for the update block(s)
    **/
  if (var[i_refr_t] > 0)
  {  

    // start rendered code for integrate_odes(I_syn_exc, I_syn_inh, refr_t)

    // analytic solver: integrating state variables (first step): I_syn_exc, I_syn_inh, refr_t, 
    const double I_syn_exc__tmp = var[i_I_syn_exc] * param[i___P__I_syn_exc__I_syn_exc];
    const double I_syn_inh__tmp = var[i_I_syn_inh] * param[i___P__I_syn_inh__I_syn_inh];
    const double refr_t__tmp = param[i___P__refr_t__refr_t] * var[i_refr_t] - 1.0 * param[i___h];
    // analytic solver: integrating state variables (second step): I_syn_exc, I_syn_inh, refr_t, 
    /* replace analytically solvable variables with precisely integrated values  */
    var[i_I_syn_exc] = I_syn_exc__tmp;
    var[i_I_syn_inh] = I_syn_inh__tmp;
    var[i_refr_t] = refr_t__tmp;
  }
  else
  {  

    // start rendered code for integrate_odes(I_syn_exc, I_syn_inh, V_m)

    // analytic solver: integrating state variables (first step): I_syn_exc, I_syn_inh, V_m, 
    const double I_syn_exc__tmp = var[i_I_syn_exc] * param[i___P__I_syn_exc__I_syn_exc];
    const double I_syn_inh__tmp = var[i_I_syn_inh] * param[i___P__I_syn_inh__I_syn_inh];
    const double V_m__tmp = (-param[i_E_L]) * param[i___P__V_m__V_m] + param[i_E_L] + var[i_I_syn_exc] * param[i___P__V_m__I_syn_exc] + var[i_I_syn_inh] * param[i___P__V_m__I_syn_inh] + var[i_V_m] * param[i___P__V_m__V_m] - param[i_I_e] * param[i___P__V_m__V_m] * param[i_tau_m] / param[i_C_m] + param[i_I_e] * param[i_tau_m] / param[i_C_m] - param[i_I_stim] * param[i___P__V_m__V_m] * param[i_tau_m] / param[i_C_m] + param[i_I_stim] * param[i_tau_m] / param[i_C_m];
    // analytic solver: integrating state variables (second step): I_syn_exc, I_syn_inh, V_m, 
    /* replace analytically solvable variables with precisely integrated values  */
    var[i_I_syn_exc] = I_syn_exc__tmp;
    var[i_I_syn_inh] = I_syn_inh__tmp;
    var[i_V_m] = V_m__tmp;
  }

    /**
     * Begin NESTML generated code for the onReceive block(s)
    **/

    if (var[N_SCAL_VAR + i_exc_spikes])
    {      
      var[i_I_syn_exc] += (0.001 * var[N_SCAL_VAR + i_exc_spikes]) * 1.0 * 1000.0;      
      var[N_SCAL_VAR + i_exc_spikes] = 0; // reset the value
    }
    if (var[N_SCAL_VAR + i_inh_spikes])
    {      
      var[i_I_syn_inh] += (0.001 * var[N_SCAL_VAR + i_inh_spikes]) * 1.0 * 1000.0;      
      var[N_SCAL_VAR + i_inh_spikes] = 0; // reset the value
    }

    /**
     * subthreshold updates of the convolution variables
     *
     * step 2: regardless of whether and how integrate_odes() was called, update variables due to convolutions. Set to the updated values at the end of the timestep.
    **/

    /**
     * Begin NESTML generated code for the onCondition block(s)
    **/

    if (var[i_refr_t] <= 0 && var[i_V_m] >= param[i_V_th])
    {
      var[i_refr_t] = param[i_refr_T];
      var[i_V_m] = param[i_V_reset];
      PushSpike(i_node_0 + i_neuron, 1.0);;
    }
  }
}


iaf_psc_exp_neuron_nestml::~iaf_psc_exp_neuron_nestml()
{
  FreeVarArr();
  FreeParamArr();
}

int iaf_psc_exp_neuron_nestml::Init(int i_node_0, int n_node, int /*n_port*/,
			 int i_group)
{
  BaseNeuron::Init(i_node_0, n_node, 2 /*n_port*/, i_group);
  node_type_ = i_iaf_psc_exp_neuron_nestml_model;

  // State variables
  n_scal_var_ = N_SCAL_VAR;
  n_port_var_ = N_PORT_VAR;
  n_var_ = n_scal_var_ + n_port_var_;

  // Parameters
  n_scal_param_ = N_SCAL_PARAM;
  n_param_ = n_scal_param_;

  AllocParamArr();
  AllocVarArr();

  scal_var_name_ = iaf_psc_exp_neuron_nestml_scal_var_name;
  scal_param_name_ = iaf_psc_exp_neuron_nestml_scal_param_name;
  port_var_name_ = iaf_psc_exp_neuron_nestml_port_var_name;

  // Parameters
    SetScalParam(0, n_node, "C_m", 
  250);  // as pF
    SetScalParam(0, n_node, "tau_m", 
  10);  // as ms
    SetScalParam(0, n_node, "tau_syn_inh", 
  2);  // as ms
    SetScalParam(0, n_node, "tau_syn_exc", 
  2);  // as ms
    SetScalParam(0, n_node, "refr_T", 
  2);  // as ms
    SetScalParam(0, n_node, "E_L", 
  (-70));  // as mV
    SetScalParam(0, n_node, "V_reset", 
  (-70));  // as mV
    SetScalParam(0, n_node, "V_th", 
  (-55));  // as mV
    SetScalParam(0, n_node, "I_e", 
  0);  // as pA

    // Internal variables
    SetScalParam(0, n_node, "__h", 0.0);
    SetScalParam(0, n_node, "__P__I_syn_exc__I_syn_exc", 0.0);
    SetScalParam(0, n_node, "__P__I_syn_inh__I_syn_inh", 0.0);
    SetScalParam(0, n_node, "__P__V_m__I_syn_exc", 0.0);
    SetScalParam(0, n_node, "__P__V_m__I_syn_inh", 0.0);
    SetScalParam(0, n_node, "__P__V_m__V_m", 0.0);
    SetScalParam(0, n_node, "__P__refr_t__refr_t", 0.0);

    // Continuous input port: set to 0
    SetScalParam(0, n_node, "I_stim", 0.0);

    // State variables
    SetScalVar(0, n_node, "V_m", 
  *GetScalParam(0, n_node, "E_L"));
    SetScalVar(0, n_node, "refr_t", 
  0);
    SetScalVar(0, n_node, "I_syn_exc", 
  0);
    SetScalVar(0, n_node, "I_syn_inh", 
  0);

  // multiplication factor of input signal is always 1 for all nodes
  float input_weight = 1.0;
  gpuErrchk(cudaMalloc(&port_weight_arr_, sizeof(float)));
  gpuErrchk(cudaMemcpy(port_weight_arr_, &input_weight,
			 sizeof(float), cudaMemcpyHostToDevice));
  port_weight_arr_step_ = 0;
  port_weight_port_step_ = 0;

  // process the input spikes
  port_input_arr_ = GetVarArr() + n_scal_var_ + GetPortVarIdx("exc_spikes");
  port_input_arr_step_ = n_var_;
  port_input_port_step_ = 1;




  return 0;
}

int iaf_psc_exp_neuron_nestml::Free()
{
  FreeVarArr();
  FreeParamArr();

  return 0;
}

int iaf_psc_exp_neuron_nestml::Calibrate(double time_min, float time_resolution)
{
   iaf_psc_exp_neuron_nestml_Calibrate<<<(n_node_+1023)/1024, 1024>>>
    (n_node_, param_arr_, n_param_, time_resolution);
  return 0;
}

int iaf_psc_exp_neuron_nestml::Update(long long it, double t1)
{
  iaf_psc_exp_neuron_nestml_Update<<<(n_node_+1023)/1024, 1024>>>
    (n_node_, i_node_0_, var_arr_, param_arr_, n_var_, n_param_);
  // gpuErrchk( cudaDeviceSynchronize() );
  return 0;
}