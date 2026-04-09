
/*
 *  izhikevich_neuron_nestml.cu
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
#include "izhikevich_neuron_nestml.h"
#include "spike_buffer.h"

namespace izhikevich_neuron_nestml_ns
{

__device__
void NodeInit(int n_var, int n_param, double x, float *y, float *param,
	      izhikevich_neuron_nestml_rk5 data_struct)
{
    // Parameters


  param[i_a] = 0.02; // as real


  param[i_b] = 0.2; // as real


  param[i_c] = (-65); // as mV


  param[i_d] = 8.0; // as real


  param[i_V_m_init] = (-65); // as mV


  param[i_V_min] = (-INFINITY) * 1.0; // as mV


  param[i_V_th] = 30; // as mV


  param[i_I_e] = 0; // as pA

    // Internal variables

    // State variables


  y[i_V_m] = param[i_V_m_init]; // as mV


  y[i_U_m] = param[i_V_m_init] * param[i_b]; // as real

  
  // Input port variables
    y[N_SCAL_VAR + i_spikes] = 0;  
}

__device__
void NodeCalibrate(int n_var, int n_param, double x, float *y,
		       float *param, izhikevich_neuron_nestml_rk5 data_struct)
{
  // refractory_step = 0;    
} template<int NVAR, int NPARAM> //, class DataStruct>
__device__
    void Derivatives(double x, float *y, float *dydx, float *param,
		     izhikevich_neuron_nestml_rk5 data_struct)
{  

    dydx[i_V_m] =1.0 * param[i_I_e] + 1.0 * param[i_I_stim] - 1.0 * y[i_U_m] + 0.04 * pow(y[i_V_m], 2) + 5.0 * y[i_V_m] + 140.0;
    dydx[i_U_m] =(-1.0) * y[i_U_m] * param[i_a] + 1.0 * y[i_V_m] * param[i_a] * param[i_b];

    // Input port variables should always be set to 0
    dydx[N_SCAL_VAR + i_spikes] = 0;
}

 template<int NVAR, int NPARAM> //, class DataStruct>
__device__
    void ExternalUpdate
    (double x, float *y, float *param, bool end_time_step,
			izhikevich_neuron_nestml_rk5 data_struct)
{

  // start rendered code for integrate_odes()
  y[i_V_m] += (0.001 * y[N_SCAL_VAR + i_spikes]) * 1.0 * 1000.0;
  y[i_V_m] = max(param[i_V_min], y[i_V_m]);

    /**
     * Begin NESTML generated code for the onReceive block(s)
    **/


    /**
     * Begin NESTML generated code for the onCondition block(s)
    **/

    if (y[i_V_m] >= param[i_V_th])
    {
      y[i_V_m] = param[i_c];
      y[i_U_m] += param[i_d];
      int i_neuron = threadIdx.x + blockIdx.x * blockDim.x;
                       PushSpike(data_struct.i_node_0_ + i_neuron, 1.0);;
    }

  // Reset the input port variables
    y[N_SCAL_VAR + i_spikes] = 0; 
}

}; // namespace
			    
__device__
void NodeInit(int n_var, int n_param, double x, float *y,
	     float *param, izhikevich_neuron_nestml_rk5 data_struct)
{
   izhikevich_neuron_nestml_ns::NodeInit(n_var, n_param, x, y, param, data_struct);
}

__device__
void NodeCalibrate(int n_var, int n_param, double x, float *y,
		  float *param, izhikevich_neuron_nestml_rk5 data_struct)

{
    izhikevich_neuron_nestml_ns::NodeCalibrate(n_var, n_param, x, y, param, data_struct);
}

int Update(long long it, double t1);

template<int NVAR, int NPARAM>
__device__
void ExternalUpdate(double x, float *y, float *param, bool end_time_step,
		    izhikevich_neuron_nestml_rk5 data_struct)
{
    izhikevich_neuron_nestml_ns::ExternalUpdate<NVAR, NPARAM>(x, y, param,
						    end_time_step,
						    data_struct);
}

using namespace izhikevich_neuron_nestml_ns;

template<int NVAR, int NPARAM>
__device__
void Derivatives(double x, float *y, float *dydx, float *param,
		 izhikevich_neuron_nestml_rk5 data_struct)
{
    izhikevich_neuron_nestml_ns::Derivatives<NVAR, NPARAM>(x, y, dydx, param,
						 data_struct);
}

int izhikevich_neuron_nestml::Init(int i_node_0, int n_node, int /*n_port*/,
			 int i_group)
{
  BaseNeuron::Init(i_node_0, n_node, 1 /*n_port*/, i_group);
  node_type_ = i_izhikevich_neuron_nestml_model;

  // State variables
  n_scal_var_ = N_SCAL_VAR;
  n_port_var_ = N_PORT_VAR;
  n_var_ = n_scal_var_ + n_port_var_;

  // Parameters
  n_scal_param_ = N_SCAL_PARAM;
  n_param_ = n_scal_param_;
  n_group_param_ = N_GROUP_PARAM;
  group_param_ = new float[N_GROUP_PARAM];

  scal_var_name_ = izhikevich_neuron_nestml_scal_var_name;
  scal_param_name_ = izhikevich_neuron_nestml_scal_param_name;
  port_var_name_ = izhikevich_neuron_nestml_port_var_name;

  group_param_name_ = izhikevich_neuron_nestml_group_param_name;
  rk5_data_struct_.i_node_0_ = i_node_0_;

  SetGroupParam("h_min_rel", 1.0e-3);
  SetGroupParam("h0_rel",  1.0e-2);
  h_ = group_param_[i_h0_rel] * 0.1;
  rk5_.Init(n_node, n_var_, n_param_, 0.0, h_, rk5_data_struct_);
  var_arr_ = rk5_.GetYArr();
  param_arr_ = rk5_.GetParamArr();

  
  

  // multiplication factor of input signal is always 1 for all nodes
  float input_weight = 1.0;
  gpuErrchk(cudaMalloc(&port_weight_arr_, sizeof(float)));
  gpuErrchk(cudaMemcpy(port_weight_arr_, &input_weight,
			 sizeof(float), cudaMemcpyHostToDevice));
  port_weight_arr_step_ = 0;
  port_weight_port_step_ = 0;

  // process the input spikes
  port_input_arr_ = GetVarArr() + n_scal_var_ + GetPortVarIdx("spikes");
  port_input_arr_step_ = n_var_;
  port_input_port_step_ = 0;




  return 0;
}

int izhikevich_neuron_nestml::Calibrate(double time_min, float time_resolution)
{
  h_min_ = group_param_[i_h_min_rel] * time_resolution;
  h_ = group_param_[i_h0_rel] * time_resolution;
  rk5_.Calibrate(time_min, h_, rk5_data_struct_);
  return 0;
}

int izhikevich_neuron_nestml::Update(long long it, double t1)
{
  rk5_.Update<N_SCAL_VAR + N_PORT_VAR, N_SCAL_PARAM>(t1, h_min_, rk5_data_struct_); 
  return 0;
}