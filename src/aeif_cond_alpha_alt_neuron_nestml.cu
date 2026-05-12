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
#include <boost/numeric/odeint.hpp>
#include <boost/numeric/odeint/external/thrust/thrust.hpp>
#include <boost/ref.hpp>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/for_each.h>
#include <thrust/iterator/counting_iterator.h>

#include "aeif_cond_alpha_alt_neuron_nestml.h"
#include "spike_buffer.h"

using namespace aeif_cond_alpha_alt_neuron_nestml_ns;

namespace
{

typedef float value_type;
typedef thrust::device_vector<value_type> ode_state_type;


/*
 * Copy scalar NEST GPU state variables into a compact Odeint state vector.
 *
 * NEST layout: var_arr[i_neuron * n_var + scalar_index]
 *
 * Odeint layout: ode_state[i_neuron * N_SCAL_VAR + scalar_index]
 *
 * Port variables are intentionally not copied.
 */
__global__ void aeif_cond_alpha_alt_neuron_nestml_CopyNestToOdeState(
    int n_node,
    const float* var_arr,
    int n_var,
    float* ode_state)
{
  const int i_neuron = threadIdx.x + blockIdx.x * blockDim.x;

  if (i_neuron >= n_node)
    return;

  const float* src = var_arr + n_var * i_neuron;
  float* dst = ode_state + N_SCAL_VAR * i_neuron;

  for (int i = 0; i < N_SCAL_VAR; ++i)
    dst[i] = src[i];
}


/*
 * Copy the compact Odeint scalar state back into the NEST GPU state array.
 * Port variables remain untouched here.
 */
__global__ void aeif_cond_alpha_alt_neuron_nestml_CopyOdeStateToNest(
    int n_node,
    float* var_arr,
    int n_var,
    const float* ode_state)
{
  const int i_neuron = threadIdx.x + blockIdx.x * blockDim.x;

  if (i_neuron >= n_node)
    return;

  float* dst = var_arr + n_var * i_neuron;
  const float* src = ode_state + N_SCAL_VAR * i_neuron;

  for (int i = 0; i < N_SCAL_VAR; ++i)
    dst[i] = src[i];

  if (dst[i_refr_t] < 0.0f)
    dst[i_refr_t] = 0.0f;
}


/*
 * Device functor for the right-hand side of the ODE system.
 *
 * Odeint calls AeifCondAlphaAltOdeintSystem on the host.
 * That system function launches this Thrust functor on the GPU.
 */
struct AeifCondAlphaAltDerivativeFunctor
{
  const float* x;
  float* dxdt;
  const float* param;
  int param_stride;

  __host__ __device__
  void operator()(int i_neuron) const
  {
    const float* y = x + i_neuron * N_SCAL_VAR;
    float* dydt = dxdt + i_neuron * N_SCAL_VAR;
    const float* p = param + i_neuron * param_stride;

    const float V_m     = y[i_V_m];
    const float w       = y[i_w];
    const float refr_t  = y[i_refr_t];
    const float g_exc   = y[i_g_exc];
    const float g_exc_d = y[i_g_exc__d];
    const float g_inh   = y[i_g_inh];
    const float g_inh_d = y[i_g_inh__d];

    /*
     * Same voltage clamp idea as the original model.
     */
    const float Vb = V_m < p[i_V_peak] ? V_m : p[i_V_peak];

    const float tau_exc = p[i_tau_syn_exc];
    const float tau_inh = p[i_tau_syn_inh];


    /*
     * Alpha-function synaptic conductance dynamics.
     */
    dydt[i_g_exc] =
        g_exc_d;

    dydt[i_g_exc__d] =
        -g_exc / (tau_exc * tau_exc)
        - 2.0f * g_exc_d / tau_exc;

    dydt[i_g_inh] =
        g_inh_d;

    dydt[i_g_inh__d] =
        -g_inh / (tau_inh * tau_inh)
        - 2.0f * g_inh_d / tau_inh;


    /*
     * Adaptation current.
     */
    dydt[i_w] =
        p[i_a] * ((Vb - p[i_E_L]) / p[i_tau_w])
        - w / p[i_tau_w];


    if (refr_t > 0.0f)
    {
      /*
       * Refractory branch:
       * V_m is frozen, refr_t decreases, w still evolves.
       */
      dydt[i_V_m]    = 0.0f;
      dydt[i_refr_t] = -1.0f;
    }
    else
    {
      /*
       * Normal branch:
       * V_m and w evolve, refr_t remains zero.
       */
      dydt[i_V_m] =
        (
          p[i_Delta_T] * p[i_g_L]
            * expf((Vb - p[i_V_th]) / p[i_Delta_T])
          + p[i_E_L] * p[i_g_L]
          + p[i_E_exc] * g_exc
          + p[i_E_inh] * g_inh
          + p[i_I_e]
          + p[i_I_stim]
          - p[i_g_L] * Vb
          - g_exc * Vb
          - g_inh * Vb
          - w
        ) / p[i_C_m];

      dydt[i_refr_t] = 0.0f;
    }
  }
};


/*
 * Boost.Odeint system function.
 *
 * x : compact scalar state vector on GPU
 * dxdt : compact derivative vector on GPU
 */
struct AeifCondAlphaAltOdeintSystem
{
  const float* param;
  int param_stride;
  int n_neuron;

  void operator()(const ode_state_type& x,
                  ode_state_type& dxdt,
                  const value_type /*t*/) const
  {
    const float* x_ptr = thrust::raw_pointer_cast(x.data());
    float* dxdt_ptr = thrust::raw_pointer_cast(dxdt.data());

    thrust::counting_iterator<int> begin(0);
    thrust::counting_iterator<int> end(n_neuron);

    AeifCondAlphaAltDerivativeFunctor functor;
    functor.x = x_ptr;
    functor.dxdt = dxdt_ptr;
    functor.param = param;
    functor.param_stride = param_stride;

    thrust::for_each(thrust::device, begin, end, functor);
  }
};

} // anonymous namespace


/*
 * PRE-INTEGRATION KERNEL
 *
 * Handles onReceive before ODE integration.
 *
 * This is important for comparison with built-in aeif_cond_alpha.
 * Incoming spikes must affect g_exc__d / g_inh__d before the ODE step.
 */
__global__ void aeif_cond_alpha_alt_neuron_nestml_PreUpdate(
    int n_node,
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


  /*
   * onReceive: excitatory spikes
   */
  if (var[N_SCAL_VAR + i_exc_spikes] != 0.0f)
  {
    var[i_g_exc__d] +=
        (0.001f * var[N_SCAL_VAR + i_exc_spikes])
        * (M_E / param[i_tau_syn_exc])
        * 1000.0f;

    var[N_SCAL_VAR + i_exc_spikes] = 0.0f;
  }


  /*
   * onReceive: inhibitory spikes
   */
  if (var[N_SCAL_VAR + i_inh_spikes] != 0.0f)
  {
    var[i_g_inh__d] +=
        (0.001f * var[N_SCAL_VAR + i_inh_spikes])
        * (M_E / param[i_tau_syn_inh])
        * 1000.0f;

    var[N_SCAL_VAR + i_inh_spikes] = 0.0f;
  }
}


/*
 * POST-INTEGRATION KERNEL
 *
 * Handles only threshold, reset and spike emission after ODE integration.
 *
 * (onReceive is not handled here anymore, because it is now handled before integration in aeif_cond_alpha_alt_neuron_nestml_PreUpdate()).
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


  /*
   * onCondition: threshold / reset / spike
   */
  if (var[i_refr_t] <= 0.0f && var[i_V_m] >= param[i_V_peak])
  {
    var[i_refr_t] = param[i_refr_T];
    var[i_V_m]    = param[i_V_reset];
    var[i_w]     += param[i_b];

    PushSpike(i_node_0 + i_neuron, 1.0f);
  }


  /*
   * Safety reset of input ports.
   * Normally they are already reset in PreUpdate.
   */
  var[N_SCAL_VAR + i_exc_spikes] = 0.0f;
  var[N_SCAL_VAR + i_inh_spikes] = 0.0f;
}



// Class methods

aeif_cond_alpha_alt_neuron_nestml::~aeif_cond_alpha_alt_neuron_nestml()
{
  Free();
}


int aeif_cond_alpha_alt_neuron_nestml::Init(int i_node_0,
                                            int n_node,
                                            int /*n_port*/,
                                            int i_group,
                                            unsigned long long* seed)
{
  BaseNeuron::Init(i_node_0, n_node, 2 /*n_port*/, i_group, seed);

  node_type_ = i_aeif_cond_alpha_alt_neuron_nestml_model;


  /*
   * State variables
   */
  n_scal_var_ = N_SCAL_VAR;
  n_port_var_ = N_PORT_VAR;
  n_var_      = n_scal_var_ + n_port_var_;


  /*
   * Parameters
   */
  n_scal_param_ = N_SCAL_PARAM;
  n_param_      = n_scal_param_;


  AllocParamArr();
  AllocVarArr();


  scal_var_name_   = aeif_cond_alpha_alt_neuron_nestml_scal_var_name;
  scal_param_name_ = aeif_cond_alpha_alt_neuron_nestml_scal_param_name;
  port_var_name_   = aeif_cond_alpha_alt_neuron_nestml_port_var_name;


  /*
   * Parameters
   */
  SetScalParam(0, n_node, "C_m",         281.0);    // pF
  SetScalParam(0, n_node, "refr_T",        2.0);    // ms
  SetScalParam(0, n_node, "V_reset",     -60.0);    // mV
  SetScalParam(0, n_node, "g_L",          30.0);    // nS
  SetScalParam(0, n_node, "E_L",         -70.6);    // mV
  SetScalParam(0, n_node, "a",             4.0);    // nS
  SetScalParam(0, n_node, "b",            80.5);    // pA
  SetScalParam(0, n_node, "Delta_T",       2.0);    // mV
  SetScalParam(0, n_node, "tau_w",       144.0);    // ms
  SetScalParam(0, n_node, "V_th",        -50.4);    // mV
  SetScalParam(0, n_node, "V_peak",        0.0);    // mV
  SetScalParam(0, n_node, "tau_syn_exc",   0.2);    // ms
  SetScalParam(0, n_node, "tau_syn_inh",   2.0);    // ms
  SetScalParam(0, n_node, "E_exc",         0.0);    // mV
  SetScalParam(0, n_node, "E_inh",       -85.0);    // mV
  SetScalParam(0, n_node, "I_e",           0.0);    // pA
  SetScalParam(0, n_node, "I_stim",        0.0);    // pA


  /*
   * State variables
   */
  SetScalVar(0, n_node, "V_m",       *GetScalParam(0, n_node, "E_L"));
  SetScalVar(0, n_node, "w",          0.0);
  SetScalVar(0, n_node, "refr_t",     0.0);
  SetScalVar(0, n_node, "g_exc",      0.0);
  SetScalVar(0, n_node, "g_exc__d",   0.0);
  SetScalVar(0, n_node, "g_inh",      0.0);
  SetScalVar(0, n_node, "g_inh__d",   0.0);


  /*
   * Compact state buffer for Boost.Odeint's built-in RK45 / Dopri5 stepper.
   *
   * Size = n_node_ * N_SCAL_VAR
   */
  ode_state_ = new thrust::device_vector<float>(
      static_cast<size_t>(n_node_) * static_cast<size_t>(N_SCAL_VAR));


  /*
   * Multiplication factor of input signal is always 1 for all nodes.
   */
  float input_weight = 1.0f;

  gpuErrchk(cudaMalloc(&port_weight_arr_, sizeof(float)));

  gpuErrchk(cudaMemcpy(port_weight_arr_,
                       &input_weight,
                       sizeof(float),
                       cudaMemcpyHostToDevice));

  port_weight_arr_step_  = 0;
  port_weight_port_step_ = 0;


  /*
   * Process the input spikes.
   */
  port_input_arr_       = GetVarArr() + n_scal_var_ + GetPortVarIdx("exc_spikes");
  port_input_arr_step_  = n_var_;
  port_input_port_step_ = 1;


  return 0;
}


int aeif_cond_alpha_alt_neuron_nestml::Calibrate(double /*time_min*/,
                                                 float /*time_resolution*/)
{
  /*
   * Boost.Odeint path does not need old RK5 calibration.
   */
  return 0;
}


int aeif_cond_alpha_alt_neuron_nestml::Update(long long /*it*/, double t1)
{
  float dt = 0.0f;

  gpuErrchk(cudaMemcpyFromSymbol(&dt,
                                 NESTGPUTimeResolution,
                                 sizeof(float)));

  const float t0    = static_cast<float>(t1) - dt;
  const float t_end = static_cast<float>(t1);


  /*
   * Important:
   *
   * Apply incoming spikes before ODE integration.
   *
   * This makes the event timing closer to built-in aeif_cond_alpha,
   * where incoming conductance increments are visible to the RK solver
   * during the current integration interval.
   */
  aeif_cond_alpha_alt_neuron_nestml_PreUpdate
      <<< (n_node_ + 1023) / 1024, 1024 >>>(
          n_node_,
          var_arr_,
          param_arr_,
          n_var_,
           gpuErrchk(cudaMemcpyFromSymbol(&dt,
                                 NESTGPUTimeResolution,
                                 sizeof(float)));

  const float t0    = static_cast n_param_);

  gpuErrchk(cudaPeekAtLastError());


  float* ode_state_ptr = thrust::raw_pointer_cast(ode_state_->data());


  /*
   * Copy current NEST GPU scalar state into compact Odeint state.
   *
   * This is done after PreUpdate, so incoming spikes already affected
   * g_exc__d / g_inh__d before integration starts.
   */
  aeif_cond_alpha_alt_neuron_nestml_CopyNestToOdeState
      <<< (n_node_ + 1023) / 1024, 1024 >>>(
          n_node_,
          var_arr_,
          n_var_,
          ode_state_ptr);

  gpuErrchk(cudaPeekAtLastError());


  /*
   * Built-in Boost.Odeint Dormand-Prince RK45 stepper.
   *
   * This replaces the old separate solver.
   *
   * Built-in aeif_cond_alpha uses h0_rel = 1.0e-2 by default.
   * Therefore we use dt * 1.0e-2 as initial Odeint step size instead of dt.
   */
  typedef boost::numeric::odeint::runge_kutta_dopri5<
      ode_state_type,
      value_type,
      ode_state_type,
      value_type,
      boost::numeric::odeint::thrust_algebra,
      boost::numeric::odeint::thrust_operations> stepper_type;


  AeifCondAlphaAltOdeintSystem system;
  system.param        = param_arr_;
  system.param_stride = n_param_;
  system.n_neuron     = n_node_;


  const float odeint_h0 = dt * 1.0e-2f;


  boost::numeric::odeint::integrate_adaptive(
      boost::numeric::odeint::make_controlled(
          1.0e-5f,  // absolute tolerance
          1.0e-4f,  // relative tolerance
          stepper_type()),
      boost::ref(system),
      *ode_state_,
      t0,
      t_end,
      odeint_h0);


  /*
   * Copy integrated scalar state back to normal NEST GPU state array.
   * Port variables are not touched here.
   */
  aeif_cond_alpha_alt_neuron_nestml_CopyOdeStateToNest
      <<< (n_node_ + 1023) / 1024, 1024 >>>(
          n_node_,
          var_arr_,
          n_var_,
          ode_state_ptr);

  gpuErrchk(cudaPeekAtLastError());


  /*
   * Handle threshold, reset and PushSpike after ODE integration.
   */
  aeif_cond_alpha_alt_neuron_nestml_PostUpdate
      <<< (n_node_ + 1023) / 1024, 1024 >>>(
          n_node_,
          i_node_0_,
          var_arr_,
          param_arr_,
          n_var_,
          n_param_);

  gpuErrchk(cudaPeekAtLastError());


  return 0;
}


int aeif_cond_alpha_alt_neuron_nestml::Free()
{
  delete ode_state_;
  ode_state_ = nullptr;

  FreeVarArr();
  FreeParamArr();

  return 0;
}
