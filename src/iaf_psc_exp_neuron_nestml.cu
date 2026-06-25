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

#if USE_ODEINT_THRUST
#include <boost/numeric/odeint.hpp>
#include <boost/numeric/odeint/external/thrust/thrust.hpp>
#include <boost/ref.hpp>

#include <thrust/device_vector.h>
#include <thrust/device_ptr.h>
#include <thrust/for_each.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/system/cuda/execution_policy.h>
#endif

using namespace iaf_psc_exp_neuron_nestml_ns;


#if USE_ODEINT_THRUST

namespace
{

typedef float value_type;
typedef thrust::device_vector<value_type> ode_state_type;

/*
 * Copy scalar NEST GPU state variables into a compact Odeint state vector.
 *
 * NEST layout:var_arr[i_neuron * n_var + scalar_index]
 * Odeint layout:ode_state[i_neuron * N_SCAL_VAR + scalar_index]
 * Port variables are intentionally not copied.
 */
__global__
void iaf_psc_exp_neuron_nestml_CopyNestToOdeState(
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
 * Port variables remain untouched here and are consumed by PostUpdate.
 */
__global__
void iaf_psc_exp_neuron_nestml_CopyOdeStateToNest(
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
 * Odeint calls IafPscExpOdeintSystem on the host.
 * That system function launches this Thrust functor on the GPU.
 */
struct IafPscExpDerivativeFunctor
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

    const float V      = y[i_V_m];
    const float refr_t = y[i_refr_t];
    const float Iexc   = y[i_I_syn_exc];
    const float Iinh   = y[i_I_syn_inh];

    const float C_m         = p[i_C_m];
    const float tau_m       = p[i_tau_m];
    const float tau_syn_exc = p[i_tau_syn_exc];
    const float tau_syn_inh = p[i_tau_syn_inh];
    const float E_L         = p[i_E_L];
    const float I_e         = p[i_I_e];
    const float I_stim      = p[i_I_stim];

    /*
     * Synaptic current decay.
     * Synaptic currents decay in both normal and refractory states.
     */
    dydt[i_I_syn_exc] = -Iexc / tau_syn_exc;
    dydt[i_I_syn_inh] = -Iinh / tau_syn_inh;

    if (refr_t > 0.0f)
    {
      /*
       * Refractory branch:
       * membrane potential is frozen,
       * refractory timer decreases.
       */
      dydt[i_V_m]    = 0.0f;
      dydt[i_refr_t] = -1.0f;
    }
    else
    {
      /*
       * Normal subthreshold dynamics.
       * Inhibitory current is subtracted, matching the old numeric solver.
       */
      dydt[i_V_m] =
          (-(V - E_L) / tau_m)
          + ((Iexc - Iinh + I_e + I_stim) / C_m);

      dydt[i_refr_t] = 0.0f;
    }
  }
};


/*
 * Boost.Odeint system function.
 * x : compact scalar state vector on GPU
 * dxdt : compact derivative vector on GPU
 */
struct IafPscExpOdeintSystem
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

    IafPscExpDerivativeFunctor functor;
    functor.x = x_ptr;
    functor.dxdt = dxdt_ptr;
    functor.param = param;
    functor.param_stride = param_stride;

    /*
     * CUDA execution policy.
     * This is more robust than thrust::device on newer CUDA/Thrust versions.
     */
    thrust::for_each(thrust::cuda::par, begin, end, functor);
  }
};

} // anonymous namespace

#endif // USE_ODEINT_THRUST


/*
 * POST INTEGRATION KERNEL
 *
 * Used by Boost.Odeint numeric solver.
 * Handles: onReceive and onCondition.
 */
__global__
void iaf_psc_exp_neuron_nestml_PostUpdate(
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
   * onReceive: excitatory spikes
   *
   * Port variables are stored after scalar variables:
   *   var[N_SCAL_VAR + port_idx]
   */
  if (var[N_SCAL_VAR + i_exc_spikes] != 0.0f)
  {
    var[i_I_syn_exc] +=
      (0.001f * var[N_SCAL_VAR + i_exc_spikes]) * 1000.0f;

    var[N_SCAL_VAR + i_exc_spikes] = 0.0f;
  }

  /*
   * onReceive: inhibitory spikes
   */
  if (var[N_SCAL_VAR + i_inh_spikes] != 0.0f)
  {
    var[i_I_syn_inh] +=
      (0.001f * var[N_SCAL_VAR + i_inh_spikes]) * 1000.0f;

    var[N_SCAL_VAR + i_inh_spikes] = 0.0f;
  }

  /*
   * onCondition: threshold, reset, spike emission
   */
  if (var[i_refr_t] <= 0.0f && var[i_V_m] >= param[i_V_th])
  {
    var[i_refr_t] = param[i_refr_T];
    var[i_V_m]    = param[i_V_reset];

    PushSpike(i_node_0 + i_neuron, 1.0f);
  }

  /*
   * Final safety reset of input ports.
   */
  var[N_SCAL_VAR + i_exc_spikes] = 0.0f;
  var[N_SCAL_VAR + i_inh_spikes] = 0.0f;
}


/*
 * ANALYTIC CALIBRATION KERNEL
 *
 * Used only by analytic solver.
 * Numeric Boost.Odeint solver does not need these coefficients.
 */
__global__
void iaf_psc_exp_neuron_nestml_Calibrate(
    int n_node,
    float* param_arr,
    int n_param,
    float h)
{
  const int i_neuron = threadIdx.x + blockIdx.x * blockDim.x;

  if (i_neuron >= n_node)
    return;

  float* param = param_arr + n_param * i_neuron;

  param[i___h] = h;

  param[i___P__I_syn_exc__I_syn_exc] =
      expf((-param[i___h]) / param[i_tau_syn_exc]);

  param[i___P__I_syn_inh__I_syn_inh] =
      expf((-param[i___h]) / param[i_tau_syn_inh]);

  param[i___P__V_m__I_syn_exc] =
    param[i_tau_m] * param[i_tau_syn_exc] *
    ((-expf(param[i___h] / param[i_tau_m]))
      + expf(param[i___h] / param[i_tau_syn_exc])) *
    expf((-param[i___h]) * (param[i_tau_m] + param[i_tau_syn_exc]) /
         (param[i_tau_m] * param[i_tau_syn_exc])) /
    (param[i_C_m] * (param[i_tau_m] - param[i_tau_syn_exc]));

  param[i___P__V_m__I_syn_inh] =
    param[i_tau_m] * param[i_tau_syn_inh] *
    (expf(param[i___h] / param[i_tau_m])
      - expf(param[i___h] / param[i_tau_syn_inh])) *
    expf((-param[i___h]) * (param[i_tau_m] + param[i_tau_syn_inh]) /
         (param[i_tau_m] * param[i_tau_syn_inh])) /
    (param[i_C_m] * (param[i_tau_m] - param[i_tau_syn_inh]));

  param[i___P__V_m__V_m] =
      expf((-param[i___h]) / param[i_tau_m]);

  param[i___P__refr_t__refr_t] =
      1.0f;
}


/*
 * ANALYTIC UPDATE KERNEL
 *
 * This is the original analytic solver path.
 */
__global__
void iaf_psc_exp_neuron_nestml_Update(
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

  if (var[i_refr_t] > 0.0f)
  {
    const double I_syn_exc__tmp =
        var[i_I_syn_exc] * param[i___P__I_syn_exc__I_syn_exc];

    const double I_syn_inh__tmp =
        var[i_I_syn_inh] * param[i___P__I_syn_inh__I_syn_inh];

    const double refr_t__tmp =
        param[i___P__refr_t__refr_t] * var[i_refr_t]
        - 1.0 * param[i___h];

    var[i_I_syn_exc] = static_cast<float>(I_syn_exc__tmp);
    var[i_I_syn_inh] = static_cast<float>(I_syn_inh__tmp);
    var[i_refr_t]    = static_cast<float>(refr_t__tmp);

    if (var[i_refr_t] < 0.0f)
      var[i_refr_t] = 0.0f;
  }
  else
  {
    const double I_syn_exc__tmp =
        var[i_I_syn_exc] * param[i___P__I_syn_exc__I_syn_exc];

    const double I_syn_inh__tmp =
        var[i_I_syn_inh] * param[i___P__I_syn_inh__I_syn_inh];

    const double V_m__tmp =
      (-param[i_E_L]) * param[i___P__V_m__V_m] + param[i_E_L]
      + var[i_I_syn_exc] * param[i___P__V_m__I_syn_exc]
      + var[i_I_syn_inh] * param[i___P__V_m__I_syn_inh]
      + var[i_V_m] * param[i___P__V_m__V_m]
      - param[i_I_e]    * param[i___P__V_m__V_m] * param[i_tau_m] / param[i_C_m]
      + param[i_I_e]    * param[i_tau_m] / param[i_C_m]
      - param[i_I_stim] * param[i___P__V_m__V_m] * param[i_tau_m] / param[i_C_m]
      + param[i_I_stim] * param[i_tau_m] / param[i_C_m];

    var[i_I_syn_exc] = static_cast<float>(I_syn_exc__tmp);
    var[i_I_syn_inh] = static_cast<float>(I_syn_inh__tmp);
    var[i_V_m]       = static_cast<float>(V_m__tmp);
  }

  /*
   * onReceive
   */
  if (var[N_SCAL_VAR + i_exc_spikes] != 0.0f)
  {
    var[i_I_syn_exc] +=
      (0.001f * var[N_SCAL_VAR + i_exc_spikes]) * 1000.0f;

    var[N_SCAL_VAR + i_exc_spikes] = 0.0f;
  }

  if (var[N_SCAL_VAR + i_inh_spikes] != 0.0f)
  {
    var[i_I_syn_inh] +=
      (0.001f * var[N_SCAL_VAR + i_inh_spikes]) * 1000.0f;

    var[N_SCAL_VAR + i_inh_spikes] = 0.0f;
  }

  /*
   * onCondition
   */
  if (var[i_refr_t] <= 0.0f && var[i_V_m] >= param[i_V_th])
  {
    var[i_refr_t] = param[i_refr_T];
    var[i_V_m]    = param[i_V_reset];

    PushSpike(i_node_0 + i_neuron, 1.0f);
  }

  var[N_SCAL_VAR + i_exc_spikes] = 0.0f;
  var[N_SCAL_VAR + i_inh_spikes] = 0.0f;
}


// Class methods

iaf_psc_exp_neuron_nestml::~iaf_psc_exp_neuron_nestml()
{
  Free();
}


int iaf_psc_exp_neuron_nestml::Init(int i_node_0,
                                    int n_node,
                                    int /*n_port*/,
                                    int i_group,
                                    unsigned long long* seed)
{
  BaseNeuron::Init(i_node_0, n_node, 2 /*n_port*/, i_group, seed);

  node_type_ = i_iaf_psc_exp_neuron_nestml_model;

  // State variables
  n_scal_var_ = N_SCAL_VAR;
  n_port_var_ = N_PORT_VAR;
  n_var_      = n_scal_var_ + n_port_var_;

  // Parameters
  n_scal_param_ = N_SCAL_PARAM;
  n_param_      = n_scal_param_;

  AllocParamArr();
  AllocVarArr();

  scal_var_name_   = iaf_psc_exp_neuron_nestml_scal_var_name;
  scal_param_name_ = iaf_psc_exp_neuron_nestml_scal_param_name;
  port_var_name_   = iaf_psc_exp_neuron_nestml_port_var_name;

  // Parameters
  SetScalParam(0, n_node, "C_m",         250.0);   // pF
  SetScalParam(0, n_node, "tau_m",        10.0);   // ms
  SetScalParam(0, n_node, "tau_syn_inh",   2.0);   // ms
  SetScalParam(0, n_node, "tau_syn_exc",   2.0);   // ms
  SetScalParam(0, n_node, "refr_T",        2.0);   // ms
  SetScalParam(0, n_node, "E_L",         -70.0);   // mV
  SetScalParam(0, n_node, "V_reset",     -70.0);   // mV
  SetScalParam(0, n_node, "V_th",        -55.0);   // mV
  SetScalParam(0, n_node, "I_e",           0.0);   // pA

  // Internal variables for analytic solver precomputation
  SetScalParam(0, n_node, "__h", 0.0);
  SetScalParam(0, n_node, "__P__I_syn_exc__I_syn_exc", 0.0);
  SetScalParam(0, n_node, "__P__I_syn_inh__I_syn_inh", 0.0);
  SetScalParam(0, n_node, "__P__V_m__I_syn_exc", 0.0);
  SetScalParam(0, n_node, "__P__V_m__I_syn_inh", 0.0);
  SetScalParam(0, n_node, "__P__V_m__V_m", 0.0);
  SetScalParam(0, n_node, "__P__refr_t__refr_t", 0.0);

  // Continuous input port
  SetScalParam(0, n_node, "I_stim", 0.0);

  // State variables
  SetScalVar(0, n_node, "V_m",      *GetScalParam(0, n_node, "E_L"));
  SetScalVar(0, n_node, "refr_t",    0.0);
  SetScalVar(0, n_node, "I_syn_exc", 0.0);
  SetScalVar(0, n_node, "I_syn_inh", 0.0);

#if USE_ODEINT_THRUST
  /*
   * Compact state buffer for Boost.Odeint's built-in RK45 / Dopri5 stepper.
   *
   * Size: n_node_ * N_SCAL_VAR
   * This replaces the separate Solver object.
   */
  ode_state_ = new thrust::device_vector<float>(
      static_cast<size_t>(n_node_) * static_cast<size_t>(N_SCAL_VAR));
#endif

  // Multiplication factor of input signal is always 1 for all nodes
  float input_weight = 1.0f;

  gpuErrchk(cudaMalloc(&port_weight_arr_, sizeof(float)));

  gpuErrchk(cudaMemcpy(port_weight_arr_,
                       &input_weight,
                       sizeof(float),
                       cudaMemcpyHostToDevice));

  port_weight_arr_step_  = 0;
  port_weight_port_step_ = 0;

  // Process the input spikes
  port_input_arr_ =
    GetVarArr() + n_scal_var_ + GetPortVarIdx("exc_spikes");

  port_input_arr_step_  = n_var_;
  port_input_port_step_ = 1;

  return 0;
}


int iaf_psc_exp_neuron_nestml::Calibrate(double /*time_min*/,
                                         float time_resolution)
{
#if !USE_ODEINT_THRUST

  /*
   * Analytic solver needs precomputed coefficients.
   */
  iaf_psc_exp_neuron_nestml_Calibrate
      <<< (n_node_ + 1023) / 1024, 1024 >>>(
          n_node_,
          param_arr_,
          n_param_,
          time_resolution);

  gpuErrchk(cudaPeekAtLastError());

#else

  /*
   * Boost.Odeint numeric solver does not need analytic precomputation.
   */
  (void) time_resolution;

#endif

  return 0;
}


int iaf_psc_exp_neuron_nestml::Update(long long /*it*/, double t1)
{
#if USE_ODEINT_THRUST

  float dt = 0.0f;

  gpuErrchk(cudaMemcpyFromSymbol(&dt,
                                 NESTGPUTimeResolution,
                                 sizeof(float)));

  const float t0    = static_cast<float>(t1) - dt;
  const float t_end = static_cast<float>(t1);

  float* ode_state_ptr = thrust::raw_pointer_cast(ode_state_->data());

  /*
   * Copy current NEST GPU scalar state into compact Odeint state.
   */
  iaf_psc_exp_neuron_nestml_CopyNestToOdeState
      <<< (n_node_ + 1023) / 1024, 1024 >>>(
          n_node_,
          var_arr_,
          n_var_,
          ode_state_ptr);

  gpuErrchk(cudaPeekAtLastError());

  /*
   * Built-in Boost.Odeint Dormand-Prince RK45 stepper.
   *
   * This replaces the separate solver.
   */
  typedef boost::numeric::odeint::runge_kutta_dopri5<
      ode_state_type,
      value_type,
      ode_state_type,
      value_type,
      boost::numeric::odeint::thrust_algebra,
      boost::numeric::odeint::thrust_operations> stepper_type;

  IafPscExpOdeintSystem system;
  system.param        = param_arr_;
  system.param_stride = n_param_;
  system.n_neuron     = n_node_;

  boost::numeric::odeint::integrate_adaptive(
      boost::numeric::odeint::make_controlled(
          1.0e-5f,
          1.0e-4f,
          stepper_type()),
      boost::ref(system),
      *ode_state_,
      t0,
      t_end,
      dt);

  /*
   * Copy integrated scalar state back to normal NEST GPU state array.
   * Port variables are not touched here.
   */
  iaf_psc_exp_neuron_nestml_CopyOdeStateToNest
      <<< (n_node_ + 1023) / 1024, 1024 >>>(
          n_node_,
          var_arr_,
          n_var_,
          ode_state_ptr);

  gpuErrchk(cudaPeekAtLastError());

  /*
   * Handle onReceive and onCondition after ODE integration.
   */
  iaf_psc_exp_neuron_nestml_PostUpdate
      <<< (n_node_ + 1023) / 1024, 1024 >>>(
          n_node_,
          i_node_0_,
          var_arr_,
          param_arr_,
          n_var_,
          n_param_);

  gpuErrchk(cudaPeekAtLastError());

#else

  /*
   * Original analytic solver.
   */
  iaf_psc_exp_neuron_nestml_Update
      <<< (n_node_ + 1023) / 1024, 1024 >>>(
          n_node_,
          i_node_0_,
          var_arr_,
          param_arr_,
          n_var_,
          n_param_);

  gpuErrchk(cudaPeekAtLastError());

#endif

  return 0;
}


int iaf_psc_exp_neuron_nestml::Free()
{
#if USE_ODEINT_THRUST
  delete ode_state_;
  ode_state_ = nullptr;
#endif

  FreeVarArr();
  FreeParamArr();

  return 0;
}
