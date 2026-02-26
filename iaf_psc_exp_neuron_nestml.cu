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

/*
 * ============================================================
 * POST-INTEGRATION KERNEL (needed for numeric solver)
 * ============================================================
 *
 * Handles:
 *   - onReceive: apply buffered spikes to synaptic currents
 *   - onCondition: threshold check, reset, spike emission
 *
 * This separation matches the NEST/NESTML pattern:
 *   integrate ODEs first  -> then handle events.
 */
__global__ void iaf_psc_exp_neuron_nestml_PostUpdate(
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

  // ----------------------------
  // onReceive (spike input ports)
  // ----------------------------
  // Port variables are stored after scalar variables: var[N_SCAL_VAR + port_idx]
  if (var[N_SCAL_VAR + i_exc_spikes] != 0.0f)
  {
    // Same scaling as your analytic onReceive:
    // (0.001 * multiplicity) * 1000 -> effectively multiplicity
    var[i_I_syn_exc] += (0.001f * var[N_SCAL_VAR + i_exc_spikes]) * 1000.0f;
    var[N_SCAL_VAR + i_exc_spikes] = 0.0f;
  }

  if (var[N_SCAL_VAR + i_inh_spikes] != 0.0f)
  {
    var[i_I_syn_inh] += (0.001f * var[N_SCAL_VAR + i_inh_spikes]) * 1000.0f;
    var[N_SCAL_VAR + i_inh_spikes] = 0.0f;
  }

  // ----------------------------
  // onCondition (threshold/spike)
  // ----------------------------
  if (var[i_refr_t] <= 0.0f && var[i_V_m] >= param[i_V_th])
  {
    var[i_refr_t] = param[i_refr_T];
    var[i_V_m]    = param[i_V_reset];
    PushSpike(i_node_0 + i_neuron, 1.0);
  }
}

/*
 * ============================================================
 * ANALYTIC CALIBRATION KERNEL
 * ============================================================
 *
 * Used only by analytic solver.
 * Numeric solver does not need these precomputed coefficients.
 */
__global__ void iaf_psc_exp_neuron_nestml_Calibrate(
    int n_node,
    float* param_arr,
    int n_param,
    float h)
{
  const int i_neuron = threadIdx.x + blockIdx.x * blockDim.x;
  if (i_neuron >= n_node)
    return;

  float* param = param_arr + n_param * i_neuron;
  param[i___h] = h; // ms

  param[i___P__I_syn_exc__I_syn_exc] = expf((-param[i___h]) / param[i_tau_syn_exc]);
  param[i___P__I_syn_inh__I_syn_inh] = expf((-param[i___h]) / param[i_tau_syn_inh]);

  param[i___P__V_m__I_syn_exc] =
    param[i_tau_m] * param[i_tau_syn_exc] *
    ((-expf(param[i___h] / param[i_tau_m])) + expf(param[i___h] / param[i_tau_syn_exc])) *
    expf((-param[i___h]) * (param[i_tau_m] + param[i_tau_syn_exc]) /
         (param[i_tau_m] * param[i_tau_syn_exc])) /
    (param[i_C_m] * (param[i_tau_m] - param[i_tau_syn_exc]));

  param[i___P__V_m__I_syn_inh] =
    param[i_tau_m] * param[i_tau_syn_inh] *
    (expf(param[i___h] / param[i_tau_m]) - expf(param[i___h] / param[i_tau_syn_inh])) *
    expf((-param[i___h]) * (param[i_tau_m] + param[i_tau_syn_inh]) /
         (param[i_tau_m] * param[i_tau_syn_inh])) /
    (param[i_C_m] * (param[i_tau_m] - param[i_tau_syn_inh]));

  param[i___P__V_m__V_m] = expf((-param[i___h]) / param[i_tau_m]);
  param[i___P__refr_t__refr_t] = 1.0f;
}

/*
 * ============================================================
 * ANALYTIC UPDATE KERNEL (unchanged)
 * ============================================================
 */
__global__ void iaf_psc_exp_neuron_nestml_Update(
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
    // integrate_odes(I_syn_exc, I_syn_inh, refr_t)
    const double I_syn_exc__tmp = var[i_I_syn_exc] * param[i___P__I_syn_exc__I_syn_exc];
    const double I_syn_inh__tmp = var[i_I_syn_inh] * param[i___P__I_syn_inh__I_syn_inh];
    const double refr_t__tmp    = param[i___P__refr_t__refr_t] * var[i_refr_t] - 1.0 * param[i___h];

    var[i_I_syn_exc] = (float) I_syn_exc__tmp;
    var[i_I_syn_inh] = (float) I_syn_inh__tmp;
    var[i_refr_t]    = (float) refr_t__tmp;
  }
  else
  {
    // integrate_odes(I_syn_exc, I_syn_inh, V_m)
    const double I_syn_exc__tmp = var[i_I_syn_exc] * param[i___P__I_syn_exc__I_syn_exc];
    const double I_syn_inh__tmp = var[i_I_syn_inh] * param[i___P__I_syn_inh__I_syn_inh];
    const double V_m__tmp =
      (-param[i_E_L]) * param[i___P__V_m__V_m] + param[i_E_L]
      + var[i_I_syn_exc] * param[i___P__V_m__I_syn_exc]
      + var[i_I_syn_inh] * param[i___P__V_m__I_syn_inh]
      + var[i_V_m] * param[i___P__V_m__V_m]
      - param[i_I_e]   * param[i___P__V_m__V_m] * param[i_tau_m] / param[i_C_m]
      + param[i_I_e]   * param[i_tau_m] / param[i_C_m]
      - param[i_I_stim]* param[i___P__V_m__V_m] * param[i_tau_m] / param[i_C_m]
      + param[i_I_stim]* param[i_tau_m] / param[i_C_m];

    var[i_I_syn_exc] = (float) I_syn_exc__tmp;
    var[i_I_syn_inh] = (float) I_syn_inh__tmp;
    var[i_V_m]       = (float) V_m__tmp;
  }

  // onReceive
  if (var[N_SCAL_VAR + i_exc_spikes] != 0.0f)
  {
    var[i_I_syn_exc] += (0.001f * var[N_SCAL_VAR + i_exc_spikes]) * 1000.0f;
    var[N_SCAL_VAR + i_exc_spikes] = 0.0f;
  }
  if (var[N_SCAL_VAR + i_inh_spikes] != 0.0f)
  {
    var[i_I_syn_inh] += (0.001f * var[N_SCAL_VAR + i_inh_spikes]) * 1000.0f;
    var[N_SCAL_VAR + i_inh_spikes] = 0.0f;
  }

  // onCondition
  if (var[i_refr_t] <= 0.0f && var[i_V_m] >= param[i_V_th])
  {
    var[i_refr_t] = param[i_refr_T];
    var[i_V_m]    = param[i_V_reset];
    PushSpike(i_node_0 + i_neuron, 1.0);
  }
}

// ------------------------------------------------------------
// Class methods
// ------------------------------------------------------------

iaf_psc_exp_neuron_nestml::~iaf_psc_exp_neuron_nestml()
{
  // Ensure consistent cleanup (analytic + numeric)
  Free();
}

int iaf_psc_exp_neuron_nestml::Init(
    int i_node_0,
    int n_node,
    int /*n_port*/,
    int i_group)
{
  BaseNeuron::Init(i_node_0, n_node, 2 /*n_port*/, i_group);
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

#if USE_ODEINT_THRUST
  // Host-driven numeric solver.
  // We pass var_arr_ and param_arr_ (device pointers) + their strides.
  odeint_solver_ = new IafPscExpOdeintSolver(
      n_node_, var_arr_, n_var_,
      param_arr_, n_param_);
#endif

  scal_var_name_   = iaf_psc_exp_neuron_nestml_scal_var_name;
  scal_param_name_ = iaf_psc_exp_neuron_nestml_scal_param_name;
  port_var_name_   = iaf_psc_exp_neuron_nestml_port_var_name;

  // Parameters (defaults)
  SetScalParam(0, n_node, "C_m",         250);   // pF
  SetScalParam(0, n_node, "tau_m",       10);    // ms
  SetScalParam(0, n_node, "tau_syn_inh", 2);     // ms
  SetScalParam(0, n_node, "tau_syn_exc", 2);     // ms
  SetScalParam(0, n_node, "refr_T",      2);     // ms
  SetScalParam(0, n_node, "E_L",         -70);   // mV
  SetScalParam(0, n_node, "V_reset",     -70);   // mV
  SetScalParam(0, n_node, "V_th",        -55);   // mV
  SetScalParam(0, n_node, "I_e",         0);     // pA

  // Internal variables (analytic solver precomputation)
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
  SetScalVar(0, n_node, "V_m", *GetScalParam(0, n_node, "E_L"));
  SetScalVar(0, n_node, "refr_t", 0);
  SetScalVar(0, n_node, "I_syn_exc", 0);
  SetScalVar(0, n_node, "I_syn_inh", 0);

  // Input weight (continuous input) - unchanged
  float input_weight = 1.0f;
  gpuErrchk(cudaMalloc(&port_weight_arr_, sizeof(float)));
  gpuErrchk(cudaMemcpy(port_weight_arr_, &input_weight,
                       sizeof(float), cudaMemcpyHostToDevice));
  port_weight_arr_step_      = 0;
  port_weight_port_step_     = 0;

  // Process the input spikes
  port_input_arr_            = GetVarArr() + n_scal_var_ + GetPortVarIdx("exc_spikes");
  port_input_arr_step_       = n_var_;
  port_input_port_step_      = 1;

  return 0;
}

int iaf_psc_exp_neuron_nestml::Calibrate(double /*time_min*/, float time_resolution)
{
#if !USE_ODEINT_THRUST
  // Analytic solver needs calibration coefficients
  iaf_psc_exp_neuron_nestml_Calibrate<<<(n_node_ + 1023) / 1024, 1024>>>(
      n_node_, param_arr_, n_param_, time_resolution);
#else
  // Numeric solver: no analytic precomputation required
  (void) time_resolution;
#endif
  return 0;
}

int iaf_psc_exp_neuron_nestml::Update(long long /*it*/, double t1)
{
#if USE_ODEINT_THRUST
  // Numeric integration (host-driven, device execution via Thrust)
  const float dt = NESTGPUTimeResolution;
  const float t0 = static_cast<float>(t1) - dt;

  odeint_solver_->Step(t0, dt);

  // Event handling after integration (device kernel)
  iaf_psc_exp_neuron_nestml_PostUpdate<<<(n_node_ + 1023) / 1024, 1024>>>(
      n_node_, i_node_0_, var_arr_, param_arr_, n_var_, n_param_);
#else
  // Original analytic solver (unchanged)
  iaf_psc_exp_neuron_nestml_Update<<<(n_node_ + 1023) / 1024, 1024>>>(
      n_node_, i_node_0_, var_arr_, param_arr_, n_var_, n_param_);
#endif

  return 0;
}

int iaf_psc_exp_neuron_nestml::Free()
{
#if USE_ODEINT_THRUST
  delete odeint_solver_;
  odeint_solver_ = nullptr;
#endif
  FreeVarArr();
  FreeParamArr();
  return 0;
}
