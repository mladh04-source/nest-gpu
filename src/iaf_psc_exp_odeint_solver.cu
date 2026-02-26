/*
 *  iaf_psc_exp_odeint_solver.cu
 *
 *  Numeric solver implementation for iaf_psc_exp using Thrust.
 *
 * MODEL (iaf_psc_exp ODEs):
 * -------------------------
 * Synaptic currents (exponential decay):
 *   dI_exc/dt = -I_exc / tau_syn_exc
 *   dI_inh/dt = -I_inh / tau_syn_inh
 *
 * Membrane potential:
 *   dV/dt = -(V - E_L)/tau_m + (I_exc + I_inh + I_e + I_stim)/C_m
 *
 * Refractory timer:
 *   if refr_t > 0: refr_t decreases with slope -1 (ms/ms)
 *   else: refr_t stays <= 0 (no integration needed)
 *
 * IMPORTANT:
 * ----------
 * No onReceive or threshold check here.
 * Because those are handled by iaf_psc_exp_neuron_nestml_PostUpdate().
 */

#include "iaf_psc_exp_odeint_solver.h"

// Keep indices consistent with iaf_psc_exp_neuron_nestml.h
// (duplicated here to avoid circular includes)
namespace iaf_psc_exp_idx
{
  // var indices
  constexpr int i_V_m      = 0;
  constexpr int i_refr_t   = 1;
  constexpr int i_I_exc    = 2;
  constexpr int i_I_inh    = 3;

  // param indices
  constexpr int i_C_m         = 0;
  constexpr int i_tau_m       = 1;
  constexpr int i_tau_syn_inh = 2;
  constexpr int i_tau_syn_exc = 3;
  constexpr int i_refr_T      = 4;
  constexpr int i_E_L         = 5;
  constexpr int i_V_reset     = 6;
  constexpr int i_V_th        = 7;
  constexpr int i_I_e         = 8;
  // ... analytic precomp indices omitted
  constexpr int i_I_stim      = 16;
}

// ------------------------------------------------------------
// Device functor (Euler step) executed by Thrust
// ------------------------------------------------------------
struct IafPscExpEulerFunctor
{
  float* var;
  int var_stride;
  float* par;
  int par_stride;
  float dt;

  __host__ __device__
  void operator()(int i_neuron) const
  {
    float* y = var + i_neuron * var_stride;
    float* p = par + i_neuron * par_stride;

    const float V      = y[iaf_psc_exp_idx::i_V_m];
    const float refr_t = y[iaf_psc_exp_idx::i_refr_t];
    const float Iexc   = y[iaf_psc_exp_idx::i_I_exc];
    const float Iinh   = y[iaf_psc_exp_idx::i_I_inh];

    const float C_m         = p[iaf_psc_exp_idx::i_C_m];
    const float tau_m       = p[iaf_psc_exp_idx::i_tau_m];
    const float tau_syn_exc = p[iaf_psc_exp_idx::i_tau_syn_exc];
    const float tau_syn_inh = p[iaf_psc_exp_idx::i_tau_syn_inh];
    const float E_L         = p[iaf_psc_exp_idx::i_E_L];
    const float I_e         = p[iaf_psc_exp_idx::i_I_e];
    const float I_stim      = p[iaf_psc_exp_idx::i_I_stim];

    // ---- Synaptic current decay (always active) ----
    // Euler: I(t+dt) = I(t) + dt * (-I/tau)
    float Iexc_new = Iexc + dt * (-Iexc / tau_syn_exc);
    float Iinh_new = Iinh + dt * (-Iinh / tau_syn_inh);

    // ---- Refractory handling ----
    if (refr_t > 0.0f)
    {
      // Refractory: V_m is clamped (no change), refr_t decreases.
      float refr_new = refr_t - dt;

      y[iaf_psc_exp_idx::i_I_exc]  = Iexc_new;
      y[iaf_psc_exp_idx::i_I_inh]  = Iinh_new;
      y[iaf_psc_exp_idx::i_refr_t] = refr_new;
      // y[V_m] unchanged on purpose
      return;
    }

    // ---- Normal subthreshold dynamics ----
    // dV/dt = -(V - E_L)/tau_m + (Iexc + Iinh + I_e + I_stim)/C_m
    const float I_total = Iexc + Iinh + I_e + I_stim;
    const float dVdt = (-(V - E_L) / tau_m) + (I_total / C_m);

    const float V_new = V + dt * dVdt;

    y[iaf_psc_exp_idx::i_V_m]    = V_new;
    y[iaf_psc_exp_idx::i_I_exc]  = Iexc_new;
    y[iaf_psc_exp_idx::i_I_inh]  = Iinh_new;
    // refr_t remains <= 0 and is not advanced
  }
};

// ------------------------------------------------------------
// Solver class methods
// ------------------------------------------------------------

IafPscExpOdeintSolver::IafPscExpOdeintSolver(
    int n_neuron,
    float* var_arr,
    int var_stride,
    float* param_arr,
    int par_stride)
: n_(n_neuron)
, var_(var_arr)
, var_stride_(var_stride)
, par_(param_arr)
, par_stride_(par_stride)
{
  // Nothing else to initialize.
  // Ownership of var_/par_ stays in NEST GPU.
}

void IafPscExpOdeintSolver::Step(float /*t0*/, float dt)
{
  thrust::counting_iterator<int> begin(0);
  thrust::counting_iterator<int> end(n_);

  IafPscExpEulerFunctor f;
  f.var        = var_;
  f.var_stride = var_stride_;
  f.par        = par_;
  f.par_stride = par_stride_;
  f.dt         = dt;

  // Force device execution policy
  thrust::for_each(thrust::device, begin, end, f);
}
