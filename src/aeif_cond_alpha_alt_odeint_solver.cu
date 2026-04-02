/*
 *  aeif_cond_alpha_alt_odeint_solver.cu
 *
 *  Experimental numeric solver for aeif_cond_alpha_alt_neuron_nestml
 *  using a host-driven odeint/thrust execution model.
 *
 *  Info:
 *  ------
 *  This solver mimics the "host step -> device parallel update" idea.
 *  It is intentionally separated from event handling.
 *  onReceive / onCondition are executed afterwards in a dedicated kernel.
 *
 *  This preserves the original NESTML logic as much as possible while
 *  replacing the RK5 integration path with an experimental numeric path.
 */

#include <cmath>
#include <thrust/for_each.h>
#include <thrust/iterator/counting_iterator.h>

#include "aeif_cond_alpha_alt_odeint_solver.h"
#include "aeif_cond_alpha_alt_neuron_nestml.h"

using namespace aeif_cond_alpha_alt_neuron_nestml_ns;

/*
 * ============================================================
 * Device functor executed by Thrust
 * ============================================================
 *
 * Each invocation updates ONE neuron.
 * The functor performs a simple explicit Euler step for the ODEs.
 */
struct AeifCondAlphaAltEulerStepFunctor
{
  float* var;
  int var_stride;
  float* param;
  int param_stride;
  float dt;

  __host__ __device__
  void operator()(int i_neuron) const
  {
    float* y = var + i_neuron * var_stride;
    float* p = param + i_neuron * param_stride;

    // Local copies for readability
    const float V_m      = y[i_V_m];
    const float w        = y[i_w];
    const float refr_t   = y[i_refr_t];
    const float g_exc    = y[i_g_exc];
    const float g_exc_d  = y[i_g_exc__d];
    const float g_inh    = y[i_g_inh];
    const float g_inh_d  = y[i_g_inh__d];

    const float V_bounded = (V_m < p[i_V_peak]) ? V_m : p[i_V_peak];

    // Common synapse derivatives
    const float dg_exc    = g_exc_d;
    const float dg_exc_d  = (-g_exc) / (p[i_tau_syn_exc] * p[i_tau_syn_exc])
                            - 2.0f * g_exc_d / p[i_tau_syn_exc];

    const float dg_inh    = g_inh_d;
    const float dg_inh_d  = (-g_inh) / (p[i_tau_syn_inh] * p[i_tau_syn_inh])
                            - 2.0f * g_inh_d / p[i_tau_syn_inh];

    float dV = 0.0f;
    float dw = 0.0f;
    float drefr = 0.0f;

    if (refr_t > 0.0f)
    {
      // Refractory branch: V_m frozen, refr_t decreases, w evolves
      dV = 0.0f;

      dw =
        p[i_a] * ((-p[i_E_L]) / p[i_tau_w] + V_bounded / p[i_tau_w])
        - w / p[i_tau_w];

      drefr = -1.0f;
    }
    else
    {
      // Normal branch: V_m, w evolve; refr_t remains zero
      dV =
        ( p[i_Delta_T] * p[i_g_L] * expf(((-p[i_V_th]) + V_bounded) / p[i_Delta_T])
          + p[i_E_L]  * p[i_g_L]
          + p[i_E_exc] * g_exc
          + p[i_E_inh] * g_inh
          + p[i_I_e]
          + p[i_I_stim]
          - p[i_g_L] * V_bounded
          - g_exc * V_bounded
          - g_inh * V_bounded
          - w ) / p[i_C_m];

      dw =
        p[i_a] * ((-p[i_E_L]) / p[i_tau_w] + V_bounded / p[i_tau_w])
        - w / p[i_tau_w];

      drefr = 0.0f;
    }

    // Explicit Euler step
    y[i_V_m]      = V_m     + dt * dV;
    y[i_w]        = w       + dt * dw;
    y[i_refr_t]   = refr_t  + dt * drefr;
    y[i_g_exc]    = g_exc   + dt * dg_exc;
    y[i_g_exc__d] = g_exc_d + dt * dg_exc_d;
    y[i_g_inh]    = g_inh   + dt * dg_inh;
    y[i_g_inh__d] = g_inh_d + dt * dg_inh_d;

    /*
     * Port variables are intentionally NOT integrated here:
     *   y[N_SCAL_VAR + i_exc_spikes]
     *   y[N_SCAL_VAR + i_inh_spikes]
     *
     * They are consumed later by the PostUpdate kernel,
     * (matching the same separation used in the iaf_psc_exp experiment).
     */
  }
};

//Class implementation

AeifCondAlphaAltOdeintSolver::AeifCondAlphaAltOdeintSolver(
    int n_neuron,
    float* var_arr,
    int var_stride,
    float* param_arr,
    int param_stride)
: n_(n_neuron)
, var_(var_arr)
, var_stride_(var_stride)
, param_(param_arr)
, param_stride_(param_stride)
{
  // Nothing else to initialize.
  // Memory ownership remains in NEST GPU.
}

void AeifCondAlphaAltOdeintSolver::Step(float /*t0*/, float dt)
{
  /*
   * IMPORTANT:
   * This method is called from the HOST.
   * Thrust launches device kernels internally,
   * but control flow remains host-driven.
   *
   * This differs from the original RK5 implementation and is
   * what we want to investigate.
   */

  thrust::counting_iterator<int> begin(0);
  thrust::counting_iterator<int> end(n_);

  AeifCondAlphaAltEulerStepFunctor functor;
  functor.var = var_;
  functor.var_stride = var_stride_;
  functor.param = param_;
  functor.param_stride = param_stride_;
  functor.dt = dt;

  thrust::for_each(begin, end, functor);

  /*
   * After this step:
   * ODE states have been advanced numerically
   * NO event handling has happened yet
   *
   * The caller must then launch a separate PostUpdate kernel
   * for: onReceive, onCondition and spike emission
   */
}
