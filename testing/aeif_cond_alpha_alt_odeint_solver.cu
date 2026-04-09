/*
 *  aeif_cond_alpha_alt_odeint_solver.cu
 *
 *  Experimental numeric solver for aeif_cond_alpha_alt_neuron_nestml
 *  using a host-driven odeint/thrust-style execution model.
 *
 *  IMPORTANT:
 *  This solver advances ONLY the ODE state variables.
 *  It does NOT execute onReceive or onCondition.
 *  Event handling must be done afterwards by a separate PostUpdate kernel.
 *
 *  Current implementation:
 *  explicit Euler step
 *  one Thrust functor call per neuron
 *  operates directly on NEST GPU raw arrays
 */

#include <cmath>
#include <thrust/for_each.h>
#include <thrust/iterator/counting_iterator.h>

#include "aeif_cond_alpha_alt_odeint_solver.h"
#include "aeif_cond_alpha_alt_neuron_nestml.h"

using namespace aeif_cond_alpha_alt_neuron_nestml_ns;

/*
 * Device functor executed by Thrust
 * Each invocation updates exactly one neuron.
 * The functor performs one explicit Euler step for the ODE system.
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
    float* y = var   + i_neuron * var_stride;
    float* p = param + i_neuron * param_stride;

    // Current state
    const float V_m     = y[i_V_m];
    const float w       = y[i_w];
    const float refr_t  = y[i_refr_t];
    const float g_exc   = y[i_g_exc];
    const float g_exc_d = y[i_g_exc__d];
    const float g_inh   = y[i_g_inh];
    const float g_inh_d = y[i_g_inh__d];

    // Clamp membrane potential in the same way as the original model
    const float V_bounded = (V_m < p[i_V_peak]) ? V_m : p[i_V_peak];

    const float tau_syn_exc_sq = p[i_tau_syn_exc] * p[i_tau_syn_exc];
    const float tau_syn_inh_sq = p[i_tau_syn_inh] * p[i_tau_syn_inh];

    // Synaptic alpha-system derivatives
    const float dg_exc   = g_exc_d;
    const float dg_exc_d = -g_exc / tau_syn_exc_sq
                           - 2.0f * g_exc_d / p[i_tau_syn_exc];

    const float dg_inh   = g_inh_d;
    const float dg_inh_d = -g_inh / tau_syn_inh_sq
                           - 2.0f * g_inh_d / p[i_tau_syn_inh];

    float dV    = 0.0f;
    float dw    = 0.0f;
    float drefr = 0.0f;

    // Adaptation variable evolves in both branches
    dw = p[i_a] * ((-p[i_E_L] + V_bounded) / p[i_tau_w])
         - w / p[i_tau_w];

    if (refr_t > 0.0f)
    {
      // Refractory branch:
      // V_m frozen, refr_t decreases, w still evolves
      dV    = 0.0f;
      drefr = -1.0f;
    }
    else
    {
      // Normal branch:
      // V_m and w evolve, refr_t remains zero
      dV =
        ( p[i_Delta_T] * p[i_g_L]
            * expf((-p[i_V_th] + V_bounded) / p[i_Delta_T])
          + p[i_E_L]  * p[i_g_L]
          + p[i_E_exc] * g_exc
          + p[i_E_inh] * g_inh
          + p[i_I_e]
          + p[i_I_stim]
          - p[i_g_L] * V_bounded
          - g_exc * V_bounded
          - g_inh * V_bounded
          - w ) / p[i_C_m];

      drefr = 0.0f;
    }

    // Explicit Euler update
    y[i_V_m]      = V_m     + dt * dV;
    y[i_w]        = w       + dt * dw;
    y[i_g_exc]    = g_exc   + dt * dg_exc;
    y[i_g_exc__d] = g_exc_d + dt * dg_exc_d;
    y[i_g_inh]    = g_inh   + dt * dg_inh;
    y[i_g_inh__d] = g_inh_d + dt * dg_inh_d;

    // Keep refractory counter numerically well-behaved
    const float refr_new = refr_t + dt * drefr;
    y[i_refr_t] = (refr_new > 0.0f) ? refr_new : 0.0f;

    /*
     * Port variables are intentionally NOT updated here:
     *   y[N_SCAL_VAR + i_exc_spikes]
     *   y[N_SCAL_VAR + i_inh_spikes]
     * They are consumed afterwards by the PostUpdate kernel.
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
  // Memory ownership remains in NEST GPU / neuron class.
}

void AeifCondAlphaAltOdeintSolver::Step(float /*t0*/, float dt)
{
  /*
   * This method is called on the HOST.
   * Thrust launches device-side work internally.
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
   * NO onReceive / onCondition logic has been executed yet
   *
   * The caller must launch the separate PostUpdate kernel afterwards.
   */
}
