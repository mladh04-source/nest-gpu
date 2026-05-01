/*
 *  iaf_psc_exp_odeint_solver.cu
 *
 *  Numeric solver implementation for iaf_psc_exp using Thrust.
 *
 * MODEL (iaf_psc_exp ODEs):
 * Synaptic currents:
 *   dI_exc/dt = -I_exc / tau_syn_exc
 *   dI_inh/dt = -I_inh / tau_syn_inh
 *
 * Membrane potential:
 *   if refr_t > 0:
 *     dV_m/dt = 0
 *   else:
 *     dV_m/dt = -(V_m - E_L)/tau_m
 *               + (I_exc - I_inh + I_e + I_stim)/C_m
 *
 * Refractory timer:
 *   if refr_t > 0:
 *     refr_t decreases with slope -1
 *   else:
 *     refr_t remains unchanged
 *
 * IMPORTANT:
 * This solver advances ONLY the ODE state variables.
 * It does NOT execute onReceive or onCondition.
 *
 * Spike input handling, threshold detection, reset and PushSpike
 * are handled afterwards by iaf_psc_exp_neuron_nestml_PostUpdate().
 *
 * Current implementation (new):
 * adaptive Dormand-Prince RK45 stepper
 * one Thrust functor call per neuron
 * operates directly on NEST GPU raw arrays
 */

#include <cmath>
#include <thrust/for_each.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/execution_policy.h>

#include "iaf_psc_exp_odeint_solver.h"
#include "iaf_psc_exp_neuron_nestml.h"

using namespace iaf_psc_exp_neuron_nestml_ns;


/*
 * Device functor executed by Thrust.
 *
 * Each invocation updates exactly one neuron.
 * The functor performs an adaptive RK45 integration over one global
 * simulation step dt.
 *
 * The integrator only updates scalar ODE state variables.
 * Event handling is intentionally not part of this functor.
 */

struct IafPscExpRK45Functor
{
  float* var;
  int var_stride;
  float* par;
  int par_stride;
  float dt;

  /*
   * Small helper functions.
   *
   * These are marked __host__ __device__ because they are used inside
   * the Thrust functor, which is executed on the GPU.
   */

  __host__ __device__
  float abs_val(float x) const
  {
    return x < 0.0f ? -x : x;
  }

  __host__ __device__
  float min_val(float a, float b) const
  {
    return a < b ? a : b;
  }

  __host__ __device__
  float max_val(float a, float b) const
  {
    return a > b ? a : b;
  }

  /*
   * Compute the right-hand side of the ODE system.
   *
   * y : current temporary state vector
   * p  : parameter vector of the neuron
   * dydt : output derivative vector
   *
   * This function computes derivatives only.
   * It does not write to the global NEST GPU state array directly.
   */

  __host__ __device__
  void compute_derivatives(
      const float* y,
      const float* p,
      float* dydt) const
  {
    // Current state variables
    const float V      = y[i_V_m];
    const float refr_t = y[i_refr_t];
    const float Iexc   = y[i_I_syn_exc];
    const float Iinh   = y[i_I_syn_inh];

    // Neuron parameters
    const float C_m         = p[i_C_m];
    const float tau_m       = p[i_tau_m];
    const float tau_syn_exc = p[i_tau_syn_exc];
    const float tau_syn_inh = p[i_tau_syn_inh];
    const float E_L         = p[i_E_L];
    const float I_e         = p[i_I_e];
    const float I_stim      = p[i_I_stim];

    /*
     * Synaptic current decay.
     *
     * The synaptic currents are integrated in both normal and refractory
     * states.
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
       * Normal subthreshold dynamics:
       * membrane potential follows the iaf_psc_exp ODE.
       *
       * The inhibitory current is subtracted here because this is the
       * convention used by the current NESTML-generated state variables.
       */

      dydt[i_V_m] =
          (-(V - E_L) / tau_m)
          + ((Iexc - Iinh + I_e + I_stim) / C_m);

      // refr_t remains zero / non-positive outside refractory period
      dydt[i_refr_t] = 0.0f;
    }
  }

  /*
   * Main per-neuron update routine.
   *
   * This method is executed once per neuron by Thrust.
   * It copies the neuron state into local arrays, performs adaptive RK45
   * substeps over the interval [0, dt], and writes the final accepted
   * state back to the NEST GPU state array.
   */

  __host__ __device__
  void operator()(int i_neuron) const
  {
    float* y = var + i_neuron * var_stride;
    float* p = par + i_neuron * par_stride;

    /*
     * Local temporary storage.
     *
     * y_cur holds the currently accepted state.
     * y_tmp is used for intermediate RK stages.
     *
     * k1 to k7 are the Dormand-Prince RK45 stage derivatives.
     * y5 is the 5th-order solution.
     * y4 is the embedded 4th-order solution used for error estimation.
     */

    float y_cur[N_SCAL_VAR];
    float y_tmp[N_SCAL_VAR];

    float k1[N_SCAL_VAR];
    float k2[N_SCAL_VAR];
    float k3[N_SCAL_VAR];
    float k4[N_SCAL_VAR];
    float k5[N_SCAL_VAR];
    float k6[N_SCAL_VAR];
    float k7[N_SCAL_VAR];

    float y5[N_SCAL_VAR];
    float y4[N_SCAL_VAR];

    // Copy current global state into local per-neuron state
    for (int i = 0; i < N_SCAL_VAR; ++i)
      y_cur[i] = y[i];

    /*
     * Error tolerances for adaptive step-size control.
     *
     * abs_tol controls the absolute local error.
     * rel_tol controls the error relative to the magnitude of the state.
     */

    const float abs_tol = 1.0e-5f;
    const float rel_tol = 1.0e-4f;

    /*
     * Step-size adaptation parameters.
     *
     * safety prevents overly aggressive step-size growth.
     * min_factor and max_factor bound how much the step size may shrink
     * or grow after one attempted RK45 step.
     */

    const float safety = 0.9f;
    const float min_factor = 0.2f;
    const float max_factor = 5.0f;

    float t = 0.0f;
    float h = dt;

    /*
     * Maximum number of internal RK45 substeps.
     *
     * This prevents a single neuron from staying in the adaptive loop forever
     * if the requested tolerance cannot be reached.
     */

    const int max_steps = 100;
    int step_count = 0;

    /*
     * Integrate internally from t = 0 to t = dt.
     *
     * Each loop iteration attempts one adaptive RK45 substep of size h.
     * If the estimated local error is acceptable, the substep is accepted.
     * Otherwise, the state is not advanced and h is reduced.
     */

    while (t < dt && step_count < max_steps)
    {
      // Do not step beyond the requested global time step dt
      if (t + h > dt)
        h = dt - t;

      // k1
      compute_derivatives(y_cur, p, k1);

      // k2
      for (int i = 0; i < N_SCAL_VAR; ++i)
        y_tmp[i] = y_cur[i] + h * (1.0f / 5.0f) * k1[i];

      compute_derivatives(y_tmp, p, k2);

      // k3
      for (int i = 0; i < N_SCAL_VAR; ++i)
        y_tmp[i] = y_cur[i] + h *
          ((3.0f / 40.0f) * k1[i]
         + (9.0f / 40.0f) * k2[i]);

      compute_derivatives(y_tmp, p, k3);

      // k4
      for (int i = 0; i < N_SCAL_VAR; ++i)
        y_tmp[i] = y_cur[i] + h *
          ((44.0f / 45.0f) * k1[i]
         + (-56.0f / 15.0f) * k2[i]
         + (32.0f / 9.0f) * k3[i]);

      compute_derivatives(y_tmp, p, k4);

      // k5
      for (int i = 0; i < N_SCAL_VAR; ++i)
        y_tmp[i] = y_cur[i] + h *
          ((19372.0f / 6561.0f) * k1[i]
         + (-25360.0f / 2187.0f) * k2[i]
         + (64448.0f / 6561.0f) * k3[i]
         + (-212.0f / 729.0f) * k4[i]);

      compute_derivatives(y_tmp, p, k5);

      // k6
      for (int i = 0; i < N_SCAL_VAR; ++i)
        y_tmp[i] = y_cur[i] + h *
          ((9017.0f / 3168.0f) * k1[i]
         + (-355.0f / 33.0f) * k2[i]
         + (46732.0f / 5247.0f) * k3[i]
         + (49.0f / 176.0f) * k4[i]
         + (-5103.0f / 18656.0f) * k5[i]);

      compute_derivatives(y_tmp, p, k6);

      // k7
      for (int i = 0; i < N_SCAL_VAR; ++i)
        y_tmp[i] = y_cur[i] + h *
          ((35.0f / 384.0f) * k1[i]
         + (500.0f / 1113.0f) * k3[i]
         + (125.0f / 192.0f) * k4[i]
         + (-2187.0f / 6784.0f) * k5[i]
         + (11.0f / 84.0f) * k6[i]);

      compute_derivatives(y_tmp, p, k7);

      /*
       * 5th-order Dormand-Prince solution.
       *
       * This higher-order solution is used as the accepted state when
       * the local error estimate is small enough.
       */

      for (int i = 0; i < N_SCAL_VAR; ++i)
      {
        y5[i] = y_cur[i] + h *
          ((35.0f / 384.0f) * k1[i]
         + (500.0f / 1113.0f) * k3[i]
         + (125.0f / 192.0f) * k4[i]
         + (-2187.0f / 6784.0f) * k5[i]
         + (11.0f / 84.0f) * k6[i]);
      }

      /*
       * Embedded 4th-order Dormand-Prince solution.
       *
       * The difference between y5 and y4 is used to estimate the local
       * truncation error of the attempted substep.
       */

      for (int i = 0; i < N_SCAL_VAR; ++i)
      {
        y4[i] = y_cur[i] + h *
          ((5179.0f / 57600.0f) * k1[i]
         + (7571.0f / 16695.0f) * k3[i]
         + (393.0f / 640.0f) * k4[i]
         + (-92097.0f / 339200.0f) * k5[i]
         + (187.0f / 2100.0f) * k6[i]
         + (1.0f / 40.0f) * k7[i]);
      }

      /*
       * Compute normalized maximum error over all scalar state variables.
       *
       * err <= 1 means the attempted substep is accepted.
       * err > 1 means the substep is rejected and retried with smaller h.
       */

      float err = 0.0f;

      for (int i = 0; i < N_SCAL_VAR; ++i)
      {
        const float scale =
            abs_tol + rel_tol * max_val(abs_val(y_cur[i]), abs_val(y5[i]));

        const float local_err = abs_val(y5[i] - y4[i]) / scale;

        if (local_err > err)
          err = local_err;
      }

      /*
       * Accept or reject the attempted RK45 substep.
       *
       * If the error is acceptable, y_cur is advanced to y5.
       * If h has reached the minimum allowed value, the step is accepted
       * to avoid an infinite rejection loop.
       */

      if (err <= 1.0f || h <= 1.0e-6f)
      {
        for (int i = 0; i < N_SCAL_VAR; ++i)
          y_cur[i] = y5[i];

        // Keep refractory counter numerically well-behaved
        if (y_cur[i_refr_t] < 0.0f)
          y_cur[i_refr_t] = 0.0f;

        t += h;
      }

      /*
       * Compute next internal step size.
       *
       * Dormand-Prince RK45 uses exponent 1/5 for the adaptive
       * error controller.
       */

      float factor;

      if (err == 0.0f)
      {
        factor = max_factor;
      }
      else
      {
        factor = safety * powf(1.0f / err, 0.2f);
        factor = max_val(min_factor, min_val(max_factor, factor));
      }

      h *= factor;

      // Enforce minimum internal step size
      if (h < 1.0e-6f)
        h = 1.0e-6f;

      ++step_count;
    }

    /*
     * Write final accepted local state back to the global NEST GPU array.
     *
     * If max_steps was reached before t == dt, the last accepted state is
     * written back. No spike/event logic is executed here.
     */

    for (int i = 0; i < N_SCAL_VAR; ++i)
      y[i] = y_cur[i];

    // Final safety clamp for refractory counter
    if (y[i_refr_t] < 0.0f)
      y[i_refr_t] = 0.0f;

    /*
     * Port variables and spike-related variables are intentionally not
     * handled here.
     *
     * They are consumed or modified afterwards by
     * iaf_psc_exp_neuron_nestml_PostUpdate().
     */
  }
};


// Solver class methods


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
  /*
   * Nothing else to initialize.
   *
   * Memory ownership remains in NEST GPU / neuron class.
   * This solver only stores raw pointers to already allocated device arrays.
   */
}


void IafPscExpOdeintSolver::Step(float /*t0*/, float dt)
{
  /*
   * This method is called on the HOST.
   *
   * Thrust launches GPU work internally.
   * The time value t0 is currently unused because the implemented ODE
   * dynamics are autonomous over one simulation step.
   */

  thrust::counting_iterator<int> begin(0);
  thrust::counting_iterator<int> end(n_);

  IafPscExpRK45Functor functor;
  functor.var = var_;
  functor.var_stride = var_stride_;
  functor.par = par_;
  functor.par_stride = par_stride_;
  functor.dt = dt;

  /*
   * Launch one functor invocation per neuron.
   *
   * Each invocation integrates the ODE state variables of one neuron
   * over the interval [t0, t0 + dt] using adaptive internal RK45 substeps.
   */

  thrust::for_each(thrust::device, begin, end, functor);

  /*
   * After this step:
   * ODE states have been advanced numerically.
   * NO onReceive / onCondition logic has been executed yet.
   *The caller must launch iaf_psc_exp_neuron_nestml_PostUpdate()
   * afterwards.
   */
}
