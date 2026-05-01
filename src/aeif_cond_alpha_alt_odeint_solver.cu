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
 *  adaptive Dormand-Prince RK45 stepper
 *  one Thrust functor call per neuron
 *  operates directly on NEST GPU raw arrays
 *
 *  The solver performs internal substeps until the full external
 *  simulation time step dt has been covered.
 */

#include <cmath>
#include <thrust/for_each.h>
#include <thrust/iterator/counting_iterator.h>

#include "aeif_cond_alpha_alt_odeint_solver.h"
#include "aeif_cond_alpha_alt_neuron_nestml.h"

using namespace aeif_cond_alpha_alt_neuron_nestml_ns;


/*
 * Device functor executed by Thrust.
 *
 * Each invocation updates exactly one neuron.
 * The functor performs an adaptive RK45 integration step for the ODE system.
 *
 * Only scalar ODE state variables are integrated.
 * Port variables are intentionally left untouched and must be handled
 * later by the PostUpdate kernel.
 */

struct AeifCondAlphaAltRK45Functor
{
  float* var;
  int var_stride;
  float* param;
  int param_stride;
  float dt;

  /*
   * Small helper functions.
   *
   * These are marked __host__ __device__ so that they can be used
   * inside the Thrust functor on the GPU.
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
   * y    : current temporary state vector
   * p    : parameter vector of the neuron
   * dydt : output derivative vector
   *
   * This function mirrors the continuous dynamics of the model.
   * It does not modify the actual NEST GPU state array directly.
   */

  __host__ __device__
  void compute_derivatives(
      const float* y,
      const float* p,
      float* dydt) const
  {
    // Current state
    const float V_m     = y[i_V_m];
    const float w       = y[i_w];
    const float refr_t  = y[i_refr_t];
    const float g_exc   = y[i_g_exc];
    const float g_exc_d = y[i_g_exc__d];
    const float g_inh   = y[i_g_inh];
    const float g_inh_d = y[i_g_inh__d];

    // Clamp membrane potential in the same way as the original model
    const float Vb = V_m < p[i_V_peak] ? V_m : p[i_V_peak];

    const float tau_exc = p[i_tau_syn_exc];
    const float tau_inh = p[i_tau_syn_inh];

    // Synaptic alpha-system derivatives
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

    // Adaptation variable evolves in both normal and refractory branches
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

  /*
   * Main per-neuron update routine.
   *
   * This method is executed once per neuron by Thrust.
   * It copies the current neuron state into local arrays, performs adaptive
   * RK45 substepping over the interval [0, dt], and writes the final accepted
   * state back to the NEST GPU state array.
   */

  __host__ __device__
  void operator()(int i_neuron) const
  {
    float* y = var + i_neuron * var_stride;
    float* p = param + i_neuron * param_stride;

    /*
     * Local temporary storage.
     *
     * y_cur holds the currently accepted state.
     * y_tmp is used for intermediate RK stages.
     *
     * k1 to k7 are the Dormand-Prince stage derivatives.
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
     * safety reduces aggressive step growth.
     * min_factor and max_factor bound how much h may shrink or grow
     * after one attempted RK45 step.
     */

    const float safety = 0.9f;
    const float min_factor = 0.2f;
    const float max_factor = 5.0f;

    float t = 0.0f;
    float h = dt;

    /*
     * Maximum number of internal substeps.
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
     * If the estimated error is acceptable, the substep is accepted.
     * Otherwise, only h is reduced and the state is not advanced.
     */

    while (t < dt && step_count < max_steps)
    {
      // Do not step beyond the external simulation step dt
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
       * This is the higher-order solution and is used as the accepted state
       * when the local error estimate is small enough.
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
       * The difference between y5 and y4 is used as an estimate of the
       * local truncation error for adaptive step-size control.
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
       * err <= 1 means the substep is accepted.
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
       * Accept or reject the attempted substep.
       *
       * If the error is acceptable, y_cur is advanced to y5.
       * If h has already reached the minimum allowed step size, the step
       * is accepted to avoid an infinite rejection loop.
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
       * Dormand-Prince RK45 uses an exponent of 1/5 for the 5th-order
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
     * Write final accepted local state back to the global NEST GPU state array.
     *
     * If max_steps was reached before t == dt, the last accepted state is
     * written back. No event logic is executed here.
     */

    for (int i = 0; i < N_SCAL_VAR; ++i)
      y[i] = y_cur[i];

    // Final safety clamp for the refractory counter
    if (y[i_refr_t] < 0.0f)
      y[i_refr_t] = 0.0f;

    /*
     * Port variables are intentionally NOT updated here:
     *   y[N_SCAL_VAR + i_exc_spikes]
     *   y[N_SCAL_VAR + i_inh_spikes]
     *
     * They are consumed afterwards by aeif_cond_alpha_alt_neuron_nestml_PostUpdate.
     */
  }
};


// Class implementation

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
  /*
   * Memory ownership remains in NEST GPU / neuron class.
   *
   * This solver only stores raw pointers to already allocated device arrays.
   * It does not allocate, free, or resize neuron state or parameter memory.
   */
}


void AeifCondAlphaAltOdeintSolver::Step(float /*t0*/, float dt)
{
  /*
   * This method is called on the HOST.
   * Thrust launches device-side work internally.
   *
   * The time value t0 is currently unused because the neuron dynamics
   * implemented here are autonomous over one simulation step.
   */

  thrust::counting_iterator<int> begin(0);
  thrust::counting_iterator<int> end(n_);

  AeifCondAlphaAltRK45Functor functor;
  functor.var = var_;
  functor.var_stride = var_stride_;
  functor.param = param_;
  functor.param_stride = param_stride_;
  functor.dt = dt;

  /*
   * Launch one functor invocation per neuron.
   *
   * Each invocation integrates the ODE state variables of one neuron
   * over the interval [t0, t0 + dt] using adaptive internal RK45 substeps.
   */

  thrust::for_each(begin, end, functor);

  /*
   * After this step:
   * ODE states have been advanced numerically.
   * NO onReceive / onCondition logic has been executed yet.
   *
   * The caller must launch the separate PostUpdate kernel afterwards.
   */
}
