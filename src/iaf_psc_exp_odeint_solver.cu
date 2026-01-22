/*
 *  iaf_psc_exp_odeint_solver.cu
 *
 *  Experimental numeric solver for iaf_psc_exp neuron model
 *  using Boost.odeint structure with Thrust.
 *
 *  NOTES:
 *  -------
 * 
 *  - This solver mimics odeint's execution model:
 *      host-side step() → thrust::for_each → device kernel
 *  - No neuron-local callbacks are possible.
 *  - No event handling is integrated.
 *
 *  This highlights the architectural mismatch between
 *  odeint and NEST GPU.
 */

#include "iaf_psc_exp_odeint_solver.h"

/*
 * Device functor executed by Thrust.
 *
 * Each invocation updates ONE neuron.
 *
 * LIMITATIONS:
 * -----------
 * - No access to spike buffers
 * - No threshold checks
 * - No reset logic
 * - No access to end_time_step
 *
 * This is the core reason why odeint does not fit well.
 */
template<int NVAR>
struct IafEulerStepFunctor
{
  float* var;
  int stride;
  float dt;

  __host__ __device__
  void operator()(int i_neuron) const
  {
    // Pointer to state variables of one neuron
    float* y = var + i_neuron * stride;

    // State layout:
    // y[0] = V_m
    // y[1] = refr_t
    // y[2] = I_syn_exc
    // y[3] = I_syn_inh

    // ----------------------------
    // it's a VERY SIMPLE NUMERIC MODEL (just an example to show the limits)
    // ----------------------------

    // dI/dt = -I
    y[2] += dt * (-y[2]);
    y[3] += dt * (-y[3]);

    // dV/dt = -V
    y[0] += dt * (-y[0]);

    // drefr_t/dt = -1
    y[1] += dt * (-1.0f);

    // NOTE:
    //------
    // No threshold detection here.
    // No spike generation.
    // No reset.
  }
};

template<int NVAR>
IafPscExpOdeintSolver<NVAR>::IafPscExpOdeintSolver(
    int n_neuron,
    float* var_arr,
    int stride)
: n_(n_neuron)
, var_(var_arr)
, stride_(stride)
{
  // Nothing else to initialize.
  // We explicitly rely on NEST GPU memory ownership.
}

template<int NVAR>
void IafPscExpOdeintSolver<NVAR>::Step(float /*t0*/, float dt)
{
  /*
   * IMPORTANT ARCHITECTURAL POINT:
   * ------------------------------
   * This function is called on the HOST.
   * Thrust internally launches GPU kernels,
   * but the control flow remains host-driven.
   *
   * There is NO way to:
   *  - call ExternalUpdate()
   *  - react to spikes during integration
   *  - adapt step size per neuron
   */

  thrust::counting_iterator<int> begin(0);
  thrust::counting_iterator<int> end(n_);

  IafEulerStepFunctor<NVAR> functor;
  functor.var = var_;
  functor.stride = stride_;
  functor.dt = dt;

  // Launch Thrust kernel
  thrust::for_each(begin, end, functor);

  /*
   * After this point:
   * - State variables have been updated numerically
   * - NO event handling has occurred
   *
   * NEST GPU must now run a separate kernel
   * to process:
   *   - onReceive
   *   - onCondition
   *   - spike emission
   *
   * This split is exactly what breaks the
   * clean neuron-local update semantics.
   */
}

// Explicit template instantiation
template class IafPscExpOdeintSolver<4>;
