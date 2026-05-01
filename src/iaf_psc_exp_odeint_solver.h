/*
 *  iaf_psc_exp_odeint_solver.h
 *
 *  Experimental numeric solver for iaf_psc_exp neuron model
 *  using a Thrust "odeint-style" execution:
 *
 *     host Step() -> thrust::for_each -> device functor
 *
 *
 * DESIGN CHOICE (NEST/NESTML pattern):
 * ODE integration is separated from event handling.
 * After the numeric step, NEST GPU runs a PostUpdate kernel for:
 *     onReceive  : apply spike inputs and  onCondition: threshold check, reset, PushSpike
 *
 * IMPORTANT:
 * This solver integrates ONLY the continuous ODE state variables.
 * It does NOT handle spike input, threshold detection or reset internally.
 *
 * Current implementation:
 * Adaptive Dormand-Prince RK45 integration in a Thrust device functor.
 *
 * This file intentionally keeps the solver wrapper minimal.
 */

#ifndef IAF_PSC_EXP_ODEINT_SOLVER_H
#define IAF_PSC_EXP_ODEINT_SOLVER_H

#include <thrust/for_each.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/execution_policy.h>

/*
 * Numeric solver class.
 *
 * This class is a small host-side wrapper around a Thrust-based GPU
 * integration kernel.
 *
 * It stores raw pointers to the existing NEST GPU state and parameter arrays.
 * It does not allocate, free, or own this memory.
 */

class IafPscExpOdeintSolver
{
public:
  /*
   * Constructor
   *
   * n_neuron   : number of neurons
   * var_arr    : device pointer to NEST GPU var array
   * var_stride : distance between consecutive neurons in var_arr, usually n_var_
   * param_arr  : device pointer to NEST GPU parameter array
   * par_stride : distance between consecutive neurons in param_arr, usually n_param_
   *
   * Memory ownership remains outside this solver.
   */

  IafPscExpOdeintSolver(
      int n_neuron,
      float* var_arr,
      int var_stride,
      float* param_arr,
      int par_stride);

  /*
   * Integrate one global simulation time step [t0, t0 + dt].
   *
   * This function is called on the HOST.
   * Internally, it launches one Thrust functor invocation per neuron.
   *
   * Only ODE state variables are integrated here.
   * Event handling must be performed afterwards by the corresponding
   * PostUpdate kernel.
   */

  void Step(float t0, float dt);

private:
  int n_;             // number of neurons
  float* var_;        // raw pointer to NEST GPU state array
  int var_stride_;    // stride in var_ between consecutive neurons
  float* par_;        // raw pointer to NEST GPU parameter array
  int par_stride_;    // stride in par_ between consecutive neurons
};

#endif // IAF_PSC_EXP_ODEINT_SOLVER_H
