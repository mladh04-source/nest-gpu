/*
 *  iaf_psc_exp_odeint_solver.h
 *
 *  Experimental numeric solver for iaf_psc_exp neuron model
 *  using a Thrust "odeint-style" execution:
 *
 *     host Step() -> thrust::for_each -> device functor
 *
 * PURPOSE (Bachelor thesis comparison):
 * ------------------------------------
 * - Provide a numeric integration path for iaf_psc_exp that can be compared to
 *   the original analytic solver.
 *
 * DESIGN CHOICE (NEST/NESTML pattern):
 * -----------------------------------
 * - ODE integration is separated from event handling.
 * - After the numeric step, NEST GPU runs a PostUpdate kernel for:
 *     onReceive (apply spike inputs) and 
 *     onCondition (threshold, reset, PushSpike)
 *
 * NOTE:
 * -----
 * This file intentionally keeps the solver minimal with:
 *  Fixed-step integration (Euler).
 *  No adaptive step size.
 *  No spike handling inside integrator.
 */

#ifndef IAF_PSC_EXP_ODEINT_SOLVER_H
#define IAF_PSC_EXP_ODEINT_SOLVER_H

#include <thrust/for_each.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/execution_policy.h>

// Numeric solver class (non-templated for simplicity).
class IafPscExpOdeintSolver
{
public:
  // n_neuron   : number of neurons
  // var_arr    : device pointer to NEST GPU var array
  // var_stride : n_var_ (distance between neurons)
  // param_arr  : device pointer to NEST GPU param array
  // par_stride : n_param_ (distance between neurons)
  IafPscExpOdeintSolver(
      int n_neuron,
      float* var_arr,
      int var_stride,
      float* param_arr,
      int par_stride);

  // Integrate one global time step [t0, t0+dt]
  void Step(float t0, float dt);

private:
  int n_;
  float* var_;
  int var_stride_;
  float* par_;
  int par_stride_;
};

#endif // IAF_PSC_EXP_ODEINT_SOLVER_H
