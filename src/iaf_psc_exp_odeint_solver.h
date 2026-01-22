/*
 *  iaf_psc_exp_odeint_solver.h
 *
 *  Experimental numeric solver for iaf_psc_exp neuron model
 *  using Boost.odeint with Thrust backend.
 *
 *  IMPORTANT:
 *  ----------
 *  - This solver is host-driven.
 *  - It does NOT handle events (spikes, threshold, reset).
 *  - It does NOT support neuron-local callbacks.
 *  - It operates on existing NEST GPU state arrays.
 *
 *  This file is a test for architectural analysis.
 */

#ifndef IAF_PSC_EXP_ODEINT_SOLVER_H
#define IAF_PSC_EXP_ODEINT_SOLVER_H

#include <thrust/device_ptr.h>
#include <thrust/for_each.h>
#include <thrust/iterator/counting_iterator.h>
/*
 * NVAR:
 *   Number of state variables integrated numerically.
 *
 * For iaf_psc_exp:
 *   V_m
 *   refr_t
 *   I_syn_exc
 *   I_syn_inh
 *
 * Port variables (exc_spikes, inh_spikes) are NOT integrated.
 */
template<int NVAR>
class IafPscExpOdeintSolver
{
public:
  /*
   * Constructor
   * n_neuron : number of neurons
   * var_arr  : pointer to NEST GPU state array (device memory)
   * stride   : distance between consecutive neurons (n_var_)
   */
  IafPscExpOdeintSolver(int n_neuron, float* var_arr, int stride);

  /*
   * Perform one integration step.
   *
   * t0 : start time of step
   * dt : step size
   *
   * This function is called from the HOST.
   * Internally, Thrust launches GPU kernels.
   */
  void Step(float t0, float dt);

private:
  int n_;          // number of neurons
  float* var_;     // raw pointer to NEST GPU state array
  int stride_;     // neuron stride (n_var_)
};

#endif // IAF_PSC_EXP_ODEINT_SOLVER_H
