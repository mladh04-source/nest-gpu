/*
 *  aeif_cond_alpha_alt_odeint_solver.h
 *
 *  Experimental numeric solver for aeif_cond_alpha_alt_neuron_nestml
 *  using a host-driven odeint/thrust-style execution model.
 *
 *  IMPORTANT:
 *  This solver is host-driven.
 *  It operates directly on the existing NEST GPU raw arrays.
 *  It integrates ONLY the ODE state variables.
 *  It does NOT handle onReceive / onCondition internally.
 *  These are handled afterwards by a separate PostUpdate kernel.
 *
 *  This file is intended for architectural experimentation and comparison.
 */

#ifndef AEIF_COND_ALPHA_ALT_ODEINT_SOLVER_H
#define AEIF_COND_ALPHA_ALT_ODEINT_SOLVER_H

#include <thrust/for_each.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/execution_policy.h>

class AeifCondAlphaAltOdeintSolver
{
public:
  /*
   * Constructor
   *
   * n_neuron    : number of neurons
   * var_arr     : pointer to NEST GPU state array (device memory)
   * var_stride  : distance between consecutive neurons in var_arr (n_var_)
   * param_arr   : pointer to NEST GPU parameter array (device memory)
   * param_stride: distance between consecutive neurons in param_arr (n_param_)
   */
  AeifCondAlphaAltOdeintSolver(
      int n_neuron,
      float* var_arr,
      int var_stride,
      float* param_arr,
      int param_stride);

  /*
   * Perform one integration step over [t0, t0 + dt].
   * This function is called on the HOST.
   * Internally, Thrust launches GPU kernels.
   */
  void Step(float t0, float dt);

private:
  int n_;             // number of neurons
  float* var_;        // raw pointer to NEST GPU state array
  int var_stride_;    // stride in var_arr
  float* param_;      // raw pointer to NEST GPU parameter array
  int param_stride_;  // stride in param_arr
};

#endif
