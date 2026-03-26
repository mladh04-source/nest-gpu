/*
 *  aeif_cond_alpha_alt_odeint_solver.h
 *
 *  Experimental solver backend for aeif_cond_alpha_alt_neuron_nestml
 *  using Boost.odeint + Thrust.
 *
 *  IMPORTANT:
 *  ----------
 *  - This is an experimental fixed-step solver backend.
 *  - It uses a global time step for all neurons.
 *  - ExternalUpdate() is executed in a separate CUDA kernel after do_step().
 *  - Therefore, this is not behavior-identical to the original adaptive RK5 path.
 *
 *  The purpose of this file is to integrate the existing NESTML-generated
 *  neuron model with Boost.odeint + Thrust while keeping the neuron logic
 *  (NodeInit, Derivatives, ExternalUpdate) as unchanged as possible.
 */

#ifndef AEIF_COND_ALPHA_ALT_ODEINT_SOLVER_H
#define AEIF_COND_ALPHA_ALT_ODEINT_SOLVER_H

#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/for_each.h>

#include <boost/numeric/odeint.hpp>
#include <boost/numeric/odeint/external/thrust/thrust.hpp>

#include "cuda_error.h"

/*
 * Forward declarations of model callbacks from aeif_cond_alpha_alt_neuron_nestml.*
 * These are defined in the neuron .cu and reused here.
 */
struct aeif_cond_alpha_alt_neuron_nestml_rk5;

template<int NVAR, int NPARAM>
__device__
void Derivatives(double x, float *y, float *dydx, float *param,
                 aeif_cond_alpha_alt_neuron_nestml_rk5 data_struct);

template<int NVAR, int NPARAM>
__device__
void ExternalUpdate(double x, float *y, float *param, bool end_time_step,
                    aeif_cond_alpha_alt_neuron_nestml_rk5 data_struct);

__device__
void NodeInit(int n_var, int n_param, double x, float *y,
              float *param, aeif_cond_alpha_alt_neuron_nestml_rk5 data_struct);

__device__
void NodeCalibrate(int n_var, int n_param, double x, float *y,
                   float *param, aeif_cond_alpha_alt_neuron_nestml_rk5 data_struct);

template<int NVAR, int NPARAM>
class AeifCondAlphaAltOdeintSolver
{
public:
  typedef thrust::device_vector<float> state_type;
  typedef boost::numeric::odeint::runge_kutta4<
      state_type,
      float,
      state_type,
      float,
      boost::numeric::odeint::thrust_algebra,
      boost::numeric::odeint::thrust_operations
    > stepper_type;

  AeifCondAlphaAltOdeintSolver(
      int n_node,
      int n_var,
      int n_param,
      double x0,
      float h,
      aeif_cond_alpha_alt_neuron_nestml_rk5 data_struct);

  ~AeifCondAlphaAltOdeintSolver() = default;

  void Calibrate(double time_min, float h);

  void Step(double t1, float h_min,
            aeif_cond_alpha_alt_neuron_nestml_rk5 data_struct);

  float* GetYArr();
  float* GetParamArr();

  int GetX(int i_neuron, int n_node, double *x);
  int GetY(int i_var, int i_neuron, int n_node, float *y);

private:
  /*
   * ODE system wrapper for Boost.odeint.
   * It computes dydt for the FULL interleaved state array using Thrust.
   */
  struct OdeintSystem
  {
    int n_node_;
    int n_var_;
    int n_param_;
    float* param_ptr_;
    aeif_cond_alpha_alt_neuron_nestml_rk5 data_struct_;

    void operator()(const state_type &x, state_type &dxdt, float t) const;
  };

private:
  int n_node_;
  int n_var_;
  int n_param_;

  double x0_;
  float h_;

  aeif_cond_alpha_alt_neuron_nestml_rk5 data_struct_;

  state_type y_vec_;
  thrust::device_vector<float> param_vec_;
  thrust::device_vector<double> x_vec_;

  stepper_type stepper_;
  OdeintSystem system_;
};

#endif
