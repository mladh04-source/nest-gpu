/*
 *  aeif_cond_alpha_alt_odeint_solver.cu
 *
 *  Experimental solver backend for aeif_cond_alpha_alt_neuron_nestml
 *  using Boost.odeint + Thrust.
 *
 *  NOTES:
 *  ------
 *  - State is stored as one interleaved device_vector<float>:
 *      [ neuron0_var0, neuron0_var1, ..., neuron0_var(n_var-1),
 *        neuron1_var0, neuron1_var1, ..., neuron1_var(n_var-1), ... ]
 *
 *  - Parameters are stored analogously in param_vec_.
 *
 *  - We reuse the NESTML-generated functions:
 *      NodeInit
 *      NodeCalibrate
 *      Derivatives
 *      ExternalUpdate
 *
 *  - odeint performs one global RK4 step over the whole interleaved state.
 *  - Event handling / spike handling is done afterwards by calling
 *    ExternalUpdate() in a dedicated CUDA kernel.
 */

#include "aeif_cond_alpha_alt_odeint_solver.h"

#include <cmath>
#include <iostream>

#include "cuda_error.h"


// Initialization kernel
__global__
void AeifCondAlphaAltInitKernel(
    int n_node,
    int n_var,
    int n_param,
    double x0,
    float *y_arr,
    float *param_arr,
    double *x_arr,
    aeif_cond_alpha_alt_neuron_nestml_rk5 data_struct)
{
  int i_neuron = threadIdx.x + blockIdx.x * blockDim.x;
  if (i_neuron >= n_node) {
    return;
  }

  float *y = y_arr + i_neuron * n_var;
  float *param = param_arr + i_neuron * n_param;

  NodeInit(n_var, n_param, x0, y, param, data_struct);
  x_arr[i_neuron] = x0;
}


// Calibration kernel
__global__
void AeifCondAlphaAltCalibrateKernel(
    int n_node,
    int n_var,
    int n_param,
    double x0,
    float *y_arr,
    float *param_arr,
    aeif_cond_alpha_alt_neuron_nestml_rk5 data_struct)
{
  int i_neuron = threadIdx.x + blockIdx.x * blockDim.x;
  if (i_neuron >= n_node) {
    return;
  }

  float *y = y_arr + i_neuron * n_var;
  float *param = param_arr + i_neuron * n_param;

  NodeCalibrate(n_var, n_param, x0, y, param, data_struct);
}


//Update x-array kernel
__global__
void AeifCondAlphaAltSetXKernel(
    int n_node,
    double *x_arr,
    double xval)
{
  int i_neuron = threadIdx.x + blockIdx.x * blockDim.x;
  if (i_neuron >= n_node) {
    return;
  }

  x_arr[i_neuron] = xval;
}

//  Post-update kernel
template<int NVAR, int NPARAM>
__global__
void AeifCondAlphaAltExternalUpdateKernel(
    int n_node,
    int n_var,
    int n_param,
    double t,
    float *y_arr,
    float *param_arr,
    aeif_cond_alpha_alt_neuron_nestml_rk5 data_struct)
{
  int i_neuron = threadIdx.x + blockIdx.x * blockDim.x;
  if (i_neuron >= n_node) {
    return;
  }

  float *y = y_arr + i_neuron * n_var;
  float *param = param_arr + i_neuron * n_param;

  ExternalUpdate<NVAR, NPARAM>(t, y, param, true, data_struct);
}


// Thrust functor to compute derivatives per neuron
template<int NVAR, int NPARAM>
struct AeifCondAlphaAltDerivFunctor
{
  const float *y_arr_;
  float *dydt_arr_;
  const float *param_arr_;
  int n_var_;
  int n_param_;
  float t_;
  aeif_cond_alpha_alt_neuron_nestml_rk5 data_struct_;

  __host__ __device__
  void operator()(int i_neuron) const
  {
    const float *y_const = y_arr_ + i_neuron * n_var_;
    float *dydt = dydt_arr_ + i_neuron * n_var_;
    const float *param_const = param_arr_ + i_neuron * n_param_;

    /*
     * Derivatives() expects non-const float* arguments.
     * We do not modify y/param inside Derivatives(), so this cast is only
     * to satisfy the generated function signature.
     */
    float *y = const_cast<float*>(y_const);
    float *param = const_cast<float*>(param_const);

    Derivatives<NVAR, NPARAM>(t_, y, dydt, param, data_struct_);
  }
};


//  OdeintSystem::operator()

template<int NVAR, int NPARAM>
void AeifCondAlphaAltOdeintSolver<NVAR, NPARAM>::OdeintSystem::operator()(
    const state_type &x,
    state_type &dxdt,
    float t) const
{
  const float *x_ptr =
    thrust::raw_pointer_cast(const_cast<state_type&>(x).data());
  float *dxdt_ptr =
    thrust::raw_pointer_cast(dxdt.data());

  AeifCondAlphaAltDerivFunctor<NVAR, NPARAM> functor;
  functor.y_arr_ = x_ptr;
  functor.dydt_arr_ = dxdt_ptr;
  functor.param_arr_ = param_ptr_;
  functor.n_var_ = n_var_;
  functor.n_param_ = n_param_;
  functor.t_ = t;
  functor.data_struct_ = data_struct_;

  thrust::counting_iterator<int> begin(0);
  thrust::counting_iterator<int> end(n_node_);

  thrust::for_each(begin, end, functor);
}

// Constructor
template<int NVAR, int NPARAM>
AeifCondAlphaAltOdeintSolver<NVAR, NPARAM>::AeifCondAlphaAltOdeintSolver(
    int n_node,
    int n_var,
    int n_param,
    double x0,
    float h,
    aeif_cond_alpha_alt_neuron_nestml_rk5 data_struct)
  : n_node_(n_node)
  , n_var_(n_var)
  , n_param_(n_param)
  , x0_(x0)
  , h_(h)
  , data_struct_(data_struct)
  , y_vec_(n_node * n_var, 0.0f)
  , param_vec_(n_node * n_param, 0.0f)
  , x_vec_(n_node, x0)
{
  float *y_ptr = thrust::raw_pointer_cast(y_vec_.data());
  float *param_ptr = thrust::raw_pointer_cast(param_vec_.data());
  double *x_ptr = thrust::raw_pointer_cast(x_vec_.data());

  system_.n_node_ = n_node_;
  system_.n_var_ = n_var_;
  system_.n_param_ = n_param_;
  system_.param_ptr_ = param_ptr;
  system_.data_struct_ = data_struct_;

  const int block_size = 256;
  const int grid_size = (n_node_ + block_size - 1) / block_size;

  AeifCondAlphaAltInitKernel<<<grid_size, block_size>>>(
      n_node_, n_var_, n_param_, x0_,
      y_ptr, param_ptr, x_ptr, data_struct_);
  gpuErrchk(cudaPeekAtLastError());
  gpuErrchk(cudaDeviceSynchronize());
}

//  Calibrate
template<int NVAR, int NPARAM>
void AeifCondAlphaAltOdeintSolver<NVAR, NPARAM>::Calibrate(double time_min, float h)
{
  h_ = h;

  float *y_ptr = thrust::raw_pointer_cast(y_vec_.data());
  float *param_ptr = thrust::raw_pointer_cast(param_vec_.data());

  const int block_size = 256;
  const int grid_size = (n_node_ + block_size - 1) / block_size;

  AeifCondAlphaAltCalibrateKernel<<<grid_size, block_size>>>(
      n_node_, n_var_, n_param_, time_min,
      y_ptr, param_ptr, data_struct_);
  gpuErrchk(cudaPeekAtLastError());
  gpuErrchk(cudaDeviceSynchronize());
}

// Step


template<int NVAR, int NPARAM>
void AeifCondAlphaAltOdeintSolver<NVAR, NPARAM>::Step(
    double t1,
    float /*h_min*/,
    aeif_cond_alpha_alt_neuron_nestml_rk5 data_struct)
{
  /* very important:
   * We perform one global fixed RK4 step over all neurons.
   * This uses Boost.odeint + Thrust.
   */
  const float t0 = static_cast<float>(t1 - h_);

  stepper_.do_step(system_, y_vec_, t0, h_);

  /*
   * ExternalUpdate is run afterwards in a dedicated CUDA kernel.
   * This is exactly the architectural difference to the original RK5 path.
   */
  float *y_ptr = thrust::raw_pointer_cast(y_vec_.data());
  float *param_ptr = thrust::raw_pointer_cast(param_vec_.data());
  double *x_ptr = thrust::raw_pointer_cast(x_vec_.data());

  const int block_size = 256;
  const int grid_size = (n_node_ + block_size - 1) / block_size;

  AeifCondAlphaAltExternalUpdateKernel<NVAR, NPARAM><<<grid_size, block_size>>>(
      n_node_, n_var_, n_param_, t1, y_ptr, param_ptr, data_struct);
  gpuErrchk(cudaPeekAtLastError());
  gpuErrchk(cudaDeviceSynchronize());

  AeifCondAlphaAltSetXKernel<<<grid_size, block_size>>>(
      n_node_, x_ptr, t1);
  gpuErrchk(cudaPeekAtLastError());
  gpuErrchk(cudaDeviceSynchronize());
}

// Get raw arrays
template<int NVAR, int NPARAM>
float* AeifCondAlphaAltOdeintSolver<NVAR, NPARAM>::GetYArr()
{
  return thrust::raw_pointer_cast(y_vec_.data());
}

template<int NVAR, int NPARAM>
float* AeifCondAlphaAltOdeintSolver<NVAR, NPARAM>::GetParamArr()
{
  return thrust::raw_pointer_cast(param_vec_.data());
}

// GetX / GetY

template<int NVAR, int NPARAM>
int AeifCondAlphaAltOdeintSolver<NVAR, NPARAM>::GetX(
    int i_neuron, int n_node, double *x)
{
  if (i_neuron < 0 || i_neuron >= n_node_) {
    return 1;
  }

  double *x_ptr = thrust::raw_pointer_cast(x_vec_.data());
  gpuErrchk(cudaMemcpy(x, x_ptr + i_neuron, sizeof(double),
                       cudaMemcpyDeviceToHost));
  return 0;
}

template<int NVAR, int NPARAM>
int AeifCondAlphaAltOdeintSolver<NVAR, NPARAM>::GetY(
    int i_var, int i_neuron, int n_node, float *y)
{
  if (i_var < 0 || i_var >= n_var_) {
    return 1;
  }
  if (i_neuron < 0 || i_neuron >= n_node_) {
    return 1;
  }

  float *y_ptr = thrust::raw_pointer_cast(y_vec_.data());
  gpuErrchk(cudaMemcpy(y, y_ptr + i_neuron * n_var_ + i_var, sizeof(float),
                       cudaMemcpyDeviceToHost));
  return 0;
}

// Explicit template instantiation for this neuron model


template class AeifCondAlphaAltOdeintSolver<9, 18>;
