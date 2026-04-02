#ifndef IAF_PSC_EXP_ODEINT_SOLVER_H
#define IAF_PSC_EXP_ODEINT_SOLVER_H

#include <thrust/for_each.h>
#include <thrust/iterator/counting_iterator.h>
#include <cuda_runtime.h>
#include <cmath>

class IafPscExpOdeintSolver
{
public:
  IafPscExpOdeintSolver(int n_node,
                        float* var_arr,
                        int n_var,
                        float* param_arr,
                        int n_param)
    : n_node_(n_node)
    , var_arr_(var_arr)
    , n_var_(n_var)
    , param_arr_(param_arr)
    , n_param_(n_param)
  {
  }

  void Step(float t0, float dt)
  {
    thrust::for_each(
      thrust::counting_iterator<int>(0),
      thrust::counting_iterator<int>(n_node_),
      StepFunctor(var_arr_, n_var_, param_arr_, n_param_, t0, dt));
  }

private:
  int n_node_;
  float* var_arr_;
  int n_var_;
  float* param_arr_;
  int n_param_;

  struct StepFunctor
  {
    float* var_arr;
    int n_var;
    float* param_arr;
    int n_param;
    float t0;
    float dt;

    __host__ __device__
    StepFunctor(float* v, int nv, float* p, int np, float t, float h)
      : var_arr(v), n_var(nv), param_arr(p), n_param(np), t0(t), dt(h)
    {
    }

    __device__
    void operator()(int i_neuron) const
    {
      float* y = var_arr + i_neuron * n_var;
      float* p = param_arr + i_neuron * n_param;

      enum ScalVarIndexes {
        i_V_m,
        i_refr_t,
        i_I_syn_exc,
        i_I_syn_inh,
        N_SCAL_VAR
      };

      enum ScalParamIndexes {
        i_C_m,
        i_tau_m,
        i_tau_syn_inh,
        i_tau_syn_exc,
        i_refr_T,
        i_E_L,
        i_V_reset,
        i_V_th,
        i_I_e,
        i___h,
        i_I_stim,
        N_SCAL_PARAM
      };

      const float V_m      = y[i_V_m];
      const float refr_t   = y[i_refr_t];
      const float I_exc    = y[i_I_syn_exc];
      const float I_inh    = y[i_I_syn_inh];

      float dV    = 0.0f;
      float drefr = 0.0f;
      float dExc  = 0.0f;
      float dInh  = 0.0f;

      if (refr_t > 0.0f)
      {
        dV    = 0.0f;
        drefr = -1.0f;
      }
      else
      {
        dV = (-(V_m - p[i_E_L])) / p[i_tau_m]
             + (I_exc - I_inh + p[i_I_e] + p[i_I_stim]) / p[i_C_m];
        drefr = 0.0f;
      }

      dExc = -I_exc / p[i_tau_syn_exc];
      dInh = -I_inh / p[i_tau_syn_inh];

      // einfacher expliziter Euler-Schritt
      y[i_V_m]       += dt * dV;
      y[i_refr_t]    += dt * drefr;
      y[i_I_syn_exc] += dt * dExc;
      y[i_I_syn_inh] += dt * dInh;
    }
  };
};

#endif
