#ifndef AEIF_COND_ALPHA_ALT_ODEINT_SOLVER_H
#define AEIF_COND_ALPHA_ALT_ODEINT_SOLVER_H

#include <thrust/for_each.h>
#include <thrust/iterator/counting_iterator.h>
#include <cuda_runtime.h>
#include <cmath>

#ifndef MIN
#define MIN(a,b) (((a)<(b))?(a):(b))
#endif

class AeifCondAlphaAltOdeintSolver
{
public:
  AeifCondAlphaAltOdeintSolver(int n_node,
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
        i_w,
        i_refr_t,
        i_g_exc,
        i_g_exc__d,
        i_g_inh,
        i_g_inh__d,
        N_SCAL_VAR
      };

      enum ScalParamIndexes {
        i_C_m,
        i_refr_T,
        i_V_reset,
        i_g_L,
        i_E_L,
        i_a,
        i_b,
        i_Delta_T,
        i_tau_w,
        i_V_th,
        i_V_peak,
        i_tau_syn_exc,
        i_tau_syn_inh,
        i_E_exc,
        i_E_inh,
        i_I_e,
        i___h,
        i_I_stim,
        N_SCAL_PARAM
      };

      const float V_m     = y[i_V_m];
      const float w       = y[i_w];
      const float refr_t  = y[i_refr_t];
      const float g_exc   = y[i_g_exc];
      const float g_exc_d = y[i_g_exc__d];
      const float g_inh   = y[i_g_inh];
      const float g_inh_d = y[i_g_inh__d];

      const float V_eff = MIN(V_m, p[i_V_peak]);

      float dV      = 0.0f;
      float dw      = 0.0f;
      float drefr   = 0.0f;
      float dg_exc  = 0.0f;
      float dg_exc_d= 0.0f;
      float dg_inh  = 0.0f;
      float dg_inh_d= 0.0f;

      dg_exc   = g_exc_d;
      dg_exc_d = -g_exc / (p[i_tau_syn_exc] * p[i_tau_syn_exc])
                 - 2.0f * g_exc_d / p[i_tau_syn_exc];

      dg_inh   = g_inh_d;
      dg_inh_d = -g_inh / (p[i_tau_syn_inh] * p[i_tau_syn_inh])
                 - 2.0f * g_inh_d / p[i_tau_syn_inh];

      dw = p[i_a] * ((V_eff - p[i_E_L]) / p[i_tau_w]) - w / p[i_tau_w];

      if (refr_t > 0.0f)
      {
        dV    = 0.0f;
        drefr = -1.0f;
      }
      else
      {
        dV =
          (
            -p[i_g_L] * (V_eff - p[i_E_L])
            + p[i_g_L] * p[i_Delta_T] * expf((V_eff - p[i_V_th]) / p[i_Delta_T])
            - g_exc * (V_eff - p[i_E_exc])
            - g_inh * (V_eff - p[i_E_inh])
            - w
            + p[i_I_e]
            + p[i_I_stim]
          ) / p[i_C_m];

        drefr = 0.0f;
      }

      y[i_V_m]      += dt * dV;
      y[i_w]        += dt * dw;
      y[i_refr_t]   += dt * drefr;
      y[i_g_exc]    += dt * dg_exc;
      y[i_g_exc__d] += dt * dg_exc_d;
      y[i_g_inh]    += dt * dg_inh;
      y[i_g_inh__d] += dt * dg_inh_d;
    }
  };
};

#endif
