#include "odeint_thrust_solver.h"
#include <thrust/system/cuda/execution_policy.h>

using namespace boost::numeric::odeint;
// No real derivatives yet – that comes later
// Dummy RHS: dy/dt = 0
struct DummySystem
{
    void operator()(const thrust::device_vector<float> &x,
                    thrust::device_vector<float> &dxdt,
                    float /*t*/) const
    {
        thrust::fill(dxdt.begin(), dxdt.end(), 0.0f);
    }
};

template<int NVAR>
OdeintThrustSolver<NVAR>::OdeintThrustSolver(int n_neuron)
: n_neuron_(n_neuron),
  y_(n_neuron * NVAR, 0.0f)
{}

template<int NVAR>
float* OdeintThrustSolver<NVAR>::GetYRaw()
{
    return thrust::raw_pointer_cast(y_.data());
}

template<int NVAR>
void OdeintThrustSolver<NVAR>::Step(float t, float dt)
{
    static thrust::device_vector<float> dydt(y_.size());
    DummySystem sys;
    stepper_.do_step(sys, y_, t, dt);
}
