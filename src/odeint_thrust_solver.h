#ifndef ODEINT_THRUST_SOLVER_H
#define ODEINT_THRUST_SOLVER_H

#include <thrust/device_vector.h>
#include <boost/numeric/odeint.hpp>
#include <boost/numeric/odeint/external/thrust/thrust.hpp>

template<int NVAR>
class OdeintThrustSolver
{
public:
    using state_type = thrust::device_vector<float>;
    using stepper_type =
        boost::numeric::odeint::runge_kutta4<
            state_type, float, state_type, float,
            boost::numeric::odeint::thrust_algebra,
            boost::numeric::odeint::thrust_operations>;

    OdeintThrustSolver(int n_neuron);

    float* GetYRaw();   // so NESTGPU work later with it 
    void Step(float t, float dt);

private:
    int n_neuron_;
    state_type y_;
    stepper_type stepper_;
};

#endif
