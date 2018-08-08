//
// Created by Forrest Laine on 7/24/18.
//

#include "dynamics.h"
#include "numerical_gradient.h"

namespace dynamics {

void Dynamics::eval_dynamics(const Eigen::VectorXd *x, const Eigen::VectorXd *u, Eigen::VectorXd &xx) {
  this->dynamics(x, u, xx);
}

void Dynamics::eval_dynamics_jacobian_state(const Eigen::VectorXd *x,
                                            const Eigen::VectorXd *u,
                                            Eigen::MatrixXd &Ax) {
  if (this->analytic_jacobians_provided) {
    this->dynamics_jacobian_state(x, u, Ax);
  } else {
    numerical_gradient::numerical_jacobian_first_input(&(this->dynamics), x, u, Ax);
  }
}

void Dynamics::eval_dynamics_jacobian_control(const Eigen::VectorXd *x,
                                              const Eigen::VectorXd *u,
                                              Eigen::MatrixXd &Au) {
  if (this->analytic_jacobians_provided) {
    this->dynamics_jacobian_control(x, u, Au);
  } else {
    numerical_gradient::numerical_jacobian_second_input(&(this->dynamics), x, u, Au);
  }
}

} // namespace dynamics