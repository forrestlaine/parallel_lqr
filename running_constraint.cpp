//
// Created by Forrest Laine on 7/24/18.
//

#include "running_constraint.h"
#include "numerical_gradient.h"

namespace running_constraint {

void RunningConstraint::eval_constraint(const Eigen::VectorXd *x,
                                        const Eigen::VectorXd *u,
                                        Eigen::VectorXd &h) {
  this->constraint(x, u, h);
}


long RunningConstraint::eval_active_indices(const Eigen::VectorXd *constraint,
                                            Eigen::Matrix<bool, Eigen::Dynamic, 1> &active_indices) {
  const double tol = 1e-8;
  const long n = (*constraint).size();
  for (int i = 0; i < n; ++i) {
    active_indices(i) = ((*constraint)(i) > -tol);
  }
  return active_indices.count();
}

void RunningConstraint::eval_constraint_jacobian_state(const Eigen::VectorXd *x,
                                                       const Eigen::VectorXd *u,
                                                       Eigen::MatrixXd &Hx) {
  if (this->analytic_jacobians_given) {
    this->constraint_jacobian_state(x, u, Hx);
  } else {
    numerical_gradient::numerical_jacobian_first_input(&(this->constraint), x, u, Hx);
  }
}

void RunningConstraint::eval_constraint_jacobian_control(const Eigen::VectorXd *x,
                                                         const Eigen::VectorXd *u,
                                                         Eigen::MatrixXd &Hu) {
  if (this->analytic_jacobians_given) {
    this->constraint_jacobian_control(x, u, Hu);
  } else {
    numerical_gradient::numerical_jacobian_second_input(&(this->constraint), x, u, Hu);
  }
}

} // namespace running_constraint