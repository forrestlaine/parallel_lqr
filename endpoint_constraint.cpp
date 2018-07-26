//
// Created by Forrest Laine on 7/24/18.
//

#include "endpoint_constraint.h"
#include "numerical_gradient.h"

namespace endpoint_constraint {

void EndPointConstraint::eval_constraint(const Eigen::VectorXd *x,
                                         Eigen::VectorXd &h) {
  this->constraint(x, h);
}

long EndPointConstraint::eval_active_indices(const Eigen::VectorXd *constraint,
                                             Eigen::Matrix<bool, Eigen::Dynamic, 1> &active_indices) {
  const double tol = 1e-8;
  const long n = (*constraint).size();
  for (int i = 0; i < n; ++i) {
    active_indices(i) = ((*constraint)(i) > -tol);
  }
  return active_indices.count();
}

void EndPointConstraint::eval_constraint_jacobian(const Eigen::VectorXd *x,
                                                  Eigen::MatrixXd &Hx) {
  if (this->analytic_jacobian_given) {
    this->constraint_jacobian(x, Hx);
  } else {
    numerical_gradient::numerical_jacobian(&(this->constraint), x, Hx);
  }
}

} // namespace endpoint_constraint