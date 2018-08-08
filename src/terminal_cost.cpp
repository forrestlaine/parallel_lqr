//
// Created by Forrest Laine on 7/24/18.
//

#include "terminal_cost.h"
#include "numerical_gradient.h"

namespace terminal_cost {

double TerminalCost::eval_cost(const Eigen::VectorXd *x) {
  return this->cost(x);
}

void TerminalCost::eval_cost_gradient_state(const Eigen::VectorXd *x,
                                            Eigen::VectorXd &Qx1) {
  if (this->analytic_gradient_given) {
    this->cost_gradient_state(x, Qx1);
  } else {
    numerical_gradient::numerical_gradient(&(this->cost), x, Qx1);
  }
}

void TerminalCost::eval_cost_hessian_state_state(const Eigen::VectorXd *x,
                                                 Eigen::MatrixXd &Qxx) {
  if (this->analytic_hessian_given) {
    this->cost_hessian_state_state(x, Qxx);
  } else if (this->analytic_gradient_given) {
    numerical_gradient::numerical_jacobian(&(this->cost_gradient_state), x, Qxx);
  } else {
    numerical_gradient::numerical_hessian(&(this->cost), x, Qxx);
  }
}

} // namespace terminal_cost