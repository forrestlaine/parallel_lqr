//
// Created by Forrest Laine on 7/24/18.
//

#include "running_cost.h"
#include "numerical_gradient.h"

namespace running_cost {

double RunningCost::eval_cost(const Eigen::VectorXd *x,
                 const Eigen::VectorXd *u) {
  return this->cost(x, u);
}

void RunningCost::eval_cost_gradient_state(const Eigen::VectorXd *x,
                              const Eigen::VectorXd *u,
                              Eigen::VectorXd &Qx1) {
  if (this->analytic_gradients_given) {
    this->cost_gradient_state(x, u, Qx1);
  } else {
    numerical_gradient::numerical_gradient_first_input(&(this->cost), x, u, Qx1);
  }
}

void RunningCost::eval_cost_gradient_control(const Eigen::VectorXd *x,
                                const Eigen::VectorXd *u,
                                Eigen::VectorXd &Qu1) {
  if (this->analytic_gradients_given) {
    this->cost_gradient_control(x, u, Qu1);
  } else {
    numerical_gradient::numerical_gradient_second_input(&(this->cost), x, u, Qu1);
  }
}

void RunningCost::eval_cost_hessian_state_state(const Eigen::VectorXd *x,
                                   const Eigen::VectorXd *u,
                                   Eigen::MatrixXd &Qxx) {
  if (this->analytic_hessians_given) {
    this->cost_hessian_state_state(x, u, Qxx);
  } else if (this->analytic_gradients_given) {
    numerical_gradient::numerical_jacobian_first_input(&(this->cost_gradient_state), x, u, Qxx);
  } else {
    numerical_gradient::numerical_hessian_first_first(&(this->cost), x, u, Qxx);
  }
}

void RunningCost::eval_cost_hessian_control_state(const Eigen::VectorXd *x,
                                     const Eigen::VectorXd *u,
                                     Eigen::MatrixXd &Qux) {
  if (this->analytic_hessians_given) {
    this->cost_hessian_control_state(x, u, Qux);
  } else if (this->analytic_gradients_given) {
    numerical_gradient::numerical_jacobian_first_input(&(this->cost_gradient_control), x, u, Qux);
  } else {
    numerical_gradient::numerical_hessian_second_first(&(this->cost), x, u, Qux);
  }
}

void RunningCost::eval_cost_hessian_control_control(const Eigen::VectorXd *x,
                                       const Eigen::VectorXd *u,
                                       Eigen::MatrixXd &Quu) {
  if (this->analytic_hessians_given) {
    this->cost_hessian_control_control(x, u, Quu);
  } else if (this->analytic_gradients_given) {
    numerical_gradient::numerical_jacobian_second_input(&(this->cost_gradient_control), x, u, Quu);
  } else {
    numerical_gradient::numerical_hessian_second_second(&(this->cost), x, u, Quu);
  }
}

} // namespace running_cost