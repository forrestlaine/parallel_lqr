//
// Created by Forrest Laine on 7/24/18.
//

#ifndef MOTORBOAT_TERMINAL_COST_H
#define MOTORBOAT_TERMINAL_COST_H

#include "Eigen3/Eigen/Dense"

namespace terminal_cost {

class TerminalCost {
 public:

  TerminalCost(std::function<double(const Eigen::VectorXd *)> *cost) :
      cost(*cost),
      analytic_gradient_given(false),
      analytic_hessian_given(false) {};

  TerminalCost(std::function<double(const Eigen::VectorXd *)> *cost,
               std::function<void(const Eigen::VectorXd *,
                                  Eigen::VectorXd &)> *cost_gradient_state,
               std::function<void(const Eigen::VectorXd *,
                                  Eigen::VectorXd &)> *cost_gradient_control) :
      cost(*cost),
      cost_gradient_state(*cost_gradient_state),
      analytic_gradient_given(true),
      analytic_hessian_given(false) {};

  TerminalCost(std::function<double(const Eigen::VectorXd *)> *cost,
               std::function<void(const Eigen::VectorXd *,
                                  Eigen::VectorXd &)> *cost_gradient_state,
               std::function<void(const Eigen::VectorXd *,
                                  Eigen::MatrixXd &)> *cost_hessian_state_state :
      cost(*cost),
      cost_gradient_state(*cost_gradient_state),
      cost_hessian_state_state(*cost_hessian_state_state),
      analytic_gradient_given(true),
      analytic_hessian_given(true) {};

  double eval_cost(const Eigen::VectorXd *x);

  void eval_cost_gradient_state(const Eigen::VectorXd *x,
                                Eigen::VectorXd &Qx1);

  void eval_cost_hessian_state_state(const Eigen::VectorXd *x,
                                     Eigen::MatrixXd &Qxx);

 private:
  std::function<double(const Eigen::VectorXd *)> cost;
  std::function<void(const Eigen::VectorXd *, Eigen::VectorXd &)> cost_gradient_state;
  std::function<void(const Eigen::VectorXd *, Eigen::MatrixXd &)> cost_hessian_state_state;

  bool analytic_gradient_given;
  bool analytic_hessian_given;
};

} // namespace terminal_cost

#endif //MOTORBOAT_TERMINAL_COST_H
