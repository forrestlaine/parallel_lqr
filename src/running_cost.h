//
// Created by Forrest Laine on 7/24/18.
//

#ifndef MOTORBOAT_RUNNING_COST_H
#define MOTORBOAT_RUNNING_COST_H

#include "Eigen3/Eigen/Dense"

namespace running_cost {

class RunningCost {
 public:

  RunningCost(std::function<double(const Eigen::VectorXd *,
                                   const Eigen::VectorXd *)> *cost) :
      cost(*cost),
      analytic_gradients_given(false),
      analytic_hessians_given(false) {};

  RunningCost(std::function<double(const Eigen::VectorXd *,
                                   const Eigen::VectorXd *)> *cost,
              std::function<void(const Eigen::VectorXd *,
                                 const Eigen::VectorXd *,
                                 Eigen::VectorXd &)> *cost_gradient_state,
              std::function<void(const Eigen::VectorXd *,
                                 const Eigen::VectorXd *,
                                 Eigen::VectorXd &)> *cost_gradient_control) :
      cost(*cost),
      cost_gradient_state(*cost_gradient_state),
      cost_gradient_control(*cost_gradient_control),
      analytic_gradients_given(true),
      analytic_hessians_given(false) {};

  RunningCost(std::function<double(const Eigen::VectorXd *,
                                   const Eigen::VectorXd *)> *cost,
              std::function<void(const Eigen::VectorXd *,
                                 const Eigen::VectorXd *,
                                 Eigen::VectorXd &)> *cost_gradient_state,
              std::function<void(const Eigen::VectorXd *,
                                 const Eigen::VectorXd *,
                                 Eigen::VectorXd &)> *cost_gradient_control,
              std::function<void(const Eigen::VectorXd *,
                                 const Eigen::VectorXd *,
                                 Eigen::MatrixXd &)> *cost_hessian_state_state,
              std::function<void(const Eigen::VectorXd *,
                                 const Eigen::VectorXd *,
                                 Eigen::MatrixXd &)> *cost_hessian_control_state,
              std::function<void(const Eigen::VectorXd *,
                                 const Eigen::VectorXd *,
                                 Eigen::MatrixXd &)> *cost_hessian_control_control) :
      cost(*cost),
      cost_gradient_state(*cost_gradient_state),
      cost_gradient_control(*cost_gradient_control),
      cost_hessian_state_state(*cost_hessian_state_state),
      cost_hessian_control_state(*cost_hessian_control_state),
      cost_hessian_control_control(*cost_hessian_control_control),
      analytic_gradients_given(true),
      analytic_hessians_given(true) {};

  double eval_cost(const Eigen::VectorXd *x,
                   const Eigen::VectorXd *u);

  void eval_cost_gradient_state(const Eigen::VectorXd *x,
                                const Eigen::VectorXd *u,
                                Eigen::VectorXd &Qx1);

  void eval_cost_gradient_control(const Eigen::VectorXd *x,
                                  const Eigen::VectorXd *u,
                                  Eigen::VectorXd &Qu1);

  void eval_cost_hessian_state_state(const Eigen::VectorXd *x,
                                     const Eigen::VectorXd *u,
                                     Eigen::MatrixXd &Qxx);

  void eval_cost_hessian_control_state(const Eigen::VectorXd *x,
                                       const Eigen::VectorXd *u,
                                       Eigen::MatrixXd &Qux);

  void eval_cost_hessian_control_control(const Eigen::VectorXd *x,
                                         const Eigen::VectorXd *u,
                                         Eigen::MatrixXd &Quu);

 private:
  std::function<double(const Eigen::VectorXd *, const Eigen::VectorXd *)> cost;
  std::function<void(const Eigen::VectorXd *, const Eigen::VectorXd *, Eigen::VectorXd &)> cost_gradient_state;
  std::function<void(const Eigen::VectorXd *, const Eigen::VectorXd *, Eigen::VectorXd &)> cost_gradient_control;
  std::function<void(const Eigen::VectorXd *, const Eigen::VectorXd *, Eigen::MatrixXd &)> cost_hessian_state_state;
  std::function<void(const Eigen::VectorXd *, const Eigen::VectorXd *, Eigen::MatrixXd &)> cost_hessian_control_state;
  std::function<void(const Eigen::VectorXd *, const Eigen::VectorXd *, Eigen::MatrixXd &)> cost_hessian_control_control;

  bool analytic_gradients_given;
  bool analytic_hessians_given;
};

} // namespace running_cost

#endif //MOTORBOAT_RUNNING_COST_H
