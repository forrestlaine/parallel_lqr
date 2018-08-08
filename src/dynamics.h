//
// Created by Forrest Laine on 7/24/18.
//

#ifndef MOTORBOAT_DYNAMICS_H
#define MOTORBOAT_DYNAMICS_H

#include "eigen3/Eigen/Dense"

namespace dynamics {

class Dynamics {

 public:
  Dynamics(std::function<void(const Eigen::VectorXd *, const Eigen::VectorXd *, Eigen::VectorXd &)> *dynamics)
      : dynamics(*dynamics),
        analytic_jacobians_provided(false) {};

  Dynamics(std::function<void(const Eigen::VectorXd *, const Eigen::VectorXd *, Eigen::VectorXd &)> *dynamics,
           std::function<void(const Eigen::VectorXd *,
                              const Eigen::VectorXd *,
                              Eigen::MatrixXd &)> *dynamics_jacobian_state,
           std::function<void(const Eigen::VectorXd *,
                              const Eigen::VectorXd *,
                              Eigen::MatrixXd &)> *dynamics_jacobian_control) :
      dynamics(*dynamics),
      dynamics_jacobian_state(*dynamics_jacobian_state),
      dynamics_jacobian_control(*dynamics_jacobian_control),
      analytic_jacobians_provided(true) {};

  void eval_dynamics(const Eigen::VectorXd *x, const Eigen::VectorXd *u, Eigen::VectorXd &xx);

  void eval_dynamics_jacobian_state(const Eigen::VectorXd *x, const Eigen::VectorXd *u, Eigen::MatrixXd &Ax);

  void eval_dynamics_jacobian_control(const Eigen::VectorXd *x, const Eigen::VectorXd *u, Eigen::MatrixXd &Au);

 private:
  std::function<void(const Eigen::VectorXd *, const Eigen::VectorXd *, Eigen::VectorXd &)> dynamics;
  std::function<void(const Eigen::VectorXd *, const Eigen::VectorXd *, Eigen::MatrixXd &)> dynamics_jacobian_state;
  std::function<void(const Eigen::VectorXd *, const Eigen::VectorXd *, Eigen::MatrixXd &)> dynamics_jacobian_control;

  bool analytic_jacobians_provided;
};

} // namespace dynamics

#endif //MOTORBOAT_DYNAMICS_H
