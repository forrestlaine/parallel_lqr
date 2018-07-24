//
// Created by Forrest Laine on 7/24/18.
//

#ifndef MOTORBOAT_RUNNING_CONSTRAINT_H
#define MOTORBOAT_RUNNING_CONSTRAINT_H

#include "Eigen3/Eigen/Dense"

namespace running_constraint {

class RunningConstraint {

 public:

  RunningConstraint(std::function<void(const Eigen::VectorXd *,
                                       const Eigen::VectorXd *,
                                       Eigen::VectorXd &)> *constraint):
      constraint(*constraint),
      analytic_jacobians_given(false) {};

  RunningConstraint(std::function<void(const Eigen::VectorXd *,
                                       const Eigen::VectorXd *,
                                       Eigen::VectorXd &)> *constraint,
                    std::function<void(const Eigen::VectorXd *,
                                       const Eigen::VectorXd *,
                                       Eigen::MatrixXd &)> *constraint_jacobian_state,
                    std::function<void(const Eigen::VectorXd *,
                                       const Eigen::VectorXd *,
                                       Eigen::MatrixXd &)> *constraint_jacobian_control) :
      constraint(*constraint),
      constraint_jacobian_state(*constraint_jacobian_state),
      constraint_jacobian_control(*constraint_jacobian_control),
      analytic_jacobians_given(true) {};

  void eval_constraint(const Eigen::VectorXd *x,
                       const Eigen::VectorXd *u,
                       Eigen::VectorXd &h);

  long eval_active_indices(const Eigen::VectorXd *constraint,
                           Eigen::Matrix<bool, Eigen::Dynamic, 1> &active_indices);

  void eval_constraint_jacobian_state(const Eigen::VectorXd *x,
                                      const Eigen::VectorXd *u,
                                      Eigen::MatrixXd &Hx);

  void eval_constraint_jacobian_control(const Eigen::VectorXd *x,
                                        const Eigen::VectorXd *u,
                                        Eigen::MatrixXd &Hu);

 private:
  std::function<void(const Eigen::VectorXd *, const Eigen::VectorXd *, Eigen::VectorXd &)> constraint;
  std::function<void(const Eigen::VectorXd *, const Eigen::VectorXd *, Eigen::MatrixXd &)> constraint_jacobian_state;
  std::function<void(const Eigen::VectorXd *, const Eigen::VectorXd *, Eigen::MatrixXd &)> constraint_jacobian_control;

  bool analytic_jacobians_given;
};

} // namespace running_constraint

#endif //MOTORBOAT_RUNNING_CONSTRAINT_H
