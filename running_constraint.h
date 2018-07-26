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
                                       int,
                                       Eigen::VectorXd &)> *constraint,
                    const int constraint_dimension) :
      constraint(*constraint),
      analytic_jacobians_given(false),
      constraint_dimension(constraint_dimension) {};

  RunningConstraint(std::function<void(const Eigen::VectorXd *,
                                       const Eigen::VectorXd *,
                                       int,
                                       Eigen::VectorXd &)> *constraint,
                    std::function<void(const Eigen::VectorXd *,
                                       const Eigen::VectorXd *,
                                       int,
                                       Eigen::MatrixXd &)> *constraint_jacobian_state,
                    std::function<void(const Eigen::VectorXd *,
                                       const Eigen::VectorXd *,
                                       int,
                                       Eigen::MatrixXd &)> *constraint_jacobian_control,
                    const int constraint_dimension) :
      constraint(*constraint),
      constraint_jacobian_state(*constraint_jacobian_state),
      constraint_jacobian_control(*constraint_jacobian_control),
      analytic_jacobians_given(true),
      constraint_dimension(constraint_dimension) {};

  void eval_constraint(const Eigen::VectorXd *x,
                       const Eigen::VectorXd *u,
                       int t,
                       Eigen::VectorXd &h);

  long eval_active_indices(const Eigen::VectorXd *constraint,
                           Eigen::Matrix<bool, Eigen::Dynamic, 1> &active_indices);

  void eval_constraint_jacobian_state(const Eigen::VectorXd *x,
                                      const Eigen::VectorXd *u,
                                      int t,
                                      Eigen::MatrixXd &Hx);

  void eval_constraint_jacobian_control(const Eigen::VectorXd *x,
                                        const Eigen::VectorXd *u,
                                        int t,
                                        Eigen::MatrixXd &Hu);

  int get_constraint_dimension() { return constraint_dimension; }

 private:
  std::function<void(const Eigen::VectorXd *, const Eigen::VectorXd *, int, Eigen::VectorXd &)> constraint;
  std::function<void(const Eigen::VectorXd *, const Eigen::VectorXd *, int, Eigen::MatrixXd &)> constraint_jacobian_state;
  std::function<void(const Eigen::VectorXd *, const Eigen::VectorXd *, int, Eigen::MatrixXd &)> constraint_jacobian_control;

  bool analytic_jacobians_given;

  const int constraint_dimension;
};

} // namespace running_constraint

#endif //MOTORBOAT_RUNNING_CONSTRAINT_H
