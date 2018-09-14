//
//
// Created by Forrest Laine on 7/24/18.
//

#ifndef MOTORBOAT_EQUALITY_CONSTRAINED_RUNNING_CONSTRAINT_H
#define MOTORBOAT_EQUALITY_CONSTRAINED_RUNNING_CONSTRAINT_H

#include "running_constraint.h"

namespace equality_constrained_running_constraint {

class EqualityConstrainedRunningConstraint : public running_constraint::RunningConstraint {

 public:
  EqualityConstrainedRunningConstraint(std::function<void(const Eigen::VectorXd *,
                                       const Eigen::VectorXd *,
                                       int,
                                       Eigen::VectorXd &)> *constraint,
                    const int constraint_dimension) : running_constraint::RunningConstraint(constraint, constraint_dimension) {};

  EqualityConstrainedRunningConstraint(std::function<void(const Eigen::VectorXd *,
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
                    const int constraint_dimension) : running_constraint::RunningConstraint(constraint, constraint_jacobian_state, constraint_jacobian_control, constraint_dimension){};

  long eval_active_indices(const Eigen::VectorXd *constraint,
                           Eigen::Matrix<bool, Eigen::Dynamic, 1> &active_indices) {
    const long n = (*constraint).size();
    for (int i = 0; i < n; ++i) {
      active_indices(i) = true;
    }
    return active_indices.count();
  }
};

} // namespace equality_constrained_running_constraint

#endif //MOTORBOAT_EQUALITY_CONSTRAINED_RUNNING_CONSTRAINT_H
