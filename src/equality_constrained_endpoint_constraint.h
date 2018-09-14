//
// Created by Forrest Laine on 8/30/18.
//

#ifndef MOTORBOAT_EQUALITY_CONSTRAINTED_ENDPOINT_CONSTRAINT_H
#define MOTORBOAT_EQUALITY_CONSTRAINTED_ENDPOINT_CONSTRAINT_H

#include "endpoint_constraint.h"

namespace equality_constrained_endpoint_constraint {

class EqualityConstrainedEndPointConstraint : public endpoint_constraint::EndPointConstraint {
 public:
  EqualityConstrainedEndPointConstraint(std::function<void(const Eigen::VectorXd *,
                                                           Eigen::VectorXd &)> *constraint,
                                        const int constraint_dimension,
                                        const bool implicit) : endpoint_constraint::EndPointConstraint(constraint,
                                                                                                       constraint_dimension,
                                                                                                       implicit) {};

  EqualityConstrainedEndPointConstraint(std::function<void(const Eigen::VectorXd *,
                                                           Eigen::VectorXd &)> *constraint,
                                        std::function<void(const Eigen::VectorXd *,
                                                           Eigen::MatrixXd &)> *constraint_jacobian,
                                        const int constraint_dimension,
                                        const bool implicit) : endpoint_constraint::EndPointConstraint(constraint,
                                                                                                       constraint_jacobian,
                                                                                                       constraint_dimension,
                                                                                                       implicit) {};

  long eval_active_indices(const Eigen::VectorXd *constraint,
                           Eigen::Matrix<bool, Eigen::Dynamic, 1> &active_indices) {
    const long n = (*constraint).size();
    for (int i = 0; i < n; ++i) {
      active_indices(i) = true;
    }
    return active_indices.count();
  }
};

} // namespace equality_constrained_endpoint_constraint

#endif //MOTORBOAT_EQUALITY_CONSTRAINTED_ENDPOINT_CONSTRAINT_H
