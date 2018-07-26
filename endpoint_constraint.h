//
// Created by Forrest Laine on 7/24/18.
//

#ifndef MOTORBOAT_ENDPOINT_CONSTRAINT_H
#define MOTORBOAT_ENDPOINT_CONSTRAINT_H

#include "Eigen3/Eigen/Dense"

namespace endpoint_constraint {

class EndPointConstraint {

 public:

  EndPointConstraint(std::function<void(const Eigen::VectorXd *,
                                        Eigen::VectorXd &)> *constraint,
                     const int constraint_dimension,
                     const bool implicit) :
      constraint(*constraint),
      analytic_jacobian_given(false),
      constraint_dimension(constraint_dimension),
      implicit(implicit) {};

  EndPointConstraint(std::function<void(const Eigen::VectorXd *,
                                        Eigen::VectorXd &)> *constraint,
                     std::function<void(const Eigen::VectorXd *,
                                        Eigen::MatrixXd &)> *constraint_jacobian,
                     const int constraint_dimension,
                     const bool implicit) :
      constraint(*constraint),
      constraint_jacobian(*constraint_jacobian),
      analytic_jacobian_given(true),
      constraint_dimension(constraint_dimension),
      implicit(implicit) {};

  void eval_constraint(const Eigen::VectorXd *x,
                       Eigen::VectorXd &h);

  long eval_active_indices(const Eigen::VectorXd *constraint,
                           Eigen::Matrix<bool, Eigen::Dynamic, 1> &active_indices);

  void eval_constraint_jacobian(const Eigen::VectorXd *x,
                                Eigen::MatrixXd &Hx);

  int get_constraint_dimension() { return constraint_dimension; }

  bool is_implicit() { return implicit; }

  void make_implicit() { this->implicit = true; }

 private:
  std::function<void(const Eigen::VectorXd *, Eigen::VectorXd &)> constraint;
  std::function<void(const Eigen::VectorXd *, Eigen::MatrixXd &)> constraint_jacobian;

  bool analytic_jacobian_given;

  bool implicit;

  const int constraint_dimension;
};

} // namespace endpoint_constraint


#endif //MOTORBOAT_ENDPOINT_CONSTRAINT_H
