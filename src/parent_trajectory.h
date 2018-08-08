//
// Created by Forrest Laine on 7/16/18.
//

#ifndef MOTORBOAT_PARENT_TRAJECTORY_H
#define MOTORBOAT_PARENT_TRAJECTORY_H

#include <vector>
#include "eigen3/Eigen/Dense"
#include "trajectory.h"

#include "dynamics.h"
#include "running_constraint.h"
#include "endpoint_constraint.h"
#include "running_cost.h"
#include "terminal_cost.h"

namespace parent_trajectory {

class ParentTrajectory {
 public:
  ParentTrajectory(unsigned int trajectory_length,
                   unsigned int state_dimension,
                   unsigned int control_dimension,
                   unsigned int initial_constraint_dimension,
                   unsigned int running_constraint_dimension,
                   unsigned int terminal_constraint_dimension,
                   const dynamics::Dynamics *dynamics,
                   const running_constraint::RunningConstraint *running_constraint,
                   const endpoint_constraint::EndPointConstraint *terminal_constraint,
                   const endpoint_constraint::EndPointConstraint *initial_constraint,
                   const running_cost::RunningCost *running_cost,
                   const terminal_cost::TerminalCost *terminal_cost) :
      trajectory_length(trajectory_length),
      num_child_trajectories(1),
      state_dimension(state_dimension),
      control_dimension(control_dimension),
      initial_constraint_dimension(initial_constraint_dimension),
      running_constraint_dimension(running_constraint_dimension),
      terminal_constraint_dimension(terminal_constraint_dimension),
      initial_state(Eigen::VectorXd::Zero(state_dimension)),
      terminal_projection(Eigen::VectorXd::Zero(terminal_constraint_dimension)),
      dynamics(*dynamics),
      running_constraint(*running_constraint),
      terminal_constraint(*terminal_constraint),
      initial_constraint(*initial_constraint),
      running_cost(*running_cost),
      terminal_cost(*terminal_cost),
      child_trajectories(std::vector<trajectory::Trajectory>()),
      empty_terminal_constraint(&this->simple_end_point_constraint,
                                &this->simple_end_point_constraint_jacobian,
                                state_dimension,
                                true),
      empty_terminal_cost(&this->empty_cost,
                          &this->empty_cost_gradient,
                          &this->empty_cost_hessian) {};

  ~ParentTrajectory() = default;

  void SetNumChildTrajectories(unsigned int num_threads);

  void PopulateChildDerivativeTerms();

  void InitializeChildTrajectories();

  void PerformChildTrajectoryCalculations();

  void PopulateDerivativeTerms();

  void CalculateFeedbackPolicies();

  void ComputeStateAndControlDependencies();

  void ComputeMultipliers();

  void SolveForChildTrajectoryLinkPoints();

  void UpdateChildTrajectories();

  std::function<void(const Eigen::VectorXd *, Eigen::VectorXd &)>
      simple_end_point_constraint = [](const Eigen::VectorXd *x, Eigen::VectorXd &val) {
    val = *x;
  };

  std::function<void(const Eigen::VectorXd *,
                            Eigen::MatrixXd &)> simple_end_point_constraint_jacobian = [](const Eigen::VectorXd *x,
                                                                                          Eigen::MatrixXd &val) { val.setIdentity(); };

  std::function<double(const Eigen::VectorXd *)> empty_cost = [](const Eigen::VectorXd *x) { return 0.0; };


  std::function<void(const Eigen::VectorXd *, Eigen::VectorXd &)> empty_cost_gradient = [](const Eigen::VectorXd *x, Eigen::VectorXd &g) { g.setZero(); };

  std::function<void(const Eigen::VectorXd *, Eigen::MatrixXd &)> empty_cost_hessian = [](const Eigen::VectorXd *x, Eigen::MatrixXd &H) { H.setZero(); };

 public:
  const unsigned int trajectory_length;
  unsigned int num_child_trajectories;
  const unsigned int state_dimension;
  const unsigned int control_dimension;
  const unsigned int initial_constraint_dimension;
  const unsigned int running_constraint_dimension;
  unsigned int terminal_constraint_dimension;

  Eigen::VectorXd initial_state;
  Eigen::VectorXd terminal_projection;

  dynamics::Dynamics dynamics;
  running_constraint::RunningConstraint running_constraint;
  endpoint_constraint::EndPointConstraint terminal_constraint;
  endpoint_constraint::EndPointConstraint initial_constraint;
  running_cost::RunningCost running_cost;
  terminal_cost::TerminalCost terminal_cost;

  std::vector<trajectory::Trajectory> child_trajectories;
  std::vector<unsigned int> child_trajectory_lengths;

  std::vector<Eigen::VectorXd> child_trajectory_link_points;

  std::vector<Eigen::MatrixXd> link_point_dependencies_prev_link_point;
  std::vector<Eigen::MatrixXd> link_point_dependencies_same_link_point;
  std::vector<Eigen::MatrixXd> link_point_dependencies_next_link_point;
  std::vector<Eigen::VectorXd> link_point_dependencies_affine_term;

  endpoint_constraint::EndPointConstraint empty_terminal_constraint;
  terminal_cost::TerminalCost empty_terminal_cost;

};

}  // namespace parent_trajectory

#endif //MOTORBOAT_PARENT_TRAJECTORY_H
