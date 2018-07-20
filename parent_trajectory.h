//
// Created by Forrest Laine on 7/16/18.
//

#ifndef MOTORBOAT_PARENT_TRAJECTORY_H
#define MOTORBOAT_PARENT_TRAJECTORY_H

#include <vector>
#include "Eigen3/Eigen/Dense"
#include "trajectory.h"


namespace parent_trajectory {

class ParentTrajectory
{
 public:
  ParentTrajectory(unsigned int trajectory_length,
                   unsigned int state_dimension,
                   unsigned int control_dimension,
                   unsigned int initial_constraint_dimension,
                   unsigned int running_constraint_dimension,
                   unsigned int terminal_constraint_dimension,
                   const std::function<void(const Eigen::VectorXd*, const Eigen::VectorXd*, Eigen::VectorXd&)> *dynamics,
                   const std::function<void(const Eigen::VectorXd*, const Eigen::VectorXd*, Eigen::VectorXd&)> *running_constraint,
                   const std::function<void(const Eigen::VectorXd*, Eigen::VectorXd&)> *final_constraint,
                   const std::function<void(const Eigen::VectorXd*, Eigen::VectorXd&)> *initial_constraint,
                   const std::function<double(const Eigen::VectorXd*, const Eigen::VectorXd*)> *running_cost,
                   const std::function<double(const Eigen::VectorXd*)> *terminal_cost):
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
      final_constraint(*final_constraint),
      initial_constraint(*initial_constraint),
      running_cost(*running_cost),
      terminal_cost(*terminal_cost),
      child_trajectories(std::vector<trajectory::Trajectory>()) {};

  ~ParentTrajectory() = default;

  void setNumChildTrajectories(unsigned int num_threads);

  void initializeChildTrajectories();

  void performChildTrajectoryCalculations();

  void solveForChildTrajectoryLinkPoints();

  void updateChildTrajectories();

  static double empty_terminal_cost(const Eigen::VectorXd* x) {return 0.0;};

  static void simple_end_point_constraint(const Eigen::VectorXd* x, Eigen::VectorXd& val) {val = *x;};


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


  std::function<void(const Eigen::VectorXd*, const Eigen::VectorXd*, Eigen::VectorXd&)> dynamics;
  std::function<void(const Eigen::VectorXd*, const Eigen::VectorXd*, Eigen::VectorXd&)> running_constraint;
  std::function<void(const Eigen::VectorXd*, Eigen::VectorXd&)> final_constraint;
  std::function<void(const Eigen::VectorXd*, Eigen::VectorXd&)> initial_constraint;
  std::function<double(const Eigen::VectorXd*, const Eigen::VectorXd*)> running_cost;
  std::function<double(const Eigen::VectorXd*)> terminal_cost;

  std::vector<trajectory::Trajectory> child_trajectories;
  std::vector<unsigned int> child_trajectory_lengths;

  std::vector<Eigen::VectorXd> child_trajectory_link_points;

  std::vector<Eigen::MatrixXd> link_point_dependencies_prev_link_point;
  std::vector<Eigen::MatrixXd> link_point_dependencies_same_link_point;
  std::vector<Eigen::MatrixXd> link_point_dependencies_next_link_point;
  std::vector<Eigen::VectorXd> link_point_dependencies_affine_term;

};

}  // namespace parent_trajectory

#endif //MOTORBOAT_PARENT_TRAJECTORY_H
