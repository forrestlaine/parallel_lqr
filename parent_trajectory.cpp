//
// Created by Forrest Laine on 7/16/18.
//
#define EIGEN_DONT_PARALLELIZE 1

#include "parent_trajectory.h"
#include <math.h>

#include <iostream>
#if defined(_OPENMP)
#include <omp.h>
#else
// omp timer replacement
#include <chrono>
double omp_get_wtime(void)
{
  static std::chrono::system_clock::time_point _start = std::chrono::system_clock::now();
  std::chrono::duration<double> elapsed = std::chrono::system_clock::now() - _start;
  return elapsed.count();
}

// omp functions used below
void omp_set_dynamic(int) {}
void omp_set_num_threads(int) {}
int omp_get_num_procs(void) {return 1;}

#endif

namespace parent_trajectory {

void ParentTrajectory::setNumChildTrajectories(unsigned int desired_num_threads) {
  unsigned int num_threads;
  if (desired_num_threads > 0) {
    num_threads = desired_num_threads;
  } else {
    num_threads = (unsigned int) omp_get_num_procs();
  }
  unsigned int total_trajectory_points = this->trajectory_length + num_threads - 1;
  unsigned int child_traj_length = total_trajectory_points / num_threads;
  unsigned int num_extra_length_trajectories = total_trajectory_points % num_threads;

  this->num_child_trajectories = num_threads;
  this->child_trajectory_lengths = std::vector<unsigned int>(num_threads, 0);

  // This works only if we assume controllability and that the break-points have no residual active constraints.
  // TODO: Write checks to ensure these assumptions are satisfied.
  for (unsigned int t = 0; t < num_threads; ++t) {
    this->child_trajectory_lengths[t] = child_traj_length;
    if (t < num_extra_length_trajectories) {
      this->child_trajectory_lengths[t] += 1;
    }
  }
}

void ParentTrajectory::initializeChildTrajectories() {
  for (unsigned int t = 0; t < this->num_child_trajectories; ++t) {
    unsigned int terminal_constraint_dimension =
        (t < this->num_child_trajectories - 1) ? this->state_dimension : this->terminal_constraint_dimension;
    std::function<void(const Eigen::VectorXd *, Eigen::VectorXd &)>
        final_constraint =
        (t < this->num_child_trajectories - 1) ? ParentTrajectory::simple_end_point_constraint : this->final_constraint;
    std::function<void(const Eigen::VectorXd *, Eigen::VectorXd &)>
        initial_constraint = (t > 0) ? ParentTrajectory::simple_end_point_constraint : this->initial_constraint;
    std::function<double(const Eigen::VectorXd *)>
        final_cost =
        (t < this->num_child_trajectories - 1) ? ParentTrajectory::empty_terminal_cost : this->terminal_cost;

    this->child_trajectories.emplace_back(trajectory::Trajectory(this->child_trajectory_lengths[t],
                                                                 this->state_dimension,
                                                                 this->control_dimension,
                                                                 this->running_constraint_dimension,
                                                                 &this->dynamics,
                                                                 &this->running_constraint,
                                                                 &final_constraint,
                                                                 &initial_constraint,
                                                                 &this->running_cost,
                                                                 &terminal_cost));
    this->child_trajectories[t].set_terminal_constraint_dimension(terminal_constraint_dimension);
  }
  this->link_point_dependencies_prev_link_point = std::vector<Eigen::MatrixXd>(this->num_child_trajectories - 1,
                                                                               Eigen::MatrixXd::Zero(this->state_dimension,
                                                                                                     this->state_dimension));
  this->link_point_dependencies_same_link_point = std::vector<Eigen::MatrixXd>(this->num_child_trajectories - 1,
                                                                               Eigen::MatrixXd::Zero(this->state_dimension,
                                                                                                     this->state_dimension));
  this->link_point_dependencies_next_link_point = std::vector<Eigen::MatrixXd>(this->num_child_trajectories - 1,
                                                                               Eigen::MatrixXd::Zero(this->state_dimension,
                                                                                                     this->state_dimension));
  this->link_point_dependencies_affine_term = std::vector<Eigen::VectorXd>(this->num_child_trajectories - 1,
                                                                           Eigen::VectorXd::Zero(this->state_dimension));
  this->child_trajectory_link_points = std::vector<Eigen::VectorXd>(this->num_child_trajectories-1, Eigen::VectorXd::Zero(this->state_dimension));
}

void ParentTrajectory::performChildTrajectoryCalculations() {
  #pragma omp parallel for num_threads(this->num_child_trajectories) 
  for (unsigned int t = 0; t < this->num_child_trajectories; ++t) {
    this->child_trajectories[t].populate_derivative_terms();
    this->child_trajectories[t].compute_feedback_policies();
    this->child_trajectories[t].compute_state_control_dependencies();
    this->child_trajectories[t].compute_multipliers();
  }
}

void ParentTrajectory::solveForChildTrajectoryLinkPoints() {

  unsigned int num_unknown_link_points = this->num_child_trajectories - 1;

  if (num_unknown_link_points > 0) {

    Eigen::MatrixXd Tx;
    Eigen::MatrixXd Tz;
    Eigen::VectorXd T1;

    this->child_trajectories[0].get_terminal_constraint_mult_initial_state_feedback_term(Tx);
    this->child_trajectories[0].get_terminal_constraint_mult_terminal_state_feedback_term(Tz);
    this->child_trajectories[0].get_terminal_constraint_mult_feedforward_term(T1);

    this->link_point_dependencies_affine_term[0] = Tx * (-this->initial_state) + T1;
    this->link_point_dependencies_same_link_point[0] = Tz;

    for (unsigned int t = 1; t < num_unknown_link_points; ++t) {
      this->child_trajectories[t].get_dynamics_mult_initial_state_feedback_term(0, Tx);
      this->child_trajectories[t].get_dynamics_mult_terminal_state_feedback_term(0, Tz);
      this->child_trajectories[t].get_dynamics_mult_feedforward_term(0, T1);

      this->link_point_dependencies_same_link_point[t - 1] += Tx;
      this->link_point_dependencies_next_link_point[t - 1] += Tz;
      this->link_point_dependencies_affine_term[t - 1] += T1;

      this->child_trajectories[t].get_terminal_constraint_mult_initial_state_feedback_term(Tx);
      this->child_trajectories[t].get_terminal_constraint_mult_terminal_state_feedback_term(Tz);
      this->child_trajectories[t].get_terminal_constraint_mult_feedforward_term(T1);

      this->link_point_dependencies_prev_link_point[t] += Tx;
      this->link_point_dependencies_same_link_point[t] += Tz;
      this->link_point_dependencies_affine_term[t] += T1;
    }
    this->child_trajectories[this->num_child_trajectories - 1].get_dynamics_mult_initial_state_feedback_term(0, Tx);
    this->child_trajectories[this->num_child_trajectories - 1].get_dynamics_mult_terminal_state_feedback_term(0, Tz);
    this->child_trajectories[this->num_child_trajectories - 1].get_dynamics_mult_feedforward_term(0, T1);

    this->link_point_dependencies_same_link_point[num_unknown_link_points - 1] += Tx;
    this->link_point_dependencies_affine_term[num_unknown_link_points - 1] +=
        Tz.leftCols(this->terminal_constraint_dimension) * this->terminal_projection + T1;

    Eigen::PartialPivLU<Eigen::MatrixXd>
        decomp(this->link_point_dependencies_same_link_point[num_unknown_link_points - 1]);
    this->link_point_dependencies_prev_link_point[num_unknown_link_points - 1] =
        (-decomp.solve(this->link_point_dependencies_prev_link_point[num_unknown_link_points - 1])).eval();
    this->link_point_dependencies_affine_term[num_unknown_link_points - 1] =
        (-decomp.solve(this->link_point_dependencies_affine_term[num_unknown_link_points - 1])).eval();

    if(num_unknown_link_points < 2) {
      this->child_trajectory_link_points[0] = this->link_point_dependencies_affine_term[0];
    }

    for (int t = num_unknown_link_points - 2; t > 0; --t) {
      decomp.compute(this->link_point_dependencies_same_link_point[t] + this->link_point_dependencies_next_link_point[t]
          * this->link_point_dependencies_prev_link_point[t + 1]);
      this->link_point_dependencies_prev_link_point[t] =
          (-decomp.solve(this->link_point_dependencies_prev_link_point[t])).eval();
      this->link_point_dependencies_affine_term[t] = (-decomp.solve(this->link_point_dependencies_affine_term[t]
                                                                        + this->link_point_dependencies_next_link_point[t]
                                                                            * this->link_point_dependencies_affine_term[
                                                                                t
                                                                                    + 1])).eval();
    }
    if (num_unknown_link_points > 1) {
      decomp.compute(this->link_point_dependencies_same_link_point[0]);

      this->link_point_dependencies_next_link_point[0] =
          (-decomp.solve(this->link_point_dependencies_next_link_point[0])).eval();
      this->link_point_dependencies_affine_term[0] =
          (-decomp.solve(this->link_point_dependencies_affine_term[0])).eval();

      decomp.compute(Eigen::MatrixXd::Identity(this->state_dimension, this->state_dimension)
                         - this->link_point_dependencies_prev_link_point[1]
                             * this->link_point_dependencies_next_link_point[0]);
      this->child_trajectory_link_points[1] = (decomp.solve(
          this->link_point_dependencies_prev_link_point[1] * this->link_point_dependencies_affine_term[0]
              + this->link_point_dependencies_affine_term[1])).eval();

//      this->link_point_dependencies_affine_term[1] = (decomp.solve(
//          this->link_point_dependencies_prev_link_point[1] * this->link_point_dependencies_affine_term[0]
//              + this->link_point_dependencies_affine_term[1])).eval();

      this->child_trajectory_link_points[0] = this->link_point_dependencies_affine_term[0]
          + (this->link_point_dependencies_next_link_point[0] * this->child_trajectory_link_points[1]).eval();
//      this->link_point_dependencies_affine_term[0] +=
//          (this->link_point_dependencies_next_link_point[0] * this->link_point_dependencies_affine_term[1]).eval();

      for (unsigned int t = 2; t < num_unknown_link_points; ++t) {
        this->child_trajectory_link_points[t] = this->link_point_dependencies_affine_term[t]
            + (this->link_point_dependencies_prev_link_point[t]
                * this->child_trajectory_link_points[t - 1]).eval();

//        this->link_point_dependencies_affine_term[t] +=
//            (this->link_point_dependencies_prev_link_point[t] * this->link_point_dependencies_affine_term[t - 1]).eval();
      }
    }
  }
}

}
