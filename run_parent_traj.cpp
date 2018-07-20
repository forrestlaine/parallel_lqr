//
// Created by Forrest Laine on 7/17/18.
//

#include <iostream>
#include <stdlib.h>
#include "trajectory.h"
#include "parent_trajectory.h"
#include "gtest/gtest.h"
#include <Eigen3/Eigen/Dense>
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

double running_cost(const Eigen::VectorXd *x, const Eigen::VectorXd *u) {
  return 0.0;
}

double terminal_cost(const Eigen::VectorXd *x) {
  return 0.0;
}

void dynamics(const Eigen::VectorXd *x, const Eigen::VectorXd *u, Eigen::VectorXd &xx) {};

void running_constraint(const Eigen::VectorXd *x, const Eigen::VectorXd *u, Eigen::VectorXd &g) {};

void final_constraint(const Eigen::VectorXd *x, Eigen::VectorXd &g) {};

void initial_constraint(const Eigen::VectorXd *x, Eigen::VectorXd &g) {};

void run_parent_traj(const Eigen::MatrixXd A,
                     const Eigen::MatrixXd B,
                     const Eigen::VectorXd c,
                     const Eigen::VectorXd r,
                     const Eigen::VectorXd x0,
                     unsigned int num_trajs,
                     unsigned int T,
                     unsigned int n,
                     unsigned int m) {
  const Eigen::MatrixXd Inn = Eigen::MatrixXd::Identity(n, n);
  const Eigen::MatrixXd Imm = Eigen::MatrixXd::Identity(m, m);
  const Eigen::VectorXd Zn = Eigen::VectorXd::Zero(n);

  std::vector<Eigen::MatrixXd> AA((unsigned int) num_trajs, A);
  std::vector<Eigen::MatrixXd> BB((unsigned int) num_trajs, B);
  std::vector<Eigen::VectorXd> cc((unsigned int) num_trajs, c);
  std::vector<Eigen::VectorXd> rr((unsigned int) num_trajs, r);

  std::function<void(const Eigen::VectorXd *, const Eigen::VectorXd *, Eigen::VectorXd &)> dynamics_f = dynamics;
  std::function<void(const Eigen::VectorXd *, const Eigen::VectorXd *, Eigen::VectorXd &)>
      running_constraint_f = running_constraint;
  std::function<void(const Eigen::VectorXd *, Eigen::VectorXd &)> final_constraint_f = final_constraint;
  std::function<void(const Eigen::VectorXd *, Eigen::VectorXd &)> initial_constraint_f = initial_constraint;
  std::function<double(const Eigen::VectorXd *, const Eigen::VectorXd *)> running_cost_f = running_cost;
  std::function<double(const Eigen::VectorXd *)> terminal_cost_f = terminal_cost;

  parent_trajectory::ParentTrajectory parent_traj = parent_trajectory::ParentTrajectory(T,
                                                                                        n,
                                                                                        m,
                                                                                        n,
                                                                                        0,
                                                                                        0,
                                                                                        &dynamics_f,
                                                                                        &running_constraint_f,
                                                                                        &final_constraint_f,
                                                                                        &initial_constraint_f,
                                                                                        &running_cost_f,
                                                                                        &terminal_cost_f);
  parent_traj.initial_state = x0;
  parent_traj.setNumChildTrajectories(num_trajs);
  parent_traj.initializeChildTrajectories();

  for (unsigned int i = 0; i < num_trajs; ++i) {
    parent_traj.child_trajectories[i].set_initial_constraint_dimension(n);
    parent_traj.child_trajectories[i].set_initial_constraint_jacobian_state(&Inn);
    parent_traj.child_trajectories[i].set_initial_constraint_affine_term(&Zn);
    for (unsigned int t = 0; t < parent_traj.child_trajectory_lengths[i] - 1; ++t) {
      parent_traj.child_trajectories[i].set_dynamics_jacobian_state(t, &AA[i]);
      parent_traj.child_trajectories[i].set_dynamics_jacobian_control(t, &BB[i]);
      parent_traj.child_trajectories[i].set_dynamics_affine_term(t, &cc[i]);
      parent_traj.child_trajectories[i].set_num_active_constraints(t, 0);
      parent_traj.child_trajectories[i].set_hamiltonian_hessians_state_state(t, &Inn);
      parent_traj.child_trajectories[i].set_hamiltonian_hessians_control_control(t, &Imm);
      parent_traj.child_trajectories[i].set_hamiltonian_gradients_state(t, &cc[i]);
      parent_traj.child_trajectories[i].set_hamiltonian_gradients_control(t, &rr[i]);
    }
    if (i < num_trajs - 1) {
      parent_traj.child_trajectories[i].set_terminal_constraint_dimension(n);
      parent_traj.child_trajectories[i].set_terminal_constraint_jacobian_state(&Inn);
      parent_traj.child_trajectories[i].set_terminal_constraint_jacobian_terminal_projection(&Inn);
      parent_traj.child_trajectories[i].set_terminal_constraint_affine_term(&Zn);
    } else {
      parent_traj.child_trajectories[i].set_terminal_constraint_dimension(0);
      parent_traj.child_trajectories[i].set_terminal_cost_hessians_state_state(&Inn);
      parent_traj.child_trajectories[i].set_terminal_cost_gradient_state(&c);
    }
  }
  double t0 = omp_get_wtime();
  parent_traj.performChildTrajectoryCalculations();
  parent_traj.solveForChildTrajectoryLinkPoints();
  double t1 = omp_get_wtime();
  for (unsigned int t = 0; t < parent_traj.num_child_trajectories - 1; ++t) {
    std::cout << parent_traj.child_trajectory_link_points[t].transpose() << std::endl;
  }
  std::cout << "Parallel Time: " << t1 - t0 << std::endl;

  Eigen::VectorXd x(n);
  Eigen::VectorXd u(m);
  Eigen::VectorXd xT(n);
  Eigen::VectorXd x00(n);

  Eigen::MatrixXd Tx;
  Eigen::MatrixXd Tz;
  Eigen::VectorXd T1;

  Eigen::MatrixXd Ux;
  Eigen::MatrixXd Uz;
  Eigen::VectorXd U1;

  for (unsigned int i = 0; i < num_trajs; ++i) {

    for (int t = 0; t < parent_traj.child_trajectory_lengths[i]; ++t) {
      parent_traj.child_trajectories[i].get_state_dependencies_initial_state_projection(t, Tx);
      parent_traj.child_trajectories[i].get_state_dependencies_terminal_state_projection(t, Tz);
      parent_traj.child_trajectories[i].get_state_dependencies_affine_term(t, T1);

      if (i < num_trajs - 1) {
        xT = parent_traj.child_trajectory_link_points[i];
      }
      if (i > 0) {
        x00 = parent_traj.child_trajectory_link_points[i-1];
      } else {
        x00 = -x0;
      }

      Eigen::VectorXd temp = Tx * (x00) + T1;// + Tz * (-xT);
      if (i < num_trajs - 1) {
        temp += (Tz * (xT)).eval();
      }
      x = temp;
      std::cout << "t: " << t + 1 << std::endl;
      std::cout << "x: " << x.transpose() << std::endl;
      if (t < parent_traj.child_trajectory_lengths[i] - 1) {
        parent_traj.child_trajectories[i].get_control_dependencies_initial_state_projection(t, Ux);
        parent_traj.child_trajectories[i].get_control_dependencies_terminal_state_projection(t, Uz);
        parent_traj.child_trajectories[i].get_control_dependencies_affine_term(t, U1);
        Eigen::VectorXd temp2 = Ux * (x00) + U1;// + Uz * (-xT);
        if (i < num_trajs - 1) {
          temp2 += (Uz * (xT)).eval();
        }
        std::cout << "u: " << temp2.transpose() << std::endl;
      }
    }
  }
}

void run_single_traj(const Eigen::MatrixXd A,
                     const Eigen::MatrixXd B,
                     const Eigen::VectorXd c,
                     const Eigen::VectorXd r,
                     const Eigen::VectorXd x0,
                     unsigned int num_trajs,
                     unsigned int T,
                     unsigned int n,
                     unsigned int m) {
  std::function<void(const Eigen::VectorXd *, const Eigen::VectorXd *, Eigen::VectorXd &)> dynamics_f = dynamics;
  std::function<void(const Eigen::VectorXd *, const Eigen::VectorXd *, Eigen::VectorXd &)>
      running_constraint_f = running_constraint;
  std::function<void(const Eigen::VectorXd *, Eigen::VectorXd &)> final_constraint_f = final_constraint;
  std::function<void(const Eigen::VectorXd *, Eigen::VectorXd &)>
      initial_constraint_f = initial_constraint;
  std::function<double(const Eigen::VectorXd *, const Eigen::VectorXd *)> running_cost_f = running_cost;
  std::function<double(const Eigen::VectorXd *)> terminal_cost_f = terminal_cost;

  trajectory::Trajectory test_traj(T,
                                   n,
                                   m,
                                   1,
                                   &dynamics_f,
                                   &running_constraint_f,
                                   &final_constraint_f,
                                   &initial_constraint_f,
                                   &running_cost_f,
                                   &terminal_cost_f);

  const Eigen::MatrixXd Inn = Eigen::MatrixXd::Identity(n, n);
  const Eigen::MatrixXd Imm = Eigen::MatrixXd::Identity(m, m);
  const Eigen::VectorXd Zn = Eigen::VectorXd::Zero(n);
  test_traj.set_initial_constraint_jacobian_state(&Inn);
  test_traj.set_initial_constraint_affine_term(&x0);

  test_traj.set_initial_constraint_dimension(n);
  test_traj.set_terminal_constraint_dimension(0);

  for (int t = 0; t < T - 1; ++t) {
    test_traj.set_dynamics_jacobian_state(t, &A);
    test_traj.set_dynamics_jacobian_control(t, &B);
    test_traj.set_dynamics_affine_term(t, &c);
    test_traj.set_num_active_constraints(t, 0);

    test_traj.set_hamiltonian_hessians_state_state(t, &Inn);
    test_traj.set_hamiltonian_hessians_control_control(t, &Imm);
    test_traj.set_hamiltonian_gradients_state(t, &c);
    test_traj.set_hamiltonian_gradients_control(t, &r);
  }
  test_traj.set_terminal_cost_hessians_state_state(&Inn);
  test_traj.set_terminal_cost_gradient_state(&c);

  std::cout << "Starting computation" << std::endl;

  double t0 = omp_get_wtime();

  test_traj.compute_feedback_policies();
  test_traj.compute_state_control_dependencies();
  test_traj.compute_multipliers();

  double t1 = omp_get_wtime();

  Eigen::VectorXd x(n);
  Eigen::VectorXd u(m);

  Eigen::MatrixXd Tx;
  Eigen::MatrixXd Tz;
  Eigen::VectorXd T1;

  Eigen::MatrixXd Ux;
  Eigen::MatrixXd Uz;
  Eigen::VectorXd U1;

  for (unsigned int t = 0; t < T; ++t) {
    test_traj.get_state_dependencies_initial_state_projection(t, Tx);
//    test_traj.get_state_dependencies_terminal_state_projection(t, Tz);
    test_traj.get_state_dependencies_affine_term(t, T1);

    auto temp = Tx * (-x0) + T1;// + Tz * (-xT);
    x = temp;
    std::cout << "t: " << t + 1 << std::endl;
    std::cout << "x: " << x.transpose() << std::endl;
    if (t < T - 1) {
      test_traj.get_control_dependencies_initial_state_projection(t, Ux);
//      test_traj.get_control_dependencies_terminal_state_projection(t, Uz);
      test_traj.get_control_dependencies_affine_term(t, U1);
      auto temp2 = Ux * (-x0) + U1;// + Uz * (-xT);
      std::cout << "u: " << temp2.transpose() << std::endl;
    }
  }
  std::cout<<"Serial time: " << t1-t0 << std::endl;
}

int main(int argc, char *argv[]) {

  int num_trajs = 0;
  int TT = 100;
  int n = 2;
  int m = 1;

  if (argc > 1) {
    if (argv[1] >= 0) {
      num_trajs = strtol(argv[1], nullptr, 0);
    }
  }

  if (argc > 2) {
    if (argv[2] >= 0) {
      TT = strtol(argv[2], nullptr, 0);
    }
  }

  if (argc > 3) {
    if (argv[3] >= 0) {
      n = strtol(argv[3], nullptr, 0);
    }
  }

  if (argc > 4) {
    if (argv[4] >= 0) {
      m = strtol(argv[4], nullptr, 0);
    }
  }

  const Eigen::MatrixXd A = Eigen::MatrixXd::Random(n, n);
  const Eigen::MatrixXd B = Eigen::MatrixXd::Random(n, m);
  const Eigen::VectorXd c = Eigen::VectorXd::Random(n);
  const Eigen::VectorXd r = Eigen::VectorXd::Random(m);
  const Eigen::VectorXd x0 = Eigen::VectorXd::Random(n);

//  Eigen::MatrixXd A(n, n);
//  Eigen::MatrixXd B(n, m);
//  Eigen::VectorXd c(n);
//  Eigen::VectorXd r(m);
//  A << 1.0, 1.0, 0.0, 1.0;
//  B << 0.0, 0.1;
//  c << 1.0, 1.0;
//  r << 1.0;
//  Eigen::VectorXd x0(n);
//  x0 << 5.0, 1.0;

  run_single_traj(A,B,c,r,x0,(unsigned int) num_trajs, (unsigned int) TT, (unsigned int) n, (unsigned int) m);
  run_parent_traj(A,B,c,r,x0,(unsigned int) num_trajs, (unsigned int) TT, (unsigned int) n, (unsigned int) m);
}