//
// Created by Forrest Laine on 7/6/18.
//

#include <iostream>
#include <vector>
#include "trajectory.h"
#include "parent_trajectory.h"
#include "gtest/gtest.h"
#include <Eigen3/Eigen/Dense>

#include "dynamics.h"
#include "running_constraint.h"
#include "endpoint_constraint.h"
#include "running_cost.h"
#include "terminal_cost.h"

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

namespace test_trajectory {

const int n = 32;
const int m = 10;
const int T = 4;
const int l = 2;
const int runs = 100;

const Eigen::MatrixXd Q = Eigen::MatrixXd::Identity(n, n);
const Eigen::MatrixXd R = Eigen::MatrixXd::Identity(m, m);
const Eigen::MatrixXd S = Eigen::MatrixXd::Zero(m, n);
const Eigen::MatrixXd I = Eigen::MatrixXd::Identity(n, n);
const Eigen::MatrixXd B = Eigen::MatrixXd::Random(n, m);
const Eigen::MatrixXd A = Eigen::MatrixXd::Random(n,n);
const Eigen::VectorXd q = Eigen::VectorXd::Constant(n, 1.0);
const Eigen::VectorXd r = Eigen::VectorXd::Constant(m, 1.0);
const Eigen::VectorXd c = Eigen::VectorXd::Constant(n, 1.0);
const Eigen::VectorXd x0 = Eigen::VectorXd::Constant(n, 1.0);
const Eigen::VectorXd xT = Eigen::VectorXd::Constant(n, 2.0);

std::function<double(const Eigen::VectorXd *)> terminal_cost_f = [](const Eigen::VectorXd *x) {
  Eigen::MatrixXd val = 0.5 * (*x).transpose() * Q * (*x) + (*x).transpose() * q;
  return val(0, 0);
};

std::function<void(const Eigen::VectorXd *, Eigen::VectorXd &)>
    terminal_grad_f = [](const Eigen::VectorXd *x, Eigen::VectorXd &g) {
  g = Q * (*x) + q;
};

std::function<void(const Eigen::VectorXd *, Eigen::MatrixXd &)>
    terminal_hess_f = [](const Eigen::VectorXd *x, Eigen::MatrixXd &H) {
  H = Q;
};

std::function<double(const Eigen::VectorXd *, const Eigen::VectorXd *)>
    running_cost_f = [](const Eigen::VectorXd *x, const Eigen::VectorXd *u) {
  Eigen::MatrixXd val =
      0.5 * ((*x).transpose() * Q * (*x) + (*u).transpose() * R * (*u)) + (*u).transpose() * S * (*x)
          + (*x).transpose() * q + (*u).transpose() * r;
  return val(0, 0);
};

std::function<void(const Eigen::VectorXd *, const Eigen::VectorXd *, Eigen::VectorXd &)>
    running_grad_x_f = [](const Eigen::VectorXd *x, const Eigen::VectorXd *u, Eigen::VectorXd &gx) {
  gx = Q * (*x) + S.transpose() * (*u) + q;
};

std::function<void(const Eigen::VectorXd *, const Eigen::VectorXd *, Eigen::VectorXd &)>
    running_grad_u_f = [](const Eigen::VectorXd *x, const Eigen::VectorXd *u, Eigen::VectorXd &gu) {
  gu = R * (*u) + S * (*x) + r;
};

std::function<void(const Eigen::VectorXd *, const Eigen::VectorXd *, Eigen::MatrixXd &)>
    running_hess_xx_f = [](const Eigen::VectorXd *x, const Eigen::VectorXd *u, Eigen::MatrixXd &Hxx) {
  Hxx = Q;
};

std::function<void(const Eigen::VectorXd *, const Eigen::VectorXd *, Eigen::MatrixXd &)>
    running_hess_ux_f = [](const Eigen::VectorXd *x, const Eigen::VectorXd *u, Eigen::MatrixXd &Hux) {
  Hux = S;
};

std::function<void(const Eigen::VectorXd *, const Eigen::VectorXd *, Eigen::MatrixXd &)>
    running_hess_uu_f = [](const Eigen::VectorXd *x, const Eigen::VectorXd *u, Eigen::MatrixXd &Huu) {
  Huu = R;
};

std::function<void(const Eigen::VectorXd *, const Eigen::VectorXd *, Eigen::VectorXd &)>
    dynamics_f = [](const Eigen::VectorXd *x, const Eigen::VectorXd *u, Eigen::VectorXd &xx) {
//  Eigen::MatrixXd A = Eigen::MatrixXd::Identity(n, n);
//  for (int i = 0; i < n - 1; ++i) {
//    A(i + 1, i) = 1.0;
//  }
  xx = A * (*x) + B * (*u) + c;
};

std::function<void(const Eigen::VectorXd *, const Eigen::VectorXd *, Eigen::MatrixXd &)>
    dynamics_jac_x_f = [](const Eigen::VectorXd *x, const Eigen::VectorXd *u, Eigen::MatrixXd &Ax) {
//  Eigen::MatrixXd A = Eigen::MatrixXd::Identity(n, n);
//  for (int i = 0; i < n - 1; ++i) {
//    A(i + 1, i) = 1.0;
//  }
  Ax = A;
};

std::function<void(const Eigen::VectorXd *, const Eigen::VectorXd *, Eigen::MatrixXd &)>
    dynamics_jac_u_f = [](const Eigen::VectorXd *x, const Eigen::VectorXd *u, Eigen::MatrixXd &Au) {
  Au = B;
};

std::function<void(const Eigen::VectorXd *, Eigen::VectorXd &)>
    initial_constraint_f = [](const Eigen::VectorXd *x, Eigen::VectorXd &c) {
  c = (*x) - x0;
};

std::function<void(const Eigen::VectorXd *, Eigen::MatrixXd &)>
    initial_constraint_jac_f = [](const Eigen::VectorXd *x, Eigen::MatrixXd &J) {
  J = I;
};

std::function<void(const Eigen::VectorXd *, Eigen::VectorXd &)>
    terminal_constraint_f = [](const Eigen::VectorXd *x, Eigen::VectorXd &c) {
  c = (*x) - xT;
};

std::function<void(const Eigen::VectorXd *, Eigen::MatrixXd &)>
    terminal_constraint_jac_f = [](const Eigen::VectorXd *x, Eigen::MatrixXd &J) {
  J = I;
};

std::function<void(const Eigen::VectorXd *, const Eigen::VectorXd *, int, Eigen::VectorXd &)>
    running_constraint_f = [](const Eigen::VectorXd *x, const Eigen::VectorXd *u, int t, Eigen::VectorXd &c) {};

std::function<void(const Eigen::VectorXd *, const Eigen::VectorXd *, int, Eigen::MatrixXd &)>
    running_constraint_jac_x_f = [](const Eigen::VectorXd *x, const Eigen::VectorXd *u, int t, Eigen::MatrixXd &Jx) {};

std::function<void(const Eigen::VectorXd *, const Eigen::VectorXd *, int, Eigen::MatrixXd &)>
    running_constraint_jac_u_f = [](const Eigen::VectorXd *x, const Eigen::VectorXd *u, int t, Eigen::MatrixXd &Ju) {};

TEST(TestTrajectory, LQRSimple) {

  dynamics::Dynamics dynamics_obj = dynamics::Dynamics(&dynamics_f, &dynamics_jac_x_f, &dynamics_jac_u_f);
  running_cost::RunningCost running_cost_obj = running_cost::RunningCost(&running_cost_f,
                                                                         &running_grad_x_f,
                                                                         &running_grad_u_f,
                                                                         &running_hess_xx_f,
                                                                         &running_hess_ux_f,
                                                                         &running_hess_uu_f);
  terminal_cost::TerminalCost terminal_cost_obj = terminal_cost::TerminalCost(&terminal_cost_f,
                                                                              &terminal_grad_f,
                                                                              &terminal_hess_f);
  endpoint_constraint::EndPointConstraint
      terminal_constraint_obj = endpoint_constraint::EndPointConstraint(&terminal_constraint_f,
                                                                        &terminal_constraint_jac_f,
                                                                        n,
                                                                        true);

  endpoint_constraint::EndPointConstraint
      initial_constraint_obj = endpoint_constraint::EndPointConstraint(&initial_constraint_f,
                                                                       &initial_constraint_jac_f,
                                                                       n,
                                                                       true);
  running_constraint::RunningConstraint
      running_constraint_obj = running_constraint::RunningConstraint(&running_constraint_f,
                                                                     &running_constraint_jac_x_f,
                                                                     &running_constraint_jac_u_f,
                                                                     0);

  int term_dim = (terminal_constraint_obj.is_implicit()) ? terminal_constraint_obj.get_constraint_dimension() : 0;

  parent_trajectory::ParentTrajectory parent_traj(T,
                                                  n,
                                                  m,
                                                  n,
                                                  0,
                                                  0,
                                                  &dynamics_obj,
                                                  &running_constraint_obj,
                                                  &terminal_constraint_obj,
                                                  &initial_constraint_obj,
                                                  &running_cost_obj,
                                                  &terminal_cost_obj);

  trajectory::Trajectory test_traj(T,
                                   n,
                                   m,
                                   &dynamics_obj,
                                   &running_constraint_obj,
                                   &terminal_constraint_obj,
                                   &initial_constraint_obj,
                                   &running_cost_obj,
                                   &terminal_cost_obj);

  test_traj.populate_derivative_terms();
  std::cout << "Made terms serial" << std::endl;
  parent_traj.initial_state = x0;
  parent_traj.setNumChildTrajectories(l);
  parent_traj.initializeChildTrajectories();
  parent_traj.populateChildDerivativeTerms();
  std::cout << "Made terms parallel" << std::endl;

//  std::cout<<test_traj.hamiltonian_hessians_state_state[0]<<std::endl<<std::endl;
//  std::cout<<test_traj.hamiltonian_hessians_control_control[0]<<std::endl<<std::endl;
//  std::cout<<test_traj.hamiltonian_hessians_control_state[0]<<std::endl<<std::endl;
//
//  std::cout<<test_traj.dynamics_jacobians_state[0]<<std::endl<<std::endl;
//  std::cout<<test_traj.dynamics_jacobians_control[0]<<std::endl<<std::endl;
//  std::cout<<test_traj.dynamics_affine_terms[0]<<std::endl<<std::endl;
//
//  std::cout<<test_traj.dynamics_jacobians_state[T-2]<<std::endl<<std::endl;
//  std::cout<<test_traj.dynamics_jacobians_control[T-2]<<std::endl<<std::endl;
//  std::cout<<test_traj.dynamics_affine_terms[T-2]<<std::endl<<std::endl;
//
//  std::cout<<test_traj.hamiltonian_hessians_state_state[T-2]<<std::endl<<std::endl;
//  std::cout<<test_traj.hamiltonian_hessians_control_control[T-2]<<std::endl<<std::endl;
//  std::cout<<test_traj.hamiltonian_hessians_control_state[T-2]<<std::endl<<std::endl;
//
//  std::cout<<test_traj.terminal_cost_hessians_state_state<<std::endl<<std::endl;
//  std::cout<<test_traj.terminal_cost_gradient_state<<std::endl<<std::endl;






  std::cout << "Starting computation" << std::endl;
  double min_serial_time = 10000.0;
  double min_parallel_time = 10000.0;
  double min_banded_time = 10000.0;
  double min_pfb_time = 10000.0;
  double min_ptr_time = 10000.0;
  double min_pmu_time = 10000.0;

  double avg_serial_time = 0.0;
  double avg_parallel_time = 0.0;
  double avg_banded_time = 0.0;
  double avg_pfb_time = 0.0;
  double avg_ptr_time = 0.0;
  double avg_pmu_time = 0.0;

  double t0, t1, t2, t3, t4;
  for (int run = 0; run < runs; ++run) {
    test_traj.populate_derivative_terms();
    t0 = omp_get_wtime();
    test_traj.compute_feedback_policies();
    t1 = omp_get_wtime();
    test_traj.compute_state_control_dependencies();
    t2 = omp_get_wtime();
    test_traj.compute_multipliers();
    t3 = omp_get_wtime();
    min_serial_time = std::min(t3-t0, min_serial_time);
    avg_serial_time += (t3-t0);
//    std::cout<<"FB time: "<<t1-t0<<std::endl;
//    std::cout<<"traj time: "<<t2-t1<<std::endl;
//    std::cout<<"mult time: "<<t3-t2<<std::endl;

    parent_traj.initial_state = x0;
    parent_traj.setNumChildTrajectories(l);
    parent_traj.initializeChildTrajectories();
    parent_traj.populateChildDerivativeTerms();
    t0 = omp_get_wtime();
//    parent_traj.performChildTrajectoryCalculations();
    parent_traj.calculateFeedbackPolicies();
    t1 = omp_get_wtime();
    parent_traj.computeStateAndControlDependencies();
    t2 = omp_get_wtime();
    parent_traj.computeMultipliers();
    t3 = omp_get_wtime();
    parent_traj.solveForChildTrajectoryLinkPoints();
    t4 = omp_get_wtime();
    min_parallel_time = std::min( t4-t0, min_parallel_time);
    avg_parallel_time += (t4-t0);

    min_pfb_time = std::min( t1-t0, min_pfb_time);
    avg_pfb_time += (t1-t0);

    min_ptr_time = std::min( t2-t1, min_ptr_time);
    avg_ptr_time += (t2-t1);

    min_pmu_time = std::min( t3-t2, min_pmu_time);
    avg_pmu_time += (t3-t2);

  }

  avg_serial_time /= runs;
  avg_parallel_time /= runs;
  avg_pfb_time /= runs;
  avg_pmu_time /= runs;
  avg_ptr_time /= runs;
//  total_serial_time = total_serial_time / runs;
//  total_parallel_time = total_parallel_time / runs;


//  test_traj.populate_bandsolve_terms();
//  for (int run = 0; run < runs; ++run) {
//    t0 = omp_get_wtime();
//    test_traj.bandsolve_traj();
//    t1 = omp_get_wtime();
//    std::cout<<"Band time : " <<t1-t0<<std::endl;
//  }
//

//  std::cout<<parent_traj.child_trajectories[1].hamiltonian_hessians_state_state[0]<<std::endl<<std::endl;
//  std::cout<<parent_traj.child_trajectories[1].hamiltonian_hessians_control_state[0]<<std::endl<<std::endl;
//  std::cout<<parent_traj.child_trajectories[1].hamiltonian_hessians_control_control[0]<<std::endl<<std::endl;
//
//  std::cout<<parent_traj.child_trajectories[1].terminal_cost_hessians_state_state<<std::endl<<std::endl;
//
//  std::cout<<parent_traj.child_trajectories[1].terminal_constraint_jacobian_terminal_projection<<std::endl<<std::endl;
//  std::cout<<parent_traj.child_trajectories[1].terminal_constraint_jacobian_state<<std::endl<<std::endl;
//  std::cout<<parent_traj.child_trajectories[1].terminal_constraint_affine_term<<std::endl<<std::endl;
//  std::cout<<parent_traj.child_trajectories[1].terminal_constraint_dimension<<std::endl;

  Eigen::VectorXd x(n);
  Eigen::VectorXd u(m);

  Eigen::MatrixXd Tx;
  Eigen::MatrixXd Tz;
  Eigen::VectorXd T1;

  Eigen::MatrixXd Ux;
  Eigen::MatrixXd Uz;
  Eigen::VectorXd U1;

  Eigen::MatrixXd Vxx, Vzx, Vzz, V11;
  Eigen::VectorXd Vx1, Vz1;

//  x = x0.eval();
//  for (int t = 0; t < T; ++t) {
//    test_traj.get_state_dependencies_initial_state_projection(t, Tx);
//    test_traj.get_state_dependencies_terminal_state_projection(t, Tz);
//    test_traj.get_state_dependencies_affine_term(t, T1);
//
//    Eigen::VectorXd temp = Tx * (-x0) + T1;
//    if (term_dim > 0) {
//      temp += Tz * (-xT);
//    }
//    x = temp;
//    std::cout << "t: " << t + 1 << std::endl;
//
//    if (t < T - 1) {
//      x = ((test_traj.dynamics_jacobians_state[t]
//          + test_traj.dynamics_jacobians_control[t] * test_traj.current_state_feedback_matrices[t]) * x
//          + test_traj.dynamics_jacobians_control[t] * test_traj.feedforward_controls[t]
//          + test_traj.dynamics_affine_terms[t]).eval();
//
//      if (term_dim > 0) {
//        x += (test_traj.dynamics_jacobians_control[t] * test_traj.terminal_state_feedback_matrices[t]) * (-xT);
//      }
//
//      std::cout << "x: " << temp.transpose() << std::endl;
//
//      test_traj.get_control_dependencies_initial_state_projection(t, Ux);
//      test_traj.get_control_dependencies_terminal_state_projection(t, Uz);
//      test_traj.get_control_dependencies_affine_term(t, U1);
//      Eigen::VectorXd temp2 = Ux * (-x0) + U1;
//      if (term_dim > 0) {
//        temp2 += Uz * (-xT);
//      }
//      std::cout << "u: " << temp2.transpose() << std::endl;
////      std::cout << "xx: " << x.transpose() << std::endl;
//    } else {
//      std::cout << "x: " << temp.transpose() << std::endl;
//
//    }
//
//    test_traj.get_dynamics_mult_initial_state_feedback_term(t, Tx);
//    test_traj.get_dynamics_mult_terminal_state_feedback_term(t, Tz);
//    test_traj.get_dynamics_mult_feedforward_term(t, T1);
//
//    Eigen::VectorXd lam = Tx * (-x0) + T1;
//    if (term_dim > 0) {
//      lam += Tz * (-xT);
//    }
//    std::cout << "Lam " << t << " = " << lam.transpose() << std::endl;
//
//    Vxx = test_traj.cost_to_go_hessians_state_state[t];
//    Vzx = test_traj.cost_to_go_hessians_terminal_state[t];
//    Vzz = test_traj.cost_to_go_hessians_terminal_terminal[t];
//    Vx1 = test_traj.cost_to_go_gradients_state[t];
//    Vz1 = test_traj.cost_to_go_gradients_terminal[t];
//    V11 = test_traj.cost_to_go_offsets[t];
//    Eigen::VectorXd lam2 = Vxx * temp + Vx1;
//    if (term_dim > 0) {
//      lam2 += Vzx.transpose() * (-xT);
//    }
////    std::cout << "Lam " << t << " = " << lam2.transpose() << std::endl;
//    if (t < T - 1) {
//      test_traj.get_running_constraint_mult_initial_state_feedback_term(t, Ux);
//      test_traj.get_running_constraint_mult_terminal_state_feedback_term(t, Uz);
//      test_traj.get_running_constraint_mult_feedforward_term(t, U1);
//
//      Eigen::VectorXd mu = Ux * (-x0) + U1;
//      if (term_dim > 0) {
//        mu += Uz * (-xT);
//      }
//      std::cout << "Mu " << t << " = " << mu.transpose() << std::endl;
//    } else {
//      test_traj.get_terminal_constraint_mult_initial_state_feedback_term(Ux);
//      test_traj.get_terminal_constraint_mult_terminal_state_feedback_term(Uz);
//      test_traj.get_terminal_constraint_mult_feedforward_term(U1);
//      Eigen::VectorXd mu = Ux * (-x0) + U1;
//      if (term_dim > 0) {
//        mu += Uz * (-xT);
//      }
//      std::cout << "Mu " << t << " = " << mu.transpose() << std::endl;
//    }
//  }
//  test_traj.get_state_dependencies_initial_state_projection(T-1, Tx);
//  test_traj.get_state_dependencies_terminal_state_projection(T-1, Tz);
//  test_traj.get_state_dependencies_affine_term(T-1, T1);
//
//  Eigen::VectorXd temp = Tx * (-x0) + T1;
//  if (term_dim > 0) {
//    temp += Tz * (-xT);
//  }
//  std::cout<<temp.transpose()<<std::endl;
////
//
//
//  Eigen::VectorXd x00, xTT;
//
//  for (unsigned int i = 0; i < l; ++i) {
//    for (int t = 0; t < parent_traj.child_trajectory_lengths[i]; ++t) {
//      parent_traj.child_trajectories[i].get_state_dependencies_initial_state_projection(t, Tx);
//      parent_traj.child_trajectories[i].get_state_dependencies_terminal_state_projection(t, Tz);
//      parent_traj.child_trajectories[i].get_state_dependencies_affine_term(t, T1);
//
//      if (i < l - 1) {
//        xTT = -parent_traj.child_trajectory_link_points[i];
//      } else {
//        xTT = xT;
//      }
//      if (i > 0) {
//        x00 = -parent_traj.child_trajectory_link_points[i - 1];
//      } else {
//        x00 = x0;
//      }
//
//      Eigen::VectorXd temp = Tx * (-x00) + T1;// + Tz * (-xT);
//      if (i < l - 1 or (terminal_constraint_obj.get_constraint_dimension() > 0 and terminal_constraint_obj.is_implicit())) {
//        temp += (Tz * (-xTT)).eval();
//      }
//      x = temp;
//      std::cout << "t: " << t + 1 << std::endl;
//      std::cout << "x: " << x.transpose() << std::endl;
//      if (t < parent_traj.child_trajectory_lengths[i] - 1) {
//        parent_traj.child_trajectories[i].get_control_dependencies_initial_state_projection(t, Ux);
//        parent_traj.child_trajectories[i].get_control_dependencies_terminal_state_projection(t, Uz);
//        parent_traj.child_trajectories[i].get_control_dependencies_affine_term(t, U1);
//        Eigen::VectorXd temp2 = Ux * (-x00) + U1;// + Uz * (-xT);
//        if (i < l - 1 or (terminal_constraint_obj.get_constraint_dimension() > 0 and terminal_constraint_obj.is_implicit())) {
//          temp2 += (Uz * (-xTT)).eval();
//        }
//        std::cout << "u: " << temp2.transpose() << std::endl;
//      }
//      parent_traj.child_trajectories[i].get_dynamics_mult_initial_state_feedback_term(t, Tx);
//      parent_traj.child_trajectories[i].get_dynamics_mult_terminal_state_feedback_term(t, Tz);
//      parent_traj.child_trajectories[i].get_dynamics_mult_feedforward_term(t, T1);
//
//      Eigen::VectorXd lam = Tx * (-x0) + T1;
//      if (term_dim > 0) {
//        lam += Tz * (-xT);
//      }
//      std::cout << "Lam " << t << " = " << lam.transpose() << std::endl;
//    }
//  }

//  parent_traj.child_trajectories[l-2].get_state_dependencies_initial_state_projection(parent_traj.child_trajectory_lengths[l-2]-1, Tx);
//  parent_traj.child_trajectories[l-2].get_state_dependencies_terminal_state_projection(parent_traj.child_trajectory_lengths[l-2]-1, Tz);
//  parent_traj.child_trajectories[l-2].get_state_dependencies_affine_term(parent_traj.child_trajectory_lengths[l-2]-1, T1);
//  temp = Tx * (-x0) + T1;
//  if (term_dim > 0) {
//    temp += Tz * (-xT);
//  }
//  std::cout<<temp.transpose()<<std::endl;
//
//  parent_traj.child_trajectories[l-1].get_state_dependencies_initial_state_projection(parent_traj.child_trajectory_lengths[l-1]-1, Tx);
//  parent_traj.child_trajectories[l-1].get_state_dependencies_terminal_state_projection(parent_traj.child_trajectory_lengths[l-1]-1, Tz);
//  parent_traj.child_trajectories[l-1].get_state_dependencies_affine_term(parent_traj.child_trajectory_lengths[l-1]-1, T1);
//  temp = Tx * (-x0) + T1;
//  if (term_dim > 0) {
//    temp += Tz * (-xT);
//  }
//  std::cout<<temp.transpose()<<std::endl;

//  for (int i = 0; i < test_traj.soln_size; ++i) {
//    std::cout<<test_traj.B[i]<<" ";
//  }
//  std::cout<<std::endl;

  std::cout<<"Num cores: " <<l<<std::endl;
  std::cout << "Serial time: " << min_serial_time << ","<<avg_serial_time<<std::endl;
  std::cout << "Parallel time: " << min_parallel_time << ","<<avg_parallel_time<<std::endl;
  std::cout << "Parallel fb time: " << min_pfb_time << ","<<avg_pfb_time<<std::endl;
  std::cout << "Parallel tr time: " << min_ptr_time << ","<<avg_ptr_time<<std::endl;
  std::cout << "Parallel mu time: " << min_pmu_time << ","<<avg_pmu_time<<std::endl;
  std::cout << "Banded time: " << min_banded_time << ","<<avg_banded_time<<std::endl;

//  std::cout << "x0: " << x0 << std::endl;
//  for (int tt = 0; tt < T; ++tt) {
//    test_traj.get_dynamics_mult_initial_state_feedback_term(tt, Tx);
//    test_traj.get_dynamics_mult_terminal_state_feedback_term(tt, Tz);
//    test_traj.get_dynamics_mult_feedforward_term(tt, T1);
//
//    Eigen::VectorXd lam = Tx * (-x0) + T1;
//    if (term_dim > 0) {
//      lam += Tz * (-xT);
//    }
//    std::cout << "Lam " << tt << " = " << lam.transpose() << std::endl;
//
//    if (tt < T - 1) {
//      test_traj.get_running_constraint_mult_initial_state_feedback_term(tt, Ux);
//      test_traj.get_running_constraint_mult_terminal_state_feedback_term(tt, Uz);
//      test_traj.get_running_constraint_mult_feedforward_term(tt, U1);
//
//      Eigen::VectorXd mu = Ux * (-x0) + U1;
//      if (term_dim > 0) {
//        mu += Uz * (-xT);
//      }
//      std::cout << "Mu " << tt << " = " << mu.transpose() << std::endl;
//    } else {
//      test_traj.get_terminal_constraint_mult_initial_state_feedback_term(Ux);
//      test_traj.get_terminal_constraint_mult_terminal_state_feedback_term(Uz);
//      test_traj.get_terminal_constraint_mult_feedforward_term(U1);
//      Eigen::VectorXd mu = Ux * (-x0) + U1;
//      if (term_dim > 0){
//        mu += Uz * (-xT);
//      }
//      std::cout << "Mu " << tt << " = " << mu.transpose() << std::endl;
//    }
//  }
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
}