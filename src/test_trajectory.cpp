//
// Created by Forrest Laine on 7/6/18.
//

#include <iostream>
#include <vector>
#include "trajectory.h"
#include "parent_trajectory.h"
#include <eigen3/Eigen/Dense>

#include "dynamics.h"
#include "running_constraint.h"
#include "equality_constrained_running_constraint.h"
#include "equality_constrained_endpoint_constraint.h"
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

void RunTrajectoryTest(int n, int m, int T, int l, int runs) {

  const Eigen::MatrixXd Q = Eigen::MatrixXd::Identity(n, n);
  const Eigen::MatrixXd Qf = 10 * Eigen::MatrixXd::Identity(n, n);
  const Eigen::MatrixXd R = Eigen::MatrixXd::Identity(m, m);
  const Eigen::MatrixXd S = Eigen::MatrixXd::Zero(m, n);
  const Eigen::MatrixXd I = Eigen::MatrixXd::Identity(n, n);
  const Eigen::MatrixXd B = Eigen::MatrixXd::Random(n, m);
  const Eigen::MatrixXd A = Eigen::MatrixXd::Random(n, n);
  const Eigen::VectorXd q = Eigen::VectorXd::Constant(n, 1.0);
  const Eigen::VectorXd qf = Eigen::VectorXd::Constant(n, 0.0);
  const Eigen::VectorXd r = Eigen::VectorXd::Constant(m, 1.0);
  const Eigen::VectorXd c = Eigen::VectorXd::Constant(n, 1.0);
  const Eigen::VectorXd x0 = Eigen::VectorXd::Constant(n, 1.0);
  const Eigen::VectorXd xT = Eigen::VectorXd::Constant(n, 2.0);

  std::function<double(const Eigen::VectorXd *)> terminal_cost_f = [Qf, qf](const Eigen::VectorXd *x) {
    Eigen::MatrixXd val = 0.5 * (*x).transpose() * Qf * (*x) + (*x).transpose() * qf;
    return val(0, 0);
  };

  std::function<void(const Eigen::VectorXd *, Eigen::VectorXd &)>
      terminal_grad_f = [Qf, qf](const Eigen::VectorXd *x, Eigen::VectorXd &g) {
    g = Qf * (*x) + qf;
  };

  std::function<void(const Eigen::VectorXd *, Eigen::MatrixXd &)>
      terminal_hess_f = [Qf](const Eigen::VectorXd *, Eigen::MatrixXd &H) {
    H = Qf;
  };

  std::function<double(const Eigen::VectorXd *, const Eigen::VectorXd *)>
      running_cost_f = [Q, q, R, r, S](const Eigen::VectorXd *x, const Eigen::VectorXd *u) {
    Eigen::MatrixXd val =
        0.5 * ((*x).transpose() * Q * (*x) + (*u).transpose() * R * (*u)) + (*u).transpose() * S * (*x)
            + (*x).transpose() * q + (*u).transpose() * r;
    return val(0, 0);
  };

  std::function<void(const Eigen::VectorXd *, const Eigen::VectorXd *, Eigen::VectorXd &)>
      running_grad_x_f = [Q, q, S](const Eigen::VectorXd *x, const Eigen::VectorXd *u, Eigen::VectorXd &gx) {
    gx = Q * (*x) + S.transpose() * (*u) + q;
  };

  std::function<void(const Eigen::VectorXd *, const Eigen::VectorXd *, Eigen::VectorXd &)>
      running_grad_u_f = [R, r, S](const Eigen::VectorXd *x, const Eigen::VectorXd *u, Eigen::VectorXd &gu) {
    gu = R * (*u) + S * (*x) + r;
  };

  std::function<void(const Eigen::VectorXd *, const Eigen::VectorXd *, Eigen::MatrixXd &)>
      running_hess_xx_f = [Q](const Eigen::VectorXd *, const Eigen::VectorXd *, Eigen::MatrixXd &Hxx) {
    Hxx = Q;
  };

  std::function<void(const Eigen::VectorXd *, const Eigen::VectorXd *, Eigen::MatrixXd &)>
      running_hess_ux_f = [S](const Eigen::VectorXd *, const Eigen::VectorXd *, Eigen::MatrixXd &Hux) {
    Hux = S;
  };

  std::function<void(const Eigen::VectorXd *, const Eigen::VectorXd *, Eigen::MatrixXd &)>
      running_hess_uu_f = [R](const Eigen::VectorXd *, const Eigen::VectorXd *, Eigen::MatrixXd &Huu) {
    Huu = R;
  };

  std::function<void(const Eigen::VectorXd *, const Eigen::VectorXd *, Eigen::VectorXd &)>
      dynamics_f = [A, B, c](const Eigen::VectorXd *x, const Eigen::VectorXd *u, Eigen::VectorXd &xx) {
//  Eigen::MatrixXd A = Eigen::MatrixXd::Identity(n, n);
//  for (int i = 0; i < n - 1; ++i) {
//    A(i + 1, i) = 1.0;
//  }
    xx = A * (*x) + B * (*u) + c;
  };

  std::function<void(const Eigen::VectorXd *, const Eigen::VectorXd *, Eigen::MatrixXd &)>
      dynamics_jac_x_f = [A](const Eigen::VectorXd *, const Eigen::VectorXd *, Eigen::MatrixXd &Ax) {
//  Eigen::MatrixXd A = Eigen::MatrixXd::Identity(n, n);
//  for (int i = 0; i < n - 1; ++i) {
//    A(i + 1, i) = 1.0;
//  }
    Ax = A;
  };

  std::function<void(const Eigen::VectorXd *, const Eigen::VectorXd *, Eigen::MatrixXd &)>
      dynamics_jac_u_f = [B](const Eigen::VectorXd *, const Eigen::VectorXd *, Eigen::MatrixXd &Au) {
    Au = B;
  };

  std::function<void(const Eigen::VectorXd *, Eigen::VectorXd &)>
      initial_constraint_f = [x0](const Eigen::VectorXd *x, Eigen::VectorXd &c) {
    c = (*x) - x0;
  };

  std::function<void(const Eigen::VectorXd *, Eigen::MatrixXd &)>
      initial_constraint_jac_f = [I](const Eigen::VectorXd *, Eigen::MatrixXd &J) {
    J = I;
  };

  std::function<void(const Eigen::VectorXd *, Eigen::VectorXd &)>
      terminal_constraint_f = [xT](const Eigen::VectorXd *x, Eigen::VectorXd &c) {
    c = (*x) - xT;
  };

  std::function<void(const Eigen::VectorXd *, Eigen::MatrixXd &)>
      terminal_constraint_jac_f = [I](const Eigen::VectorXd *, Eigen::MatrixXd &J) {
    J = I;
  };

  std::function<void(const Eigen::VectorXd *, const Eigen::VectorXd *, int, Eigen::VectorXd &)>
      running_constraint_f =
      [m](const Eigen::VectorXd *x, const Eigen::VectorXd *u, int, Eigen::VectorXd &c) {
        c.setZero();
        for (int i = 0; i < m - 1; ++i) {
          c(i) = (*u)(i);
        }
      };

  std::function<void(const Eigen::VectorXd *, const Eigen::VectorXd *, int, Eigen::MatrixXd &)>
      running_constraint_jac_x_f =
      [m](const Eigen::VectorXd *, const Eigen::VectorXd *, int, Eigen::MatrixXd &Jx) { Jx.setZero(); };

  std::function<void(const Eigen::VectorXd *, const Eigen::VectorXd *, int, Eigen::MatrixXd &)>
      running_constraint_jac_u_f =
      [m](const Eigen::VectorXd *, const Eigen::VectorXd *, int, Eigen::MatrixXd &Ju) { Ju.setIdentity(m - 1, m); };

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

  equality_constrained_endpoint_constraint::EqualityConstrainedEndPointConstraint
      ec_terminal_constraint_obj =
      equality_constrained_endpoint_constraint::EqualityConstrainedEndPointConstraint(&terminal_constraint_f,
                                                                                      &terminal_constraint_jac_f,
                                                                                      0,
                                                                                      false);
  endpoint_constraint::EndPointConstraint
      terminal_constraint_obj =
      endpoint_constraint::EndPointConstraint(&terminal_constraint_f,
                                              &terminal_constraint_jac_f,
                                              0,
                                              false);

  equality_constrained_endpoint_constraint::EqualityConstrainedEndPointConstraint
      initial_constraint_obj =
      equality_constrained_endpoint_constraint::EqualityConstrainedEndPointConstraint(&initial_constraint_f,
                                                                                      &initial_constraint_jac_f,
                                                                                      n,
                                                                                      false);
  running_constraint::RunningConstraint
      running_constraint_obj = running_constraint::RunningConstraint(&running_constraint_f,
                                                                     &running_constraint_jac_x_f,
                                                                     &running_constraint_jac_u_f,
                                                                     0);

  equality_constrained_running_constraint::EqualityConstrainedRunningConstraint
      ec_running_constraint_obj =
      equality_constrained_running_constraint::EqualityConstrainedRunningConstraint(&running_constraint_f,
                                                                                    &running_constraint_jac_x_f,
                                                                                    &running_constraint_jac_u_f,
                                                                                    0);

//  equality_constrained_running_constraint::EqualityConstrainedRunningConstraint
//      ec_running_constraint_obj =
//      equality_constrained_running_constraint::EqualityConstrainedRunningConstraint(&running_constraint_f,
//                                                                                    m-1);

  parent_trajectory::ParentTrajectory parent_traj(T,
                                                  n,
                                                  m,
                                                  n,
                                                  0,
                                                  0,
                                                  &dynamics_obj,
                                                  &running_constraint_obj,
                                                  &ec_running_constraint_obj,
                                                  &terminal_constraint_obj,
                                                  &ec_terminal_constraint_obj,
                                                  &initial_constraint_obj,
                                                  &running_cost_obj,
                                                  &terminal_cost_obj);

  trajectory::Trajectory test_traj(T,
                                   n,
                                   m,
                                   &dynamics_obj,
                                   &running_constraint_obj,
                                   &ec_running_constraint_obj,
                                   &terminal_constraint_obj,
                                   &ec_terminal_constraint_obj,
                                   &initial_constraint_obj,
                                   &running_cost_obj,
                                   &terminal_cost_obj);

//  parent_traj.initial_state = x0;
//  parent_traj.SetNumChildTrajectories(l);
//  parent_traj.InitializeChildTrajectories();

  std::cout << "Starting computation" << std::endl;
  double min_serial_time = 10000.0;
  double min_parallel_time = 10000.0;
  double min_banded_time = 10000.0;
  double max_serial_time = 0.;
  double max_parallel_time = 0.;
  double min_pfb_time = 10000.0;
  double min_ptr_time = 10000.0;
  double min_pmu_time = 10000.0;
  double min_solve_time = 100000.;
  double max_banded_time = 0.;
  double max_pfb_time = 0.;
  double max_ptr_time = 0.;
  double max_pmu_time = 0.;
  double max_solve_time = 0.;

  double avg_serial_time = 0.0;
  double avg_parallel_time = 0.0;
  double avg_banded_time = 0.0;
  double avg_pfb_time = 0.0;
  double avg_ptr_time = 0.0;
  double avg_pmu_time = 0.0;
  double avg_solve_time = 0.;

  double t0, t1, t2, t3, t4;
  for (int run = 0; run < runs; ++run) {
    test_traj.populate_derivative_terms();
  //  if (run == 0) test_traj.populate_bandsolve_terms();
    t0 = omp_get_wtime();
  //  test_traj.bandsolve_traj();
    t1 = omp_get_wtime();
    //if (run == 0) test_traj.extract_bandsoln();
    min_banded_time = std::min(t1 - t0, min_banded_time);
    avg_banded_time += (t1 - t0);
    max_banded_time = std::max(t1 - t0, max_banded_time);

    t0 = omp_get_wtime();
    test_traj.compute_feedback_policies();
    t1 = omp_get_wtime();
    test_traj.compute_state_control_dependencies();
    t2 = omp_get_wtime();
    test_traj.compute_multipliers();
    t3 = omp_get_wtime();
    min_serial_time = std::min(t1 - t0, min_serial_time);
    avg_serial_time += (t1 - t0);
    max_serial_time = std::max(t1 - t0, max_serial_time);
    test_traj.set_open_loop_traj();

    parent_traj.initial_state = x0;
    parent_traj.SetNumChildTrajectories(l);
    parent_traj.InitializeChildTrajectories();
    parent_traj.PopulateChildDerivativeTerms();
    t0 = omp_get_wtime();
    parent_traj.PerformChildTrajectoryCalculations();
  //  parent_traj.CalculateFeedbackPolicies();
//    std::cout<<"1"<<std::endl;
    t1 = omp_get_wtime();
   // parent_traj.ComputeStateAndControlDependencies();
//    std::cout<<"2"<<std::endl;
    t2 = omp_get_wtime();
    //parent_traj.ComputeMultipliers();
//    std::cout<<"3"<<std::endl;
    t3 = omp_get_wtime();
    //parent_traj.SolveForChildTrajectoryLinkPoints(0);
//    std::cout<<"4"<<std::endl;
    t4 = omp_get_wtime();
    //parent_traj.SetOpenLoopTrajectories();

    min_parallel_time = std::min(t4 - t0, min_parallel_time);
    avg_parallel_time += (t4 - t0);
    max_parallel_time = std::max(t4 - t0, max_parallel_time);

    min_pfb_time = std::min(t1 - t0, min_pfb_time);
    avg_pfb_time += (t1 - t0);
    max_pfb_time = std::max(t1 - t0, max_pfb_time);

    min_ptr_time = std::min(t2 - t1, min_ptr_time);
    avg_ptr_time += (t2 - t1);
    max_ptr_time = std::max(t2 - t1, max_ptr_time);

    min_pmu_time = std::min(t3 - t2, min_pmu_time);
    avg_pmu_time += (t3 - t2);
    max_pmu_time = std::max(t3 - t2, max_pmu_time);

    min_solve_time = std::min(t4 - t3, min_solve_time);
    avg_solve_time += (t4 - t3);
    max_solve_time = std::max(t4 - t3, max_solve_time);
  }

//  Eigen::MatrixXd Kx, Kz;
//  Eigen::VectorXd k,xxx, ttt, uuu;
//  std::cout<<"Ctrls serial:" <<std::endl;
//  Kx = test_traj.current_state_feedback_matrices[49];
//  Kz = test_traj.terminal_state_feedback_matrices[49];
//  k = test_traj.feedforward_controls[49];
//  xxx = test_traj.open_loop_states[49];
//  ttt = test_traj.terminal_state_projection;
//
//  std::cout<<test_traj.open_loop_controls[49].transpose()<<std::endl;
//
//  uuu = Kx*xxx + k;
//  if (test_traj.implicit_terminal_constraint_dimension > 0) {
//    uuu += Kz.leftCols(test_traj.implicit_terminal_constraint_dimension) * ttt;
//  }
//  std::cout<<uuu.transpose()<<std::endl;
//
//  Kx = parent_traj.child_trajectories[0].current_state_feedback_matrices[49];
//  Kz = parent_traj.child_trajectories[0].terminal_state_feedback_matrices[49];
//  k = parent_traj.child_trajectories[0].feedforward_controls[49];
//  ttt = parent_traj.child_trajectories[0].terminal_state_projection;
//
//  std::cout<<"Ctrls parallel:" <<std::endl;
//  std::cout<<parent_traj.child_trajectories[0].open_loop_controls[49].transpose()<<std::endl;
//  uuu = Kx*xxx + Kz*ttt + k;
//
//
//  std::cout<<uuu.transpose()<<std::endl;

//  int offset = 2;
//  Eigen::VectorXd state = test_traj.open_loop_states[T / 2 - offset];
//  state(0) += 0.5;
//  Eigen::VectorXd ctr1, ctr2;
//
//  std::cout << test_traj.open_loop_states[T - offset].transpose() << std::endl;
//  std::cout << parent_traj.global_open_loop_states[T - offset].transpose() << std::endl << std::endl;
//
//  std::cout << test_traj.open_loop_controls[T / 2 - offset] << std::endl;
//  std::cout << parent_traj.global_open_loop_controls[T / 2 - offset] << std::endl << std::endl;
////
////  std::cout<<test_traj.current_state_feedback_matrices[T/2-offset]<<std::endl<<std::endl;
//  std::cout << test_traj.current_state_feedback_matrices[T / 2 - offset] * state << std::endl;
//  std::cout << test_traj.feedforward_controls[T / 2 - offset] << std::endl << std::endl;
//  ctr1 = test_traj.current_state_feedback_matrices[T / 2 - offset] * state
//      + test_traj.feedforward_controls[T / 2 - offset];
//  std::cout << ctr1.transpose() << std::endl;
//
////  std::cout<<parent_traj.child_trajectories[0].current_state_feedback_matrices[T/2-offset]<<std::endl;
//  std::cout << parent_traj.child_trajectories[0].current_state_feedback_matrices[T / 2 - offset] * state << std::endl;
////  std::cout<<parent_traj.child_trajectories[0].terminal_state_feedback_matrices[T/2-offset]<<std::endl;
//  std::cout<<"5"<<std::endl;
//  std::cout << parent_traj.child_trajectories[0].terminal_state_feedback_matrices[T / 2 - offset]
//      * parent_traj.child_trajectory_link_points[0] << std::endl;
//  std::cout<<"6"<<std::endl;
//  std::cout << parent_traj.child_trajectories[0].feedforward_controls[T / 2 - offset] << std::endl;
//  ctr2 = parent_traj.child_trajectories[0].current_state_feedback_matrices[T / 2 - offset] * state +
//      parent_traj.child_trajectories[0].terminal_state_feedback_matrices[T / 2 - offset]
//          * parent_traj.child_trajectory_link_points[0] +
//      parent_traj.child_trajectories[0].feedforward_controls[T / 2 - offset];
//  std::cout << ctr2.transpose() << std::endl;

  std::cout << "Error computation" << std::endl;
  double error = 0;
  for (int t = 0; t < T - 1; ++t) {
//    error += (test_traj.open_loop_states[t] - parent_traj.global_open_loop_states[t]).norm();
//    error += (test_traj.open_loop_controls[t] - parent_traj.global_open_loop_controls[t]).norm();
    error += (test_traj.open_loop_states[t] - test_traj.sanity_states[t]).norm();
    error += (test_traj.open_loop_controls[t] - test_traj.sanity_controls[t]).norm();
//    std::cout<<test_traj.open_loop_states[t].norm()<<std::endl;
//    std::cout<<test_traj.sanity_states[t].norm()<<std::endl;
//    std::cout<<parent_traj.global_open_loop_states[t].norm()<<std::endl<<std::endl;
  }

//  error += (test_traj.open_loop_states[T - 1] - parent_traj.global_open_loop_states[T - 1]).norm();
  error += (test_traj.open_loop_states[T - 1] - test_traj.sanity_states[T - 1]).norm();

  std::cout << "Error in trajectories: " << error << std::endl;

  avg_serial_time /= runs;
  avg_banded_time /= runs;
  avg_parallel_time /= runs;
  avg_pfb_time /= runs;
  avg_pmu_time /= runs;
  avg_ptr_time /= runs;
  avg_solve_time /= runs;

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

  std::cout << "Num cores: " << l << std::endl;
  std::cout << "Serial time: " << min_serial_time << "," << avg_serial_time << "," << max_serial_time << std::endl;
  std::cout << "Parallel time: " << min_parallel_time << "," << avg_parallel_time << "," << max_parallel_time
            << std::endl;
  std::cout << "Parallel fb time: " << min_pfb_time << "," << avg_pfb_time << "," << max_pfb_time << std::endl;
  std::cout << "Parallel tr time: " << min_ptr_time << "," << avg_ptr_time << "," << max_ptr_time << std::endl;
  std::cout << "Parallel mu time: " << min_pmu_time << "," << avg_pmu_time << "," << max_pmu_time << std::endl;
  std::cout << "Parallel solve time: " << min_solve_time << "," << avg_solve_time << "," << max_solve_time << std::endl;
  std::cout << "Banded time: " << min_banded_time << "," << avg_banded_time << "," << max_banded_time << std::endl;
}

int main(int argc, char **argv) {
  int args[5];

  args[0] = 2;   // n
  args[1] = 1;   // m
  args[2] = 100; // T
  args[3] = 2;   // cores
  args[4] = 1;   // runs

  for (int t = 1; t < argc; ++t) {
    args[t - 1] = atoi(argv[t]);
  }

  RunTrajectoryTest(args[0], args[1], args[2], args[3], args[4]);
  return 0;
}
