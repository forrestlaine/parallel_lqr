//
// Created by Forrest Laine on 7/6/18.
//

#include <iostream>
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


//std::function<void(const Eigen::VectorXd*, const Eigen::VectorXd*, Eigen::VectorXd&)> dynamics;
//std::function<void(const Eigen::VectorXd*, const Eigen::VectorXd*, Eigen::VectorXd&)> running_constraint;
//std::function<void(const Eigen::VectorXd*, const Eigen::VectorXd&)> final_constraint;
//std::function<void(const Eigen::VectorXd*, const Eigen::VectorXd*, Eigen::VectorXd&)> initial_constraint;
//std::function<double(const Eigen::VectorXd*, const Eigen::VectorXd*)> running_cost;
//std::function<double(const Eigen::VectorXd*)> terminal_cost;


const int n = 3;
const int m = 1;

double running_cost(const Eigen::VectorXd *x, const Eigen::VectorXd *u) {
  return 0.;
}

double terminal_cost(const Eigen::VectorXd *x) {
  return 0;
}

void dynamics(const Eigen::VectorXd *x, const Eigen::VectorXd *u, Eigen::VectorXd &xx) {};

void running_constraint(const Eigen::VectorXd *x, const Eigen::VectorXd *u, Eigen::VectorXd &g) {};

void final_constraint(const Eigen::VectorXd *x, Eigen::VectorXd &g) {};

void initial_constraint(const Eigen::VectorXd *x, Eigen::VectorXd &g) {};

TEST(TestTrajectory, LQRSimple) {
  const int T = 9;
  Eigen::MatrixXd A(n, n);
  Eigen::MatrixXd B(n, m);
  Eigen::VectorXd c(n);
  Eigen::VectorXd r(m);
  A << 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0;
  B << 0.0, 0.0, 0.1;//, 0.0, 0.0, 1.0;
  c << 1.0, 1.0, 1.0;
  r << 1.0;//, 1.0;
  Eigen::VectorXd x0(n);
  x0 << 5.0, 1.0, -1.0;

  std::cout << "A :" << A << std::endl;
  std::cout << "B :" << B << std::endl;

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
  const Eigen::MatrixXd Znn = Eigen::MatrixXd::Zero(n, n);

  test_traj.set_initial_constraint_jacobian_state(&Inn);
  test_traj.set_initial_constraint_affine_term(&x0);

  test_traj.set_initial_constraint_dimension(n);

  int term_dim = n;

  test_traj.set_terminal_constraint_dimension(term_dim);

  Eigen::VectorXd xT(n);
  Eigen::VectorXd xP(n);
  Eigen::VectorXd xPP(n);
  xT << 44.0, -4.0, 0.0;
  xP << 31.6301, 11.3699, -15.8389;
  xPP << 16.7639, 13.8662, -3.4175;

  for (int t = 0; t < T - 1; ++t) {
    test_traj.set_dynamics_jacobian_state(t, &A);
    test_traj.set_dynamics_jacobian_control(t, &B);
    test_traj.set_dynamics_affine_term(t, &c);
    test_traj.set_num_active_constraints(t, 0);

    test_traj.set_hamiltonian_hessians_state_state(t, &Znn);
    test_traj.set_hamiltonian_hessians_control_control(t, &Imm);
    test_traj.set_hamiltonian_gradients_state(t, &c);
    test_traj.set_hamiltonian_gradients_control(t, &r);
  }

  test_traj.set_terminal_cost_hessians_state_state(&Znn);
  test_traj.set_terminal_cost_gradient_state(&c);
  if (term_dim > 0) {
    test_traj.set_terminal_constraint_jacobian_state(&Inn);
    test_traj.set_terminal_constraint_jacobian_terminal_projection(&Inn);
    test_traj.set_terminal_constraint_affine_term(&Zn);
  }

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

  Eigen::MatrixXd Vxx, Vzx, Vzz, V11;
  Eigen::VectorXd Vx1, Vz1;

  for (int t = 0; t < T; ++t) {
    test_traj.get_state_dependencies_initial_state_projection(t, Tx);
    test_traj.get_state_dependencies_terminal_state_projection(t, Tz);
    test_traj.get_state_dependencies_affine_term(t, T1);


    Eigen::VectorXd temp = Tx * (-x0) + T1;
    if (term_dim > 0) {
      temp += Tz * (-xT);
    }
    x = temp;
    std::cout << "t: " << t + 1 << std::endl;
    std::cout << "x: " << x.transpose() << std::endl;
    if (t < T - 1) {
      test_traj.get_control_dependencies_initial_state_projection(t, Ux);
      test_traj.get_control_dependencies_terminal_state_projection(t, Uz);
      test_traj.get_control_dependencies_affine_term(t, U1);
      Eigen::VectorXd temp2 = Ux * (-x0) + U1;
      if (term_dim > 0) {
        temp2 += Uz * (-xT);
      }
      std::cout << "u: " << temp2.transpose() << std::endl;
    }

    test_traj.get_dynamics_mult_initial_state_feedback_term(t, Tx);
    test_traj.get_dynamics_mult_terminal_state_feedback_term(t, Tz);
    test_traj.get_dynamics_mult_feedforward_term(t, T1);

    Eigen::VectorXd lam = Tx * (-x0) + T1;
    if (term_dim > 0) {
      lam += Tz * (-xT);
    }
    std::cout << "Lam " << t << " = " << lam.transpose() << std::endl;

    Vxx = test_traj.cost_to_go_hessians_state_state[t];
    Vzx = test_traj.cost_to_go_hessians_terminal_state[t];
    Vzz = test_traj.cost_to_go_hessians_terminal_terminal[t];
    Vx1 = test_traj.cost_to_go_gradients_state[t];
    Vz1 = test_traj.cost_to_go_gradients_terminal[t];
    V11 = test_traj.cost_to_go_offsets[t];
    Eigen::VectorXd lam2 = Vxx * x + Vx1;
    if (term_dim > 0) {
      lam2 += Vzx.transpose() * (-xT);
    }
    if (t < T - 1) {
      test_traj.get_running_constraint_mult_initial_state_feedback_term(t, Ux);
      test_traj.get_running_constraint_mult_terminal_state_feedback_term(t, Uz);
      test_traj.get_running_constraint_mult_feedforward_term(t, U1);

      Eigen::VectorXd mu = Ux * (-x0) + U1;
      if (term_dim > 0) {
        mu += Uz * (-xT);
      }
      std::cout << "Mu " << t << " = " << mu.transpose() << std::endl;
    } else {
      test_traj.get_terminal_constraint_mult_initial_state_feedback_term(Ux);
      test_traj.get_terminal_constraint_mult_terminal_state_feedback_term(Uz);
      test_traj.get_terminal_constraint_mult_feedforward_term(U1);
      Eigen::VectorXd mu = Ux * (-x0) + U1;
      if (term_dim > 0){
        mu += Uz * (-xT);
      }
      std::cout << "Mu " << t << " = " << mu.transpose() << std::endl;
    }
  }
  std::cout << "Serial time: " << t1 - t0 << std::endl;

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

TEST(TestTrajectory, LQRSplit) {
  const int TT = 9;
  const int T = 5;
  Eigen::MatrixXd A(n, n);
  Eigen::MatrixXd B(n, m);
  Eigen::VectorXd c(n);
  Eigen::VectorXd r(m);
  A << 1.0, 1.0, 0.0, 1.0;
  B << 0.0, 0.1;
  c << 1.0, 1.0;
  r << 1.0;
  Eigen::VectorXd x0(n);
  x0 << 5.0, 1.0;

  std::function<void(const Eigen::VectorXd *, const Eigen::VectorXd *, Eigen::VectorXd &)> dynamics_f = dynamics;
  std::function<void(const Eigen::VectorXd *, const Eigen::VectorXd *, Eigen::VectorXd &)>
      running_constraint_f = running_constraint;
  std::function<void(const Eigen::VectorXd *, Eigen::VectorXd &)> final_constraint_f = final_constraint;
  std::function<void(const Eigen::VectorXd *, Eigen::VectorXd &)> initial_constraint_f = initial_constraint;
  std::function<double(const Eigen::VectorXd *, const Eigen::VectorXd *)> running_cost_f = running_cost;
  std::function<double(const Eigen::VectorXd *)> terminal_cost_f = terminal_cost;

  trajectory::Trajectory test_traj1(T,
                                    n,
                                    m,
                                    1,
                                    &dynamics_f,
                                    &running_constraint_f,
                                    &final_constraint_f,
                                    &initial_constraint_f,
                                    &running_cost_f,
                                    &terminal_cost_f);

  trajectory::Trajectory test_traj2(T,
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
  test_traj1.set_initial_constraint_jacobian_state(&Inn);
  test_traj1.set_initial_constraint_affine_term(&x0);
  test_traj2.set_initial_constraint_jacobian_state(&Inn);
  test_traj2.set_initial_constraint_affine_term(&x0);

  test_traj1.set_initial_constraint_dimension(n);
  test_traj1.set_terminal_constraint_dimension(n);
  test_traj2.set_initial_constraint_dimension(n);
  test_traj2.set_terminal_constraint_dimension(n);

//  Eigen::VectorXd xT(n);
//  xT << 44.0, -4.0;

  for (unsigned int t = 0; t < T - 1; ++t) {
    test_traj1.set_dynamics_jacobian_state(t, &A);
    test_traj1.set_dynamics_jacobian_control(t, &B);
    test_traj1.set_dynamics_affine_term(t, &c);
    test_traj1.set_num_active_constraints(t, 0);

    test_traj1.set_hamiltonian_hessians_state_state(t, &Inn);
    test_traj1.set_hamiltonian_hessians_control_control(t, &Imm);
    test_traj1.set_hamiltonian_gradients_state(t, &c);
    test_traj1.set_hamiltonian_gradients_control(t, &r);

    test_traj2.set_dynamics_jacobian_state(t, &A);
    test_traj2.set_dynamics_jacobian_control(t, &B);
    test_traj2.set_dynamics_affine_term(t, &c);
    test_traj2.set_num_active_constraints(t, 0);

    test_traj2.set_hamiltonian_hessians_state_state(t, &Inn);
    test_traj2.set_hamiltonian_hessians_control_control(t, &Imm);
    test_traj2.set_hamiltonian_gradients_state(t, &c);
    test_traj2.set_hamiltonian_gradients_control(t, &r);
  }
  test_traj2.set_terminal_cost_hessians_state_state(&Inn);
  test_traj2.set_terminal_cost_gradient_state(&c);

  test_traj1.set_terminal_constraint_jacobian_state(&Inn);
  test_traj1.set_terminal_constraint_jacobian_terminal_projection(&Inn);
  test_traj1.set_terminal_constraint_affine_term(&Zn);

  test_traj2.set_terminal_constraint_jacobian_state(&Inn);
  test_traj2.set_terminal_constraint_jacobian_terminal_projection(&Inn);
  test_traj2.set_terminal_constraint_affine_term(&Zn);

  test_traj1.compute_feedback_policies();
  test_traj1.compute_state_control_dependencies();
  test_traj1.compute_multipliers();

  test_traj2.compute_feedback_policies();
  test_traj2.compute_state_control_dependencies();
  test_traj2.compute_multipliers();

  Eigen::MatrixXd Traj1X1(n, n);
  Eigen::MatrixXd Traj1X2(n, n);
  Eigen::VectorXd Traj11(n);

  Eigen::MatrixXd Traj2X1(n, n);
  Eigen::MatrixXd Traj2X2(n, n);
  Eigen::VectorXd Traj21(n);

  Eigen::MatrixXd Ux;
  Eigen::MatrixXd Uz;
  Eigen::VectorXd U1;

  test_traj1.get_terminal_constraint_mult_initial_state_feedback_term(Ux);
  test_traj1.get_terminal_constraint_mult_terminal_state_feedback_term(Uz);
  test_traj1.get_terminal_constraint_mult_feedforward_term(U1);

  Traj11 = U1 + Ux * (-x0);
  Traj1X1 = Uz;

  test_traj2.get_dynamics_mult_initial_state_feedback_term(0, Ux);
  test_traj2.get_dynamics_mult_terminal_state_feedback_term(0, Uz);
  test_traj2.get_dynamics_mult_feedforward_term(0, U1);

  Traj11 = (Traj11 + U1).eval();
  Traj1X1 = (Traj1X1 + Ux).eval();
  Traj1X2 = Uz;

  test_traj2.get_terminal_constraint_mult_initial_state_feedback_term(Traj2X1);
  test_traj2.get_terminal_constraint_mult_terminal_state_feedback_term(Traj2X2);
  test_traj2.get_terminal_constraint_mult_feedforward_term(Traj21);

  Eigen::MatrixXd Mat(2 * n, 2 * n);
  Eigen::VectorXd vec(2 * n);
  Mat << Traj1X1, Traj1X2, Traj2X1, Traj2X2;
  vec << Traj11, Traj21;

  Eigen::PartialPivLU<Eigen::MatrixXd> decomp(Mat);
  Eigen::VectorXd soln = -decomp.solve(vec);
  std::cout << soln.transpose() << std::endl;
}

TEST(TestTrajectory, LQRSplit2) {
  const int TT = 100;
  const int T = 5;
  const int num_trajs = 4;
  Eigen::MatrixXd A(n, n);
  Eigen::MatrixXd B(n, m);
  Eigen::VectorXd c(n);
  Eigen::VectorXd r(m);
  A << 1.0, 1.0, 0.0, 1.0;
  B << 0.0, 0.1;
  c << 1.0, 1.0;
  r << 1.0;
  Eigen::VectorXd x0(n);
  x0 << 5.0, 1.0;

  const Eigen::MatrixXd Inn = Eigen::MatrixXd::Identity(n, n);
  const Eigen::MatrixXd Imm = Eigen::MatrixXd::Identity(m, m);
  const Eigen::VectorXd Zn = Eigen::VectorXd::Zero(n);

  std::vector<Eigen::MatrixXd> AA(num_trajs, A);
  std::vector<Eigen::MatrixXd> BB(num_trajs, B);
  std::vector<Eigen::VectorXd> cc(num_trajs, c);
  std::vector<Eigen::VectorXd> rr(num_trajs, r);

  std::function<void(const Eigen::VectorXd *, const Eigen::VectorXd *, Eigen::VectorXd &)> dynamics_f = dynamics;
  std::function<void(const Eigen::VectorXd *, const Eigen::VectorXd *, Eigen::VectorXd &)>
      running_constraint_f = running_constraint;
  std::function<void(const Eigen::VectorXd *, Eigen::VectorXd &)> final_constraint_f = final_constraint;
  std::function<void(const Eigen::VectorXd *, Eigen::VectorXd &)> initial_constraint_f = initial_constraint;
  std::function<double(const Eigen::VectorXd *, const Eigen::VectorXd *)> running_cost_f = running_cost;
  std::function<double(const Eigen::VectorXd *)> terminal_cost_f = terminal_cost;

  parent_trajectory::ParentTrajectory parent_traj = parent_trajectory::ParentTrajectory(TT,
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

  for (int i = 0; i < num_trajs; ++i) {
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
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}