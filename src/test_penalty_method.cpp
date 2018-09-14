//
// Created by Forrest Laine on 8/28/18.
//

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
#include "endpoint_constraint.h"
#include "equality_constrained_running_constraint.h"
#include "equality_constrained_endpoint_constraint.h"
#include "running_cost.h"
#include "terminal_cost.h"

#if defined(_OPENMP)
#include <omp.h>
#include "matplotlibcpp.h"
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

namespace plt = matplotlibcpp;

void RunPenaltyTest() {
  const int T = 100;
  const int n = 9;
  const int m = 2;
  const double dt = 0.01;
  const double r = 1.0;
  const double alpha = 1.0;
  const int runs = 10;

  Eigen::MatrixXd A = Eigen::MatrixXd::Identity(n, n);
  for (int i = 0; i < n - 1; ++i) {
    A(i, i + 1) = dt;
  }
  Eigen::MatrixXd B = Eigen::MatrixXd::Zero(n, m);
  B(n-1, 0) = dt;

  std::cout << A << std::endl;
  std::cout << B << std::endl;

  std::vector<Eigen::MatrixXd> QNs(runs, Eigen::MatrixXd::Identity(n, n));
  double factor = 1.0;
  for (int i = 0; i < runs; ++i) {
    QNs[i] *= factor;
    factor *= 1010.0;
  }

  QNs[runs-2] = std::numeric_limits<double>::max() * Eigen::MatrixXd::Identity(n,n);

  QNs[runs-1] *= 0.0;

  Eigen::MatrixXd Q = 0.0 * Eigen::MatrixXd::Identity(n, n);
  const Eigen::MatrixXd R = alpha*Eigen::MatrixXd::Identity(m, m);
  const Eigen::MatrixXd I = Eigen::MatrixXd::Identity(n, n);
  Eigen::VectorXd q = 0.0 * Eigen::VectorXd::Constant(n, 2.0);
  Eigen::VectorXd x0 = Eigen::VectorXd::Zero(n);
  Eigen::VectorXd xT = Eigen::VectorXd::Zero(n);
  x0(0) = 1.5;
  x0(1) = 5.0;

  std::function<double(const Eigen::VectorXd *, const Eigen::VectorXd *)>
      running_cost_f = [R, Q, q](const Eigen::VectorXd *x, const Eigen::VectorXd *u) {
    Eigen::MatrixXd val =
        0.5 * (*u).transpose() * R * (*u) + 0.5 * (*x).transpose() * Q * (*x) + q.transpose() * (*x);
    return val(0, 0);
  };

  std::function<void(const Eigen::VectorXd *, const Eigen::VectorXd *, Eigen::VectorXd &)>
      running_grad_x_f = [Q, q](const Eigen::VectorXd *x, const Eigen::VectorXd *, Eigen::VectorXd &gx) {
    gx = Q * (*x) + q;
  };

  std::function<void(const Eigen::VectorXd *, const Eigen::VectorXd *, Eigen::VectorXd &)>
      running_grad_u_f = [R](const Eigen::VectorXd *, const Eigen::VectorXd *u, Eigen::VectorXd &gu) {
    gu = R * (*u);
  };

  std::function<void(const Eigen::VectorXd *, const Eigen::VectorXd *, Eigen::MatrixXd &)>
      running_hess_xx_f = [Q](const Eigen::VectorXd *, const Eigen::VectorXd *, Eigen::MatrixXd &Hxx) {
    Hxx = Q;
  };

  std::function<void(const Eigen::VectorXd *, const Eigen::VectorXd *, Eigen::MatrixXd &)>
      running_hess_ux_f = [](const Eigen::VectorXd *, const Eigen::VectorXd *, Eigen::MatrixXd &Hux) {
    Hux.setZero();
  };

  std::function<void(const Eigen::VectorXd *, const Eigen::VectorXd *, Eigen::MatrixXd &)>
      running_hess_uu_f = [R](const Eigen::VectorXd *, const Eigen::VectorXd *, Eigen::MatrixXd &Huu) {
    Huu = R;
  };

  std::function<void(const Eigen::VectorXd *, const Eigen::VectorXd *, Eigen::VectorXd &)>
      dynamics_f = [A, B, n](const Eigen::VectorXd *x, const Eigen::VectorXd *u, Eigen::VectorXd &xx) {
    xx = A * (*x) + B * (*u);
//    xx(1) -= (*x)(0) / (10+(*x)(2)*(*x)(2));
//    xx(3) -= (*x)(2) / (10+(*x)(1)*(*x)(1));
//    xx(1) -= 0.05;
//    xx(3) += 0.05;
  };

  std::function<void(const Eigen::VectorXd *, const Eigen::VectorXd *, Eigen::MatrixXd &)>
      dynamics_jac_x_f = [A](const Eigen::VectorXd *, const Eigen::VectorXd *, Eigen::MatrixXd &Ax) {
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
      running_constraint_f = [r](const Eigen::VectorXd *x, const Eigen::VectorXd *, int, Eigen::VectorXd &c) {
    c.setZero();
    c(0) = r * r - ((*x)(0) * (*x)(0) + (*x)(2) * (*x)(2));
  };

  std::function<void(const Eigen::VectorXd *, const Eigen::VectorXd *, int, Eigen::MatrixXd &)>
      running_constraint_jac_x_f = [](const Eigen::VectorXd *x, const Eigen::VectorXd *, int, Eigen::MatrixXd &Jx) {
    Jx.setZero();
    Jx(0) = -2.0 * (*x)(0);
    Jx(2) = -2.0 * (*x)(2);
  };

  std::function<void(const Eigen::VectorXd *, const Eigen::VectorXd *, int, Eigen::MatrixXd &)>
      running_constraint_jac_u_f = [](const Eigen::VectorXd *, const Eigen::VectorXd *, int, Eigen::MatrixXd &Ju) {
    Ju.setZero();
  };

//  dynamics::Dynamics dynamics_obj = dynamics::Dynamics(&dynamics_f, &dynamics_jac_x_f, &dynamics_jac_u_f);
  dynamics::Dynamics dynamics_obj = dynamics::Dynamics(&dynamics_f);
  running_cost::RunningCost running_cost_obj = running_cost::RunningCost(&running_cost_f,
                                                                         &running_grad_x_f,
                                                                         &running_grad_u_f,
                                                                         &running_hess_xx_f,
                                                                         &running_hess_ux_f,
                                                                         &running_hess_uu_f);



  endpoint_constraint::EndPointConstraint
      terminal_constraint_obj = endpoint_constraint::EndPointConstraint(&terminal_constraint_f,
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




  for (int i = 0; i < runs; ++i) {

    std::function<double(const Eigen::VectorXd *)> terminal_cost_f = [QNs, i](const Eigen::VectorXd *x) {
      Eigen::MatrixXd val = 0.5 * (*x).transpose() * QNs[i] * (*x);
      return val(0, 0);
    };

    std::function<void(const Eigen::VectorXd *, Eigen::VectorXd &)>
        terminal_grad_f = [QNs, i](const Eigen::VectorXd *x, Eigen::VectorXd &g) {
      g = QNs[i] * (*x);
    };

    std::function<void(const Eigen::VectorXd *, Eigen::MatrixXd &)>
        terminal_hess_f = [QNs, i](const Eigen::VectorXd *, Eigen::MatrixXd &H) {
      H = QNs[i];
    };

    terminal_cost::TerminalCost terminal_cost_obj = terminal_cost::TerminalCost(&terminal_cost_f,
                                                                                &terminal_grad_f,
                                                                                &terminal_hess_f);



    int end_constraint_dim = ( i == runs-1) ? n : 0;

    equality_constrained_endpoint_constraint::EqualityConstrainedEndPointConstraint
        ec_terminal_constraint_obj =
        equality_constrained_endpoint_constraint::EqualityConstrainedEndPointConstraint(&terminal_constraint_f,
                                                                                        &terminal_constraint_jac_f,
                                                                                        end_constraint_dim,
                                                                                        false);
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


    std::vector<double> x(T), y(T), xx(T), yy(T);

    test_traj.set_initial_point(&x0);
    test_traj.execute_current_control_policy(0.0);
    test_traj.populate_derivative_terms();

    if (i==(runs-1)) {
      test_traj.populate_bandsolve_terms();
      test_traj.execute_sanity_policy();
      for (int t = 0; t < T; ++t) {
        y[t] = test_traj.sanity_states[t](0);
        x[t] = t*dt;
      }
      plt::plot(x,y,"r-.");
    }

    std::cout<<"Finished with bandsolve:test"<<std::endl;
//    test_traj.execute_current_control_policy(0.0);
//    test_traj.populate_derivative_terms();
    test_traj.compute_feedback_policies();
    test_traj.compute_state_control_dependencies();
    test_traj.compute_multipliers();
    test_traj.set_open_loop_traj();
    test_traj.set_lq_multipliers();
    test_traj.execute_current_control_policy(100000.0);
//    test_traj.execute_open_loop_policy();
    for (int t = 0; t < T; ++t) {
      std::cout << test_traj.current_points[t].transpose() << std::endl;
      y[t] = test_traj.current_points[t](0);
      x[t] = t*dt;
    }
//    char color_buffer[649];
//    double r,g,b;
//    r = 0.0;
//    g = 1.0 - 1.0 * i / runs;
//    b = 0.0 + 1.0 * i /runs;
//    sprintf(color_buffer, "%f,%f,%f", r, g, b);
//    sprintf(color_buffer, "C%d-", i);
    std::string color;
    if (i==0) {
      color = "b-";
    } else if (i==1) {
      color = "g-";
    }else if (i==2) {
      color = "r-";
    }else if (i==3) {
      color = "c-";
    }else if (i==4) {
      color = "m-";
    }else if (i==5) {
      color = "y-";
    }else if (i==6) {
      color = "k-";
    }else if (i==7) {
      color = "b-.";
    }else if (i==8) {
      color = "g-.";
    }else if (i==9) {
      color = "k-.";
    }


    plt::plot(x, y , color);




//  test_traj.populate_derivative_terms();
//  test_traj.compute_feedback_policies();
//  test_traj.compute_state_control_dependencies();
//  test_traj.compute_multipliers();
//  test_traj.set_open_loop_traj();
//  test_traj.set_lq_multipliers();
//  test_traj.execute_current_open_loop_controls();

//  plt::figure();
//  double err;
//  for (int t = 0; t < T; ++t) {
//    xx[t] = test_traj.current_points[t](0);
//    yy[t] = test_traj.current_points[t](3);
//    if (t>=95 and t < T-1) {
//      err = test_traj.open_loop_states[t](0) * test_traj.running_constraint_jacobians_state[t](0) + test_traj.open_loop_states[t](3) * test_traj.running_constraint_jacobians_state[t](3) + test_traj.running_constraint_affine_terms[t](0);
//      std::cout<<"err at t="<<t<<": "<<err<<std::endl;
//    }
//  }
//  plt::plot(x, y, "k-");
//  plt::plot(xx, yy, "r-");
  }
  plt::show();

}

int main(int /*argc*/, char ** /*argv*/) {
  RunPenaltyTest();
  return 0;
}
