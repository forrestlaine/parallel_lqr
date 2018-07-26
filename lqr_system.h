//
// Created by Forrest Laine on 7/26/18.
//

#ifndef MOTORBOAT_LQR_SYSTEM_H
#define MOTORBOAT_LQR_SYSTEM_H

#include "Eigen3/Eigen/Dense"
#include <vector>

namespace lqr_system {

class LQRSystem {
 public:
  LQRSystem(const Eigen::MatrixXd* Q,
            const Eigen::MatrixXd* S,
            const Eigen::MatrixXd* R,
            const Eigen::VectorXd* q,
            const Eigen::VectorXd* r,
            const Eigen::MatrixXd* A,
            const Eigen::MatrixXd* B,
            const Eigen::VectorXd* c,
            const Eigen::VectorXd* x0,
            const Eigen::VectorXd* xT,
            int T,
            int n,
            int m,
            bool random_constraints):
    Q(*Q),
    S(*S),
    R(*R),
    q(*q),
    r(*r),
    A(*A),
    B(*B),
    c(*c),
    I(Eigen::MatrixXd::Identity(Q->rows(), Q->cols())),
    x0(*x0),
    xT(*xT),
    T(T),
    n(n),
    m(m),
    random_constraints(random_constraints) {};

  double terminal_cost(const Eigen::VectorXd *x);

  void terminal_grad(const Eigen::VectorXd *x, Eigen::VectorXd &g);

  void terminal_hess(const Eigen::VectorXd *x, Eigen::MatrixXd &H);


  double running_cost(const Eigen::VectorXd *x, const Eigen::VectorXd *u);

  void running_grad_x(const Eigen::VectorXd *x, const Eigen::VectorXd *u, Eigen::VectorXd &gx);

  void running_grad_u(const Eigen::VectorXd *x, const Eigen::VectorXd *u, Eigen::VectorXd &gu);

  void running_hess_xx(const Eigen::VectorXd *x, const Eigen::VectorXd *u, Eigen::MatrixXd &Hxx);

  void running_hess_ux(const Eigen::VectorXd *x, const Eigen::VectorXd *u, Eigen::MatrixXd &Hux);

  void running_hess_uu(const Eigen::VectorXd *x, const Eigen::VectorXd *u, Eigen::MatrixXd &Huu);


  void dynamics(const Eigen::VectorXd *x, const Eigen::VectorXd *u, Eigen::VectorXd &xx);

  void dynamics_jac_x(const Eigen::VectorXd *x, const Eigen::VectorXd *u, Eigen::MatrixXd &Ax);

  void dynamics_jac_u(const Eigen::VectorXd *x, const Eigen::VectorXd *u, Eigen::MatrixXd &Au);


  void initial_constraint(const Eigen::VectorXd *x, Eigen::VectorXd &c);

  void initial_constraint_jac(const Eigen::VectorXd *x, Eigen::MatrixXd &J);


  void terminal_constraint(const Eigen::VectorXd *x, Eigen::VectorXd &c);

  void terminal_constraint_jac(const Eigen::VectorXd *x, Eigen::MatrixXd &J);


  void running_constraint(const Eigen::VectorXd *x, const Eigen::VectorXd *u, int t, Eigen::VectorXd &c);

  void running_constraint_jac_x(const Eigen::VectorXd *x, const Eigen::VectorXd *u, int t, Eigen::MatrixXd &Jx);

  void running_constraint_jac_u(const Eigen::VectorXd *x, const Eigen::VectorXd *u, int t, Eigen::MatrixXd &Ju);

 private:
  const Eigen::MatrixXd Q, S, R, A, B, I;
  const Eigen::VectorXd q, r, c, x0, xT;

  const int T, n, m;

  const bool random_constraints;

};

} //namespace lqr_system

#endif //MOTORBOAT_LQR_SYSTEM_H
