//
// Created by Forrest Laine on 7/26/18.
//

#include "lqr_system.h"

namespace lqr_system {

double LQRSystem::terminal_cost(const Eigen::VectorXd *x) {
  Eigen::MatrixXd val = 0.5 * (*x).transpose() * this->Q * (*x) + (*x).transpose() * this->q;
  return val(0, 0);
}

void LQRSystem::terminal_grad(const Eigen::VectorXd *x, Eigen::VectorXd &g) {
  g = this->Q * (*x) + this->q;
}

void LQRSystem::terminal_hess(const Eigen::VectorXd *x, Eigen::MatrixXd &H) {
  H = this->Q;
}

double LQRSystem::running_cost(const Eigen::VectorXd *x, const Eigen::VectorXd *u) {
  Eigen::MatrixXd val =
      0.5 * ((*x).transpose() * this->Q * (*x) + (*u).transpose() * this->R * (*u)) + (*u).transpose() * this->S * (*x)
          + (*x).transpose() * this->q + (*u).transpose() * this->r;
}

void LQRSystem::running_grad_x(const Eigen::VectorXd *x, const Eigen::VectorXd *u, Eigen::VectorXd &gx) {
  gx = this->Q * (*x) + this->S.transpose() * (*u) + this->q;
}

void LQRSystem::running_grad_u(const Eigen::VectorXd *x, const Eigen::VectorXd *u, Eigen::VectorXd &gu) {
  gu = this->R * (*u) + this->S * (*x) + this->r;
}

void LQRSystem::running_hess_xx(const Eigen::VectorXd *x, const Eigen::VectorXd *u, Eigen::MatrixXd &Hxx) {
  Hxx = this->Q;
}

void LQRSystem::running_hess_ux(const Eigen::VectorXd *x, const Eigen::VectorXd *u, Eigen::MatrixXd &Hux) {
  Hux = this->S;
}

void LQRSystem::running_hess_uu(const Eigen::VectorXd *x, const Eigen::VectorXd *u, Eigen::MatrixXd &Huu) {
  Huu = this->R;
}

void LQRSystem::dynamics(const Eigen::VectorXd *x, const Eigen::VectorXd *u, Eigen::VectorXd &xx) {
  xx = this->A * (*x) + this->B * (*u) + this->c;
}

void LQRSystem::dynamics_jac_x(const Eigen::VectorXd *x, const Eigen::VectorXd *u, Eigen::MatrixXd &Ax) {
  Ax = this->A;
}

void LQRSystem::dynamics_jac_u(const Eigen::VectorXd *x, const Eigen::VectorXd *u, Eigen::MatrixXd &Au) {
  Au = this->B:
}

void LQRSystem::initial_constraint(const Eigen::VectorXd *x, Eigen::VectorXd &c) {
  c = (*x) - this->x0;
}

void LQRSystem::initial_constraint_jac(const Eigen::VectorXd *x, Eigen::MatrixXd &J) {
  J = this->I;
}

void LQRSystem::terminal_constraint(const Eigen::VectorXd *x, Eigen::VectorXd &c) {
  c = (*x) - this->xT;
}

void LQRSystem::terminal_constraint_jac(const Eigen::VectorXd *x, Eigen::MatrixXd &J) {
  J = this->I;
}

void LQRSystem::running_constraint(const Eigen::VectorXd *x, const Eigen::VectorXd *u, int t, Eigen::VectorXd &c) {}

void LQRSystem::running_constraint_jac_x(const Eigen::VectorXd *x, const Eigen::VectorXd *u, int t, Eigen::MatrixXd &Jx) {}

void LQRSystem::running_constraint_jac_u(const Eigen::VectorXd *x, const Eigen::VectorXd *u, int t, Eigen::MatrixXd &Ju) {}

} // namespace lqr_system