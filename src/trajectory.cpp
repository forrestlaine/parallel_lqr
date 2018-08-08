//
// Created by Forrest Laine on 6/19/18.
//

#include "trajectory.h"

#include <iostream>
#include <algorithm>
#include <vector>

namespace trajectory {

extern "C" void dgbsv_(int *N,
                       int *KL,
                       int *KU,
                       int *NRHS,
                       double *AB,
                       int *LDAB,
                       int *IPIV,
                       double *B,
                       int *LDB,
                       int *INFO);

void Trajectory::populate_derivative_terms() {
  // Evaluate derivatives, numerically or analytically, and populate corresponding terms.
  Eigen::VectorXd temp_constraint;
  Eigen::MatrixXd temp_jacobian_1(this->running_constraint_dimension, this->state_dimension);
  Eigen::MatrixXd temp_jacobian_2(this->running_constraint_dimension, this->control_dimension);

  this->initial_constraint.eval_constraint_jacobian(&this->current_points[0], this->initial_constraint_jacobian_state);
  this->initial_constraint.eval_constraint(&this->current_points[0], this->initial_constraint_affine_term);

  for (unsigned int t = 0; t < this->trajectory_length - 1; ++t) {
    this->dynamics.eval_dynamics_jacobian_state(&this->current_points[t],
                                                &this->current_controls[t],
                                                this->dynamics_jacobians_state[t]);
    this->dynamics.eval_dynamics_jacobian_control(&this->current_points[t],
                                                  &this->current_controls[t],
                                                  this->dynamics_jacobians_control[t]);
    this->dynamics.eval_dynamics(&this->current_points[t], &this->current_controls[t], this->dynamics_affine_terms[t]);

    if (this->running_constraint_dimension > 0) {
      this->running_constraint.eval_constraint(&this->current_points[t],
                                               &this->current_controls[t],
                                               t,
                                               temp_constraint);
      this->num_active_constraints[t] = (unsigned int) this->running_constraint.eval_active_indices(&temp_constraint,
                                                                                                    this->active_running_constraints[t]);
      this->running_constraint.eval_constraint_jacobian_state(&this->current_points[t],
                                                              &this->current_controls[t],
                                                              t,
                                                              temp_jacobian_1);
      this->running_constraint.eval_constraint_jacobian_control(&this->current_points[t],
                                                                &this->current_controls[t],
                                                                t,
                                                                temp_jacobian_2);

      for (int k = 0, d = 0; k < this->running_constraint.get_constraint_dimension(); ++k) {
        if (this->active_running_constraints[t](k)) {
          this->running_constraint_affine_terms[t](d) = temp_constraint(k);
          this->running_constraint_jacobians_state[t].row(d) = temp_jacobian_1.row(k);
          this->running_constraint_jacobians_control[t].row(d) = temp_jacobian_2.row(k);
          ++d;
        }
      }
    } else {
      this->num_active_constraints[t] = 0;
    }

    this->running_cost.eval_cost_hessian_state_state(&this->current_points[t],
                                                     &this->current_controls[t],
                                                     this->hamiltonian_hessians_state_state[t]);
    this->running_cost.eval_cost_hessian_control_state(&this->current_points[t],
                                                       &this->current_controls[t],
                                                       this->hamiltonian_hessians_control_state[t]);
    this->running_cost.eval_cost_hessian_control_control(&this->current_points[t],
                                                         &this->current_controls[t],
                                                         this->hamiltonian_hessians_control_control[t]);
    this->running_cost.eval_cost_gradient_state(&this->current_points[t],
                                                &this->current_controls[t],
                                                this->hamiltonian_gradients_state[t]);
    this->running_cost.eval_cost_gradient_control(&this->current_points[t],
                                                  &this->current_controls[t],
                                                  this->hamiltonian_gradients_control[t]);

  }
  temp_constraint.resize(this->terminal_constraint.get_constraint_dimension());
  int t = this->trajectory_length - 1;
  if (this->terminal_constraint.get_constraint_dimension() > 0) {
    temp_jacobian_1.resize(this->terminal_constraint.get_constraint_dimension(), this->state_dimension);
    if (not this->terminal_constraint.is_implicit()) {
      this->terminal_constraint.eval_constraint(&this->current_points[t], temp_constraint);
      this->terminal_constraint.eval_constraint_jacobian(&this->current_points[t], temp_jacobian_1);
      this->terminal_constraint_dimension =
          (unsigned int) this->terminal_constraint.eval_active_indices(&temp_constraint,
                                                                       this->active_terminal_constraints);
    }

    const Eigen::MatrixXd I = Eigen::MatrixXd::Identity(this->terminal_constraint.get_constraint_dimension(),
                                                        this->terminal_constraint.get_constraint_dimension());
    for (int k = 0, d = 0; k < this->terminal_constraint.get_constraint_dimension(); ++k) {
      if (this->terminal_constraint.is_implicit()) {
        this->terminal_constraint_jacobian_state.row(d) = I.row(k);
        this->terminal_constraint_jacobian_terminal_projection.row(d) = I.row(k);
        ++d;
      } else {
        if (this->active_terminal_constraints(k)) {
          this->terminal_constraint_affine_term(d) = temp_constraint(k);
          this->terminal_constraint_jacobian_state.row(d) = temp_jacobian_1.row(k);
          ++d;
        }
      }
    }
  }
  this->terminal_cost.eval_cost_hessian_state_state(&this->current_points[t], this->terminal_cost_hessians_state_state);
  this->terminal_cost.eval_cost_gradient_state(&this->current_points[t], this->terminal_cost_gradient_state);
}

void Trajectory::populate_bandsolve_terms() {
  unsigned int max_constraint_size = (unsigned int) this->active_terminal_constraints.count();
  long total_active_constraints = this->active_terminal_constraints.count();
  for (int t = 0; t < this->trajectory_length - 1; ++t) {
    max_constraint_size = std::max(max_constraint_size, this->num_active_constraints[t]);
    total_active_constraints += this->num_active_constraints[t];
  }
  this->soln_size = (int)
      (2 * this->trajectory_length * this->state_dimension + (this->trajectory_length - 1) * this->control_dimension
          + total_active_constraints);

  long primal_size =
      this->trajectory_length * this->state_dimension + (this->trajectory_length - 1) * this->control_dimension;
  long dual_size_1 = this->trajectory_length * this->state_dimension;
  long dual_size_2 = total_active_constraints;

  // todo: this might be wrong
  this->half_bandwidth = (int) std::max(2 * this->state_dimension + this->control_dimension,
                                        this->state_dimension + this->control_dimension + max_constraint_size) - 1;
  this->A.resize(this->soln_size, this->soln_size);
  this->B.resize((unsigned long) this->soln_size);

  int n = this->state_dimension;
  int m = this->control_dimension;

  Eigen::MatrixXd K, D1, D2, Z1, Z2, I;
  Eigen::VectorXd k, d1, d2;
  Z1 = Eigen::MatrixXd::Zero(dual_size_1, dual_size_1 + dual_size_2);
  Z2 = Eigen::MatrixXd::Zero(dual_size_2, dual_size_1 + dual_size_2);
  D1 = Eigen::MatrixXd::Zero(dual_size_1, primal_size);
  D2 = Eigen::MatrixXd::Zero(dual_size_2, primal_size);
  d1 = Eigen::VectorXd::Zero(dual_size_1);
  d2 = Eigen::VectorXd::Zero(dual_size_2);
  k = Eigen::VectorXd::Zero(primal_size);
  K = Eigen::MatrixXd::Zero(primal_size, primal_size);
  I = Eigen::MatrixXd::Identity(n, n);

  int primal_offset = 0;
  int dual_offset_1 = 0;
  int dual_offset_2 = 0;
  d1.segment(dual_offset_1, n) = -this->initial_constraint_affine_term;
  std::cout << "Made it here" << std::endl;

  for (int t = 0; t < this->trajectory_length - 1; ++t) {
    K.block(primal_offset, primal_offset, n, n) = this->hamiltonian_hessians_state_state[t];
    K.block(primal_offset + n, primal_offset, m, n) = this->hamiltonian_hessians_control_state[t];
    K.block(primal_offset, primal_offset + n, n, m) = this->hamiltonian_hessians_control_state[t].transpose();
    K.block(primal_offset + n, primal_offset + n, m, m) = this->hamiltonian_hessians_control_control[t];
    k.segment(primal_offset, n) = -this->hamiltonian_gradients_state[t];
    k.segment(primal_offset + n, m) = -this->hamiltonian_gradients_control[t];

    D1.block(dual_offset_1, primal_offset, n, n) = I;
    D1.block(dual_offset_1 + n, primal_offset, n, n) = -this->dynamics_jacobians_state[t];
    D1.block(dual_offset_1 + n, primal_offset + n, n, m) = -this->dynamics_jacobians_control[t];
    d1.segment(dual_offset_1 + n, n) = this->dynamics_affine_terms[t];
    dual_offset_1 += n;

    int l = this->num_active_constraints[t];
    D2.block(dual_offset_2, primal_offset, l, n) = this->running_constraint_jacobians_state[t].topRows(l);
    D2.block(dual_offset_2, primal_offset + n, l, m) = this->running_constraint_jacobians_control[t].topRows(l);
    d2.segment(dual_offset_2, l) = -this->running_constraint_affine_terms[t].head(l);
    dual_offset_2 += l;

    primal_offset += (n + m);
  }
  K.block(primal_offset, primal_offset, n, n) = this->terminal_cost_hessians_state_state;
  k.segment(primal_offset, n) = -this->terminal_cost_gradient_state;

  D1.block(dual_offset_1, primal_offset, n, n) = I;

  long l = this->active_terminal_constraints.count();
  D2.block(dual_offset_2, primal_offset, l, n) = this->terminal_constraint_jacobian_state.topRows(l);
  d2.segment(dual_offset_2, l) = -this->terminal_constraint_affine_term.head(l);

  Eigen::MatrixXd tempA(this->soln_size, this->soln_size);
  Eigen::VectorXd tempB(this->soln_size);

  tempA << K, D1.transpose(), D2.transpose(), D1, Z1, D2, Z2;
  tempB << k, d1, d2;

  std::cout << "Made it here" << std::endl;

  // Setup permutation vector
  Eigen::VectorXi permutation = Eigen::VectorXi::Zero(this->soln_size);
  primal_offset = 0;
  dual_offset_1 = (int) primal_size;
  dual_offset_2 = (int) (primal_size + dual_size_1);
  int perm_offset = 0;
  for (int t = 0; t < this->trajectory_length - 1; ++t) {
    for (int i = 0; i < n; ++i, ++perm_offset) {
      permutation(perm_offset) = dual_offset_1 + i;
    }
    dual_offset_1 += n;
    const int la = this->num_active_constraints[t];
    for (int i = 0; i < la; ++i, ++perm_offset) {
      permutation(perm_offset) = dual_offset_2 + i;
    }
    dual_offset_2 += la;
    for (int i = 0; i < n + m; ++i, ++perm_offset) {
      permutation(perm_offset) = primal_offset + i;
    }
    primal_offset += (n + m);
  }
  for (int i = 0; i < n; ++i, ++perm_offset) {
    permutation(perm_offset) = dual_offset_1 + i;
  }
  long lt = this->active_terminal_constraints.count();
  for (int i = 0; i < lt; ++i, ++perm_offset) {
    permutation(perm_offset) = dual_offset_2 + i;
  }
  for (int i = 0; i < n; ++i, ++perm_offset) {
    permutation(perm_offset) = primal_offset + i;
  }

  Eigen::PermutationMatrix<Eigen::Dynamic, Eigen::Dynamic, int> perm(permutation);

  this->A = perm.transpose() * tempA * perm;
  tempB = (perm.transpose() * tempB).eval();

  for (int t = 0; t < this->soln_size; ++t) {
    this->B[t] = tempB(t);
  }

  int LDAB = 3 * this->half_bandwidth + 1;

  this->AB.resize((unsigned long) LDAB * this->soln_size);

  int abi;
  for (int j = 1; j <= this->soln_size; ++j) {
    for (int i = std::max(1, j - this->half_bandwidth); i <= std::min(this->soln_size, j + this->half_bandwidth); ++i) {
      abi = 2 * this->half_bandwidth + 1 + i - j;
      this->AB[(abi - 1) + (j - 1) * LDAB] = this->A(i - 1, j - 1);
    }
  }
}

void Trajectory::bandsolve_traj() {
  int N = this->soln_size;
  int KL = this->half_bandwidth;
  int KU = this->half_bandwidth;
  int NRHS = 1;
  int LDAB = 3 * this->half_bandwidth + 1;
  int IPIV[this->soln_size];
  int LDB = std::max(1, this->soln_size);
  int INFO;

  //dgbsv_(&N, &KL, &KU, &NRHS, &*this->AB.begin(), &LDAB, IPIV, &*this->B.begin(), &LDB, &INFO);
  std::cout << "Bandsolve successful? " << INFO << std::endl;
}

void Trajectory::compute_feedback_policies() {
  const unsigned int n = this->state_dimension;
  const unsigned int m = this->control_dimension;
  const unsigned int l = this->terminal_constraint_dimension;
  const unsigned int li = this->implicit_terminal_constraint_dimension;

  Eigen::MatrixXd Mxx = Eigen::MatrixXd::Zero(n, n);
  Eigen::MatrixXd Muu = Eigen::MatrixXd::Zero(m, m);
  Eigen::MatrixXd Mzz = Eigen::MatrixXd::Zero(li, li);
  Eigen::MatrixXd Mux = Eigen::MatrixXd::Zero(m, n);
  Eigen::MatrixXd Mzx = Eigen::MatrixXd::Zero(li, n);
  Eigen::MatrixXd Mzu = Eigen::MatrixXd::Zero(li, m);
  Eigen::VectorXd Mx1 = Eigen::MatrixXd::Zero(n, 1);
  Eigen::VectorXd Mu1 = Eigen::MatrixXd::Zero(m, 1);
  Eigen::VectorXd Mz1 = Eigen::MatrixXd::Zero(li, 1);
  Eigen::MatrixXd M11 = Eigen::MatrixXd::Zero(1, 1);

  Eigen::MatrixXd Vxx = (this->terminal_cost_hessians_state_state).eval();
  Eigen::MatrixXd Vzz = Eigen::MatrixXd::Zero(li, li);
  Eigen::MatrixXd Vzx = Eigen::MatrixXd::Zero(li, n);
  Eigen::VectorXd Vx1 = (this->terminal_cost_gradient_state).eval();
  Eigen::VectorXd Vz1 = Eigen::MatrixXd::Zero(li, 1);
  Eigen::MatrixXd V11 = Eigen::MatrixXd::Zero(1, 1);

  this->cost_to_go_hessians_state_state[this->trajectory_length - 1] = this->terminal_cost_hessians_state_state.eval();
  this->cost_to_go_gradients_state[this->trajectory_length - 1] = this->terminal_cost_gradient_state.eval();

  this->auxiliary_constraints_present[this->trajectory_length - 1] = (l > 0);

  Eigen::MatrixXd Gx = this->terminal_constraint_jacobian_state.topLeftCorner(l, n).eval();
  Eigen::MatrixXd Gz = this->terminal_constraint_jacobian_terminal_projection.topLeftCorner(l, li).eval();
  Eigen::VectorXd G1 = this->terminal_constraint_affine_term.head(l).eval();

  Eigen::MatrixXd Nx = Eigen::MatrixXd::Zero(n, n);
  Eigen::MatrixXd Nu = Eigen::MatrixXd::Zero(n, m);
  Eigen::MatrixXd Nz = Eigen::MatrixXd::Zero(n, li);
  Eigen::VectorXd N1 = Eigen::MatrixXd::Zero(n, 1);

  for (int t = this->trajectory_length - 2; t >= 0; --t) {
//    double z_norms = Vzx.norm();
//    this->implicit_terminal_terms_needed[t] =
//        (((z_norms > 1e-6) or Gz.size() > 0) and this->implicit_terminal_constraint_dimension > 0);
    this->implicit_terminal_terms_needed[t] = true;

//    this->implicit_terminal_terms_needed[t]
    this->auxiliary_constraints_present[t] = ((this->num_active_constraints[t] + G1.size()) > 0);
    this->need_dynamics_mult[t] =
        (this->auxiliary_constraints_present[t + 1] > 0 and this->auxiliary_constraints_present[t] == 0);

    this->perform_constrained_dynamic_programming_backup(&this->hamiltonian_hessians_state_state[t],
                                                         &this->hamiltonian_hessians_control_control[t],
                                                         &this->hamiltonian_hessians_control_state[t],
                                                         &this->hamiltonian_gradients_state[t],
                                                         &this->hamiltonian_gradients_control[t],
                                                         &this->dynamics_jacobians_state[t],
                                                         &this->dynamics_jacobians_control[t],
                                                         &this->dynamics_affine_terms[t],
                                                         &this->running_constraint_jacobians_state[t],
                                                         &this->running_constraint_jacobians_control[t],
                                                         &this->running_constraint_affine_terms[t],
                                                         this->num_active_constraints[t],
                                                         this->implicit_terminal_terms_needed[t],
                                                         Mxx,
                                                         Muu,
                                                         Mzz,
                                                         Mux,
                                                         Mzx,
                                                         Mzu,
                                                         Mx1,
                                                         Mu1,
                                                         Mz1,
                                                         M11,
                                                         N1,
                                                         Nx,
                                                         Nu,
                                                         Nz,
                                                         Vxx,
                                                         Vzz,
                                                         Vzx,
                                                         Vx1,
                                                         Vz1,
                                                         V11,
                                                         Gx,
                                                         Gz,
                                                         G1,
                                                         this->current_state_feedback_matrices[t],
                                                         this->terminal_state_feedback_matrices[t],
                                                         this->feedforward_controls[t]);
    this->cost_to_go_hessians_state_state[t] = Vxx.eval();
    this->cost_to_go_hessians_terminal_state[t] = Vzx.eval();
    this->cost_to_go_hessians_terminal_terminal[t] = Vzz.eval();
    this->cost_to_go_gradients_state[t] = Vx1.eval();
    this->cost_to_go_gradients_terminal[t] = Vz1.eval();
    this->cost_to_go_offsets[t] = V11.eval();
  }
  this->need_dynamics_mult[0] = this->initial_constraint.is_implicit();
}

void Trajectory::perform_constrained_dynamic_programming_backup(const Eigen::MatrixXd *Qxx,
                                                                const Eigen::MatrixXd *Quu,
                                                                const Eigen::MatrixXd *Qux,
                                                                const Eigen::VectorXd *Qx,
                                                                const Eigen::VectorXd *Qu,
                                                                const Eigen::MatrixXd *Ax,
                                                                const Eigen::MatrixXd *Au,
                                                                const Eigen::VectorXd *A1,
                                                                const Eigen::MatrixXd *Dx,
                                                                const Eigen::MatrixXd *Du,
                                                                const Eigen::VectorXd *D1,
                                                                unsigned int num_active_constraints,
                                                                bool implicit_terminal_terms_needed,
                                                                Eigen::MatrixXd &Mxx,
                                                                Eigen::MatrixXd &Muu,
                                                                Eigen::MatrixXd &Mzz,
                                                                Eigen::MatrixXd &Mux,
                                                                Eigen::MatrixXd &Mzx,
                                                                Eigen::MatrixXd &Mzu,
                                                                Eigen::VectorXd &Mx1,
                                                                Eigen::VectorXd &Mu1,
                                                                Eigen::VectorXd &Mz1,
                                                                Eigen::MatrixXd &M11,
                                                                Eigen::VectorXd &N1,
                                                                Eigen::MatrixXd &Nx,
                                                                Eigen::MatrixXd &Nu,
                                                                Eigen::MatrixXd &Nz,
                                                                Eigen::MatrixXd &Vxx,
                                                                Eigen::MatrixXd &Vzz,
                                                                Eigen::MatrixXd &Vzx,
                                                                Eigen::VectorXd &Vx1,
                                                                Eigen::VectorXd &Vz1,
                                                                Eigen::MatrixXd &V11,
                                                                Eigen::MatrixXd &Gx,
                                                                Eigen::MatrixXd &Gz,
                                                                Eigen::VectorXd &G1,
                                                                Eigen::MatrixXd &Lx,
                                                                Eigen::MatrixXd &Lz,
                                                                Eigen::VectorXd &L1) {
  // Todo: Optimize these calculations for number of arithmitic operations
  const Eigen::VectorXd mtemp1 = Vxx * (*A1) + Vx1;

  M11 = V11 + (*A1).transpose() * (mtemp1 + Vx1);
  Mx1 = (*Ax).transpose() * mtemp1 + *Qx;
  Mu1 = (*Au).transpose() * mtemp1 + *Qu;
  Mxx = (*Ax).transpose() * Vxx * (*Ax) + *Qxx;
  Mux = (*Au).transpose() * Vxx * (*Ax) + *Qux;
  Muu = (*Au).transpose() * Vxx * (*Au) + *Quu;

  if (implicit_terminal_terms_needed) {
    Mz1 = Vzx * (*A1) + Vz1;
    Mzx = Vzx * (*Ax);
    Mzu = Vzx * (*Au);
    Mzz = Vzz;

    Eigen::MatrixXd BIG
        (1 + this->state_dimension + this->control_dimension + this->implicit_terminal_constraint_dimension,
         1 + this->state_dimension + this->control_dimension + this->implicit_terminal_constraint_dimension);
    BIG
        << M11, Mx1.transpose(), Mu1.transpose(), Mz1.transpose(), Mx1, Mxx, Mux.transpose(), Mzx.transpose(), Mu1, Mux, Muu, Mzu.transpose(), Mz1, Mzx, Mzu, Mzz;
    BIG = 0.5 * (BIG + BIG.transpose());
    M11 = BIG.topLeftCorner(1, 1);
    Mx1 = BIG.block(1, 0, this->state_dimension, 1);
    Mu1 = BIG.block(1 + this->state_dimension, 0, this->control_dimension, 1);
    Mz1 = BIG.block(1 + this->state_dimension + this->control_dimension,
                    0,
                    this->implicit_terminal_constraint_dimension,
                    1);
    Mxx = BIG.block(1, 1, this->state_dimension, this->state_dimension);
    Mux = BIG.block(1 + this->state_dimension, 1, this->control_dimension, this->state_dimension);
    Mzx = BIG.block(1 + this->state_dimension + this->control_dimension,
                    1,
                    this->implicit_terminal_constraint_dimension,
                    this->state_dimension);
    Muu =
        BIG.block(1 + this->state_dimension,
                  1 + this->state_dimension,
                  this->control_dimension,
                  this->control_dimension);
    Mzu = BIG.block(1 + this->state_dimension + this->control_dimension,
                    1 + this->state_dimension,
                    this->implicit_terminal_constraint_dimension,
                    this->control_dimension);
    Mzz =
        BIG.bottomRightCorner(this->implicit_terminal_constraint_dimension,
                              this->implicit_terminal_constraint_dimension);
  } else {
    Eigen::MatrixXd BIG
        (1 + this->state_dimension + this->control_dimension,
         1 + this->state_dimension + this->control_dimension);
    BIG
        << M11, Mx1.transpose(), Mu1.transpose(), Mx1, Mxx, Mux.transpose(), Mu1, Mux, Muu;
    BIG = 0.5 * (BIG + BIG.transpose());
    M11 = BIG.topLeftCorner(1, 1);
    Mx1 = BIG.block(1, 0, this->state_dimension, 1);
    Mu1 = BIG.block(1 + this->state_dimension, 0, this->control_dimension, 1);

    Mxx = BIG.block(1, 1, this->state_dimension, this->state_dimension);
    Mux = BIG.block(1 + this->state_dimension, 1, this->control_dimension, this->state_dimension);
    Muu =
        BIG.block(1 + this->state_dimension,
                  1 + this->state_dimension,
                  this->control_dimension,
                  this->control_dimension);
  }

  long constraint_size = num_active_constraints + G1.size();
  N1.resize(constraint_size);
  Nx.resize(constraint_size, this->state_dimension);
  Nu.resize(constraint_size, this->control_dimension);

  N1 << (*D1).head(num_active_constraints), Gx * (*A1) + G1;
  Nx << (*Dx).topRows(num_active_constraints), Gx * (*Ax);
  Nu << (*Du).topRows(num_active_constraints), Gx * (*Au);

  if (implicit_terminal_terms_needed) {
    Nz.resize(constraint_size, this->implicit_terminal_constraint_dimension);
    Nz << Eigen::MatrixXd::Zero(num_active_constraints, this->implicit_terminal_constraint_dimension), Gz;
  }

  if (constraint_size > 0) {
    Eigen::JacobiSVD<Eigen::MatrixXd> svd(Nu, Eigen::ComputeFullU | Eigen::ComputeFullV);
    long constraint_rank = svd.rank();
    if (constraint_rank == 0) {
      Eigen::FullPivLU<Eigen::MatrixXd> decMuu(Muu);
      Lx = -decMuu.solve(Mux);
      L1 = -decMuu.solve(Mu1);
      Gx = Nx;
      G1 = N1;

      if (implicit_terminal_terms_needed) {
        Lz = -decMuu.solve(Mzu.transpose());
        Gz = Nz;
      }

    } else {

      Eigen::MatrixXd U_Nu_t = svd.matrixU().transpose();
      Eigen::MatrixXd V_Nu = svd.matrixV();
      Eigen::MatrixXd V_Nu_l = V_Nu.leftCols(constraint_rank);
      Eigen::MatrixXd V_Nu_r = V_Nu.rightCols(this->control_dimension - constraint_rank);

      const Eigen::VectorXd singular = svd.singularValues().head(constraint_rank);
      const Eigen::VectorXd singular_inv = singular.cwiseInverse();
      const Eigen::MatrixXd Si_Nu = singular_inv.asDiagonal();

      N1 = (U_Nu_t * N1).eval();
      Nx = (U_Nu_t * Nx).eval();
      Nu = (U_Nu_t * Nu).eval();

      if (implicit_terminal_terms_needed) {
        Nz = (U_Nu_t * Nz).eval();
      }

      Gx.resize(constraint_size - constraint_rank, this->state_dimension);
      G1.resize(constraint_size - constraint_rank, 1);

      Gx << Nx.bottomRows(constraint_size - constraint_rank);
      G1 << N1.tail(constraint_size - constraint_rank);

      N1.conservativeResize(constraint_rank);
      Nx.conservativeResize(constraint_rank, Eigen::NoChange);
      Nu.conservativeResize(constraint_rank, Eigen::NoChange);

      if (implicit_terminal_terms_needed) {
        Gz.resize(constraint_size - constraint_rank, this->implicit_terminal_constraint_dimension);
        Gz << Nz.bottomRows(constraint_size - constraint_rank);
        Nz.conservativeResize(constraint_rank, Eigen::NoChange);
      }

      const Eigen::MatrixXd VlSi = -V_Nu_l * Si_Nu;
      if (constraint_rank >= this->control_dimension) { // constraint_rank == control_dimension
        Lx = VlSi * Nx;
        L1 = VlSi * N1;
        if (implicit_terminal_terms_needed) {
          Lz = VlSi * Nz;
        }
      } else { // 0 < constraint_rank < control_dimension
        Eigen::VectorXd T1 = VlSi * N1;
        Eigen::MatrixXd Tx = VlSi * Nx;
        Eigen::MatrixXd Tu = V_Nu_r;

        // Order of these operations matter
        Eigen::MatrixXd temp1, temp2, temp3;
        Eigen::MatrixXd tmuu, tmu1, tmux, tmzu;

        temp1 = Mu1 + Muu * T1;
        temp2 = Mux + Muu * Tx;
        temp3 = Muu * Tu;

        tmu1 = Tu.transpose() * temp1;
        tmux = Tu.transpose() * temp2;
        tmuu = Tu.transpose() * temp3;

        Eigen::FullPivLU<Eigen::MatrixXd> decMuu(tmuu);

        temp1 = -decMuu.solve(tmu1);
        temp2 = -decMuu.solve(tmux);

        L1 = T1 + V_Nu_r * temp1;
        Lx = Tx + V_Nu_r * temp2;

        if (implicit_terminal_terms_needed) {
          Eigen::MatrixXd Tz = VlSi * Nz;
          tmzu = Tz.transpose() * temp3 + Mzu * Tu;
          temp3 = -decMuu.solve(tmzu.transpose());
          Lz = Tz + V_Nu_r * temp3;
        }
      }
    }
  } else {

    Eigen::FullPivLU<Eigen::MatrixXd> decMuu(Muu);
    Lx = -decMuu.solve(Mux);
    L1 = -decMuu.solve(Mu1);

    Gx.resize(0, this->state_dimension);
    G1.resize(0, 1);

    if (implicit_terminal_terms_needed) {
      Lz = -decMuu.solve(Mzu.transpose());
      Gz.resize(0, this->implicit_terminal_constraint_dimension);
    }

  }

  const Eigen::VectorXd temp1 = Mu1 + Muu * L1;
  const Eigen::MatrixXd temp2 = Mux + Muu * Lx;

  V11 = M11 + Mu1.transpose() * L1 + L1.transpose() * temp1;
  Vx1 = Mx1 + Mux.transpose() * L1 + Lx.transpose() * temp1;
  Vxx = Mxx + Mux.transpose() * Lx + Lx.transpose() * temp2;

  if (implicit_terminal_terms_needed) {
    Vz1 = Mz1 + Mzu * L1 + Lz.transpose() * temp1;
    Vzx = Mzx + Mzu * Lx + Lz.transpose() * temp2;
    Vzz = Mzz + Mzu * Lz + Lz.transpose() * (Mzu.transpose() + Muu * Lz);
    Eigen::MatrixXd VIG(1 + this->state_dimension + this->implicit_terminal_constraint_dimension,
                        1 + this->state_dimension + this->implicit_terminal_constraint_dimension);
    VIG << V11, Vx1.transpose(), Vz1.transpose(), Vx1, Vxx, Vzx.transpose(), Vz1, Vzx, Vzz;
    VIG = 0.5 * (VIG + VIG.transpose());
    V11 = VIG.topLeftCorner(1, 1);
    Vx1 = VIG.block(1, 0, this->state_dimension, 1);
    Vz1 = VIG.block(1 + this->state_dimension, 0, this->implicit_terminal_constraint_dimension, 1);
    Vxx = VIG.block(1, 1, this->state_dimension, this->state_dimension);
    Vzx = VIG.block(1 + this->state_dimension, 1, this->implicit_terminal_constraint_dimension, this->state_dimension);
    Vzz =
        VIG.bottomRightCorner(this->implicit_terminal_constraint_dimension,
                              this->implicit_terminal_constraint_dimension);
  } else {
    Eigen::MatrixXd VIG(1 + this->state_dimension,
                        1 + this->state_dimension);
    VIG << V11, Vx1.transpose(), Vx1, Vxx;
    VIG = 0.5 * (VIG + VIG.transpose());
    V11 = VIG.topLeftCorner(1, 1);
    Vx1 = VIG.block(1, 0, this->state_dimension, 1);
    Vxx = VIG.block(1, 1, this->state_dimension, this->state_dimension);

    Vzx.setZero();
    Vz1.setZero();
    Vzz.setZero();
    Lz.setZero();
    Gz.setZero();
  }

//  Eigen::MatrixXd CL = (*Ax) + (*Au) * Lx;
//  Eigen::EigenSolver<Eigen::MatrixXd> eigensolver(CL);
//  std::cout<<"Eigenvalues of cl system are: " << eigensolver.eigenvalues().transpose()<<std::endl;
}

void Trajectory::compute_multipliers() {
  const unsigned int n = this->state_dimension;
  const unsigned int l = this->terminal_constraint_dimension;
  const unsigned int li = this->implicit_terminal_constraint_dimension;
  const unsigned int T = this->trajectory_length;
  unsigned int j, nextj;
  Eigen::MatrixXd H, G, Omega, Psi, Sigma, A, B, U, V, W, Y, Q, S, R, Hnext, Anext, Qnext, Snext, LHS, Pproj, Plam;
  Eigen::VectorXd q, r, qnext, P1;
  Eigen::MatrixXd Ex, Ez, Dx, Dz;
  Eigen::VectorXd E1, D1;
  H = this->terminal_constraint_jacobian_state.topRows(l);
  Omega = H * H.transpose();
  Q = this->terminal_cost_hessians_state_state;
  q = this->terminal_cost_gradient_state;
  U = H * Q;

  Ex = U * this->state_dependencies_initial_state_projection[T - 1];
  Ez = U * this->state_dependencies_terminal_state_projection[T - 1].leftCols(li);
  E1 = U * this->state_dependencies_affine_term[T - 1] + H * q;
  Eigen::PartialPivLU<Eigen::MatrixXd> decomp;
  if (l > 0) {
    decomp.compute(-Omega);
    this->terminal_constraint_mult_dynamics_mult_feedback_term.topRows(l) = decomp.solve(H);
    this->terminal_constraint_mult_initial_state_feedback_term.topRows(l) = decomp.solve(Ex);
    this->terminal_constraint_mult_terminal_state_feedback_term.topLeftCorner(l, li) = decomp.solve(Ez);
    this->terminal_constraint_mult_feedforward_term.head(l) = decomp.solve(E1);
  } else {
    this->terminal_constraint_mult_dynamics_mult_feedback_term.topRows(l) = Eigen::MatrixXd::Zero(l, n);
    this->terminal_constraint_mult_initial_state_feedback_term.topRows(l) = Eigen::MatrixXd::Zero(l, n);
    this->terminal_constraint_mult_terminal_state_feedback_term.topLeftCorner(l, li) = Eigen::MatrixXd::Zero(l, li);
    this->terminal_constraint_mult_feedforward_term.head(l) = Eigen::VectorXd::Zero(l);
  }

  for (int t = T - 2; t >= 0; --t) {
//    if (true) {
    if (this->auxiliary_constraints_present[t + 1]) {
      j = this->num_active_constraints[t];

      if ((unsigned int) t < T - 2) {
        nextj = this->num_active_constraints[t + 1];
        Hnext = this->running_constraint_jacobians_state[t + 1].topRows(nextj);
        Anext = this->dynamics_jacobians_state[t + 1];
        Qnext = this->hamiltonian_hessians_state_state[t + 1];
        qnext = this->hamiltonian_gradients_state[t + 1];
        Snext = this->hamiltonian_hessians_control_state[t + 1];
      } else {
        nextj = l;
        Hnext = this->terminal_constraint_jacobian_state.topRows(nextj);
        Qnext = this->terminal_cost_hessians_state_state;
        qnext = this->terminal_cost_gradient_state;
      }

      H = this->running_constraint_jacobians_state[t].topRows(j);
      G = this->running_constraint_jacobians_control[t].topRows(j);
      A = this->dynamics_jacobians_state[t];
      B = this->dynamics_jacobians_control[t];
      Q = this->hamiltonian_hessians_state_state[t];
      S = this->hamiltonian_hessians_control_state[t];
      R = this->hamiltonian_hessians_control_control[t];
      q = this->hamiltonian_gradients_state[t];
      r = this->hamiltonian_gradients_control[t];

      Omega = H * H.transpose() + G * G.transpose();
      Psi = -A * H.transpose() - B * G.transpose();
      Sigma = A * A.transpose() + B * B.transpose() + Eigen::MatrixXd::Identity(n, n);
      U = H * Q + G * S;
      V = -A * Q - B * S;
      W = H * S.transpose() + G * R;
      Y = -A * S.transpose() - B * R;

      Dx = V * this->state_dependencies_initial_state_projection[t] +
          Y * this->control_dependencies_initial_state_projection[t] +
          Qnext * this->state_dependencies_initial_state_projection[t + 1];

      Dz = V * this->state_dependencies_terminal_state_projection[t].leftCols(li) +
          Y * this->control_dependencies_terminal_state_projection[t].leftCols(li) +
          Qnext * this->state_dependencies_terminal_state_projection[t + 1].leftCols(li);

      D1 = V * this->state_dependencies_affine_term[t] +
          Y * this->control_dependencies_affine_term[t] +
          Qnext * this->state_dependencies_affine_term[t + 1] + qnext - A * q - B * r;

      Ex = U * this->state_dependencies_initial_state_projection[t] +
          W * this->control_dependencies_initial_state_projection[t];
      Ez = U * this->state_dependencies_terminal_state_projection[t].leftCols(li) +
          W * this->control_dependencies_terminal_state_projection[t].leftCols(li);
      E1 = U * this->state_dependencies_affine_term[t] +
          W * this->control_dependencies_affine_term[t] + H * q + G * r;

      LHS.resize(j + n, j + n);
      Plam.resize(j + n, n);
      Pproj.resize(j + n, li + n);
      P1.resize(j + n);

      if ((unsigned int) t < T - 2) {
        Sigma += (Hnext.transpose() * this->running_constraint_mult_dynamics_mult_feedback_terms[t + 1].topRows(nextj)
            - Anext.transpose() * this->dynamics_mult_dynamics_mult_feedback_terms[t + 2]).eval();
        Dx += (Hnext.transpose() * this->running_constraint_mult_initial_state_feedback_terms[t + 1].topRows(nextj)
            - Anext.transpose() * this->dynamics_mult_initial_state_feedback_terms[t + 2]
            + Snext.transpose() * this->control_dependencies_initial_state_projection[t + 1]).eval();
        Dz += (Hnext.transpose()
            * this->running_constraint_mult_terminal_state_feedback_terms[t + 1].topLeftCorner(nextj, li)
            - Anext.transpose() * this->dynamics_mult_terminal_state_feedback_terms[t + 2].leftCols(li)
            + Snext.transpose() * this->control_dependencies_terminal_state_projection[t + 1].leftCols(li)).eval();
        D1 += (Hnext.transpose() * this->running_constraint_mult_feedforward_terms[t + 1].topRows(nextj)
            - Anext.transpose() * this->dynamics_mult_feedforward_terms[t + 2]
            + Snext.transpose() * this->control_dependencies_affine_term[t + 1]).eval();
      } else {
        Sigma += (Hnext.transpose() * this->terminal_constraint_mult_dynamics_mult_feedback_term.topRows(l)).eval();
        Dx += (Hnext.transpose() * this->terminal_constraint_mult_initial_state_feedback_term.topRows(l)).eval();
        Dz +=
            (Hnext.transpose()
                * this->terminal_constraint_mult_terminal_state_feedback_term.topLeftCorner(l, li)).eval();
        D1 += (Hnext.transpose() * this->terminal_constraint_mult_feedforward_term.head(l)).eval();
      }
      Pproj.topLeftCorner(j, n) = Ex;
      Pproj.topRightCorner(j, li) = Ez;
      Pproj.bottomLeftCorner(n, n) = Dx;
      Pproj.bottomRightCorner(n, li) = Dz;
      //Pproj << Ex, Ez, Dx, Dz;
      P1.head(j) = E1;
      P1.tail(n) = D1;
      //P1 << E1, D1;
      LHS.topLeftCorner(j, j) = Omega;
      LHS.topRightCorner(j, n) = Psi.transpose();
      LHS.bottomLeftCorner(n, j) = Psi;
      LHS.bottomRightCorner(n, n) = Sigma;
      //LHS << Omega, Psi.transpose(), Psi, Sigma;
      Plam.topRows(j) = H;
      Plam.bottomRows(n) = -A;
      //Plam << H, -A;
      decomp.compute(-LHS);
      Pproj = (decomp.solve(Pproj)).eval();
      Plam = (decomp.solve(Plam)).eval();
      P1 = (decomp.solve(P1)).eval();
      this->running_constraint_mult_initial_state_feedback_terms[t].topRows(j) = Pproj.block(0, 0, j, n);
      this->running_constraint_mult_terminal_state_feedback_terms[t].topLeftCorner(j, li) = Pproj.block(0, n, j, li);
      this->running_constraint_mult_dynamics_mult_feedback_terms[t].topRows(j) = Plam.topRows(j);
      this->running_constraint_mult_feedforward_terms[t].head(j) = P1.head(j);
      this->dynamics_mult_initial_state_feedback_terms[t + 1] = Pproj.block(j, 0, n, n);
      this->dynamics_mult_terminal_state_feedback_terms[t + 1].leftCols(li) = Pproj.block(j, n, n, li);
      this->dynamics_mult_dynamics_mult_feedback_terms[t + 1] = Plam.bottomRows(n);
      this->dynamics_mult_feedforward_terms[t + 1] = P1.tail(n);
    } else if (this->need_dynamics_mult[t + 1]) {
      Eigen::MatrixXd Vxx, Vzx;
      Eigen::VectorXd Vx1;
      Vxx = this->cost_to_go_hessians_state_state[t + 1];
      Vzx = this->cost_to_go_hessians_terminal_state[t + 1];
      Vx1 = this->cost_to_go_gradients_state[t + 1];
      this->dynamics_mult_initial_state_feedback_terms[t + 1] =
          -(Vxx * this->state_dependencies_initial_state_projection[t + 1]);
      this->dynamics_mult_terminal_state_feedback_terms[t + 1] =
          -(Vzx.transpose() + Vxx * this->state_dependencies_terminal_state_projection[t + 1].leftCols(li));
      this->dynamics_mult_feedforward_terms[t + 1] = -(Vx1 + Vxx * this->state_dependencies_affine_term[t + 1]);
    }
  }
//  if (true) {
  if (this->auxiliary_constraints_present[0]) {
    j = this->num_active_constraints[0];
    Q = this->hamiltonian_hessians_state_state[0];
    S = this->hamiltonian_hessians_control_state[0];
    H = this->running_constraint_jacobians_state[0].topRows(j);
    A = this->dynamics_jacobians_state[0];
    q = this->hamiltonian_gradients_state[0];
    Dx = Q * this->state_dependencies_initial_state_projection[0] +
        S.transpose() * this->control_dependencies_initial_state_projection[0] +
        H.transpose() * this->running_constraint_mult_initial_state_feedback_terms[0].topRows(j)
        - A.transpose() * this->dynamics_mult_initial_state_feedback_terms[1];
    Dz = Q * this->state_dependencies_terminal_state_projection[0] +
        S.transpose() * this->control_dependencies_terminal_state_projection[0].leftCols(li) +
        H.transpose() * this->running_constraint_mult_terminal_state_feedback_terms[0].topLeftCorner(j, li)
        - A.transpose() * this->dynamics_mult_terminal_state_feedback_terms[1].leftCols(li);
    D1 = Q * this->state_dependencies_affine_term[0] + S.transpose() * this->control_dependencies_affine_term[0] + q +
        H.transpose() * this->running_constraint_mult_feedforward_terms[0].head(j)
        - A.transpose() * this->dynamics_mult_feedforward_terms[1];
    Plam = Eigen::MatrixXd::Identity(n, n)
        + H.transpose() * this->running_constraint_mult_dynamics_mult_feedback_terms[0].topRows(j)
        - A.transpose() * this->dynamics_mult_dynamics_mult_feedback_terms[1];
    decomp.compute(-Plam);
    this->dynamics_mult_initial_state_feedback_terms[0] = decomp.solve(Dx);
    this->dynamics_mult_terminal_state_feedback_terms[0].leftCols(li) = decomp.solve(Dz);
    this->dynamics_mult_feedforward_terms[0] = decomp.solve(D1);
  } else if (this->need_dynamics_mult[0]) {
    Eigen::MatrixXd Vxx, Vzx;
    Eigen::VectorXd Vx1;
    Vxx = this->cost_to_go_hessians_state_state[0];
    Vzx = this->cost_to_go_hessians_terminal_state[0];
    Vx1 = this->cost_to_go_gradients_state[0];
    this->dynamics_mult_initial_state_feedback_terms[0] = -(Vxx * this->state_dependencies_initial_state_projection[0]);
    this->dynamics_mult_terminal_state_feedback_terms[0] =
        -(Vzx.transpose() + Vxx * this->state_dependencies_terminal_state_projection[0]);
    this->dynamics_mult_feedforward_terms[0] = -(Vx1 + Vxx * this->state_dependencies_affine_term[0]);
  };

  // Forward solve for Multiplier dependencies
  for (unsigned int t = 0; t < T - 1; ++t) {
    j = (t < T - 1) ? this->num_active_constraints[t] : l;
    if (this->auxiliary_constraints_present[t + 1]) {
      this->running_constraint_mult_initial_state_feedback_terms[t].topRows(j) +=
          (this->running_constraint_mult_dynamics_mult_feedback_terms[t].topRows(j)
              * this->dynamics_mult_initial_state_feedback_terms[t]).eval();
      this->running_constraint_mult_terminal_state_feedback_terms[t].topLeftCorner(j, li) +=
          (this->running_constraint_mult_dynamics_mult_feedback_terms[t].topRows(j)
              * this->dynamics_mult_terminal_state_feedback_terms[t].leftCols(li)).eval();
      this->running_constraint_mult_feedforward_terms[t].topRows(j) +=
          (this->running_constraint_mult_dynamics_mult_feedback_terms[t].topRows(j)
              * this->dynamics_mult_feedforward_terms[t]).eval();
      this->dynamics_mult_initial_state_feedback_terms[t + 1] +=
          (this->dynamics_mult_dynamics_mult_feedback_terms[t + 1]
              * this->dynamics_mult_initial_state_feedback_terms[t]).eval();
      this->dynamics_mult_terminal_state_feedback_terms[t + 1].leftCols(li) +=
          (this->dynamics_mult_dynamics_mult_feedback_terms[t + 1]
              * this->dynamics_mult_terminal_state_feedback_terms[t].leftCols(li)).eval();
      this->dynamics_mult_feedforward_terms[t + 1] +=
          (this->dynamics_mult_dynamics_mult_feedback_terms[t + 1] * this->dynamics_mult_feedforward_terms[t]).eval();
    }
  }
  this->terminal_constraint_mult_initial_state_feedback_term.topRows(l) +=
      (this->terminal_constraint_mult_dynamics_mult_feedback_term.topRows(l)
          * this->dynamics_mult_initial_state_feedback_terms[T - 1]).eval();
  this->terminal_constraint_mult_terminal_state_feedback_term.topLeftCorner(l, li) +=
      (this->terminal_constraint_mult_dynamics_mult_feedback_term.topRows(l)
          * this->dynamics_mult_terminal_state_feedback_terms[T - 1].leftCols(li)).eval();
  this->terminal_constraint_mult_feedforward_term.head(l) +=
      (this->terminal_constraint_mult_dynamics_mult_feedback_term.topRows(l)
          * this->dynamics_mult_feedforward_terms[T - 1]).eval();
}

void Trajectory::compute_state_control_dependencies() {
  const unsigned int n = this->state_dimension;
  const unsigned int li = this->implicit_terminal_constraint_dimension;
  bool initial_implicit = this->initial_constraint.is_implicit();
  const unsigned int T = this->trajectory_length;
  bool initial_depencence = true;

  Eigen::PartialPivLU<Eigen::MatrixXd> dec(this->initial_constraint_jacobian_state);
  if (initial_implicit) {
    this->state_dependencies_initial_state_projection[0] = -dec.solve(Eigen::MatrixXd::Identity(n, n));
    this->state_dependencies_affine_term[0] = Eigen::VectorXd::Zero(n);
  } else {
    this->state_dependencies_initial_state_projection[0] = Eigen::MatrixXd::Zero(n, n);
    this->state_dependencies_affine_term[0] = -dec.solve(this->initial_constraint_affine_term);
  }
  this->state_dependencies_terminal_state_projection[0] = Eigen::MatrixXd::Zero(n, li);

  for (unsigned int t = 0; t < T - 1; ++t) {
    // Create shortcuts for terms
    const auto &Ax = this->dynamics_jacobians_state[t];
    const auto &Au = this->dynamics_jacobians_control[t];
    const auto &A1 = this->dynamics_affine_terms[t];
    const auto &Kx = this->current_state_feedback_matrices[t];
    const auto &K1 = this->feedforward_controls[t];
    const auto &Mx = this->state_dependencies_initial_state_projection[t];
    const auto &M1 = this->state_dependencies_affine_term[t];

    if (li > 0) {
      const auto &Kz = this->terminal_state_feedback_matrices[t].leftCols(li);
      const auto &Mz = this->state_dependencies_terminal_state_projection[t].leftCols(li);
      this->control_dependencies_terminal_state_projection[t].leftCols(li) = Kx * Mz + Kz;
      this->state_dependencies_terminal_state_projection[t + 1].leftCols(li) =
          Ax * Mz + Au * this->control_dependencies_terminal_state_projection[t].leftCols(li);
    }
    if (initial_implicit) {
      // Recursively update control dependencies
      this->control_dependencies_initial_state_projection[t] = Kx * Mx;
      this->control_dependencies_affine_term[t] = Kx * M1 + K1;
      // Recursively update state dependencies

      this->state_dependencies_initial_state_projection[t + 1] =
          Ax * Mx + Au * this->control_dependencies_initial_state_projection[t];
      this->state_dependencies_affine_term[t + 1] = Ax * M1 + Au * this->control_dependencies_affine_term[t] + A1;

    } else {
      this->control_dependencies_affine_term[t] = Kx * (M1 + Mx * this->initial_constraint_affine_term) + K1;
      // Recursively update state dependencies
      this->state_dependencies_affine_term[t + 1] = Ax * (M1 + Mx * this->initial_constraint_affine_term) + Au
          * (this->control_dependencies_affine_term[t]
              + this->control_dependencies_initial_state_projection[t] * this->initial_constraint_affine_term) + A1;
    }
  }
}

// Setters

void Trajectory::set_terminal_point(Eigen::VectorXd *terminal_projection) {
  this->terminal_state_projection = *terminal_projection;
}

void Trajectory::set_initial_constraint_dimension(unsigned int d) {
  this->initial_constraint_dimension = d;
}

void Trajectory::set_initial_constraint_jacobian_state(const Eigen::MatrixXd *H) {
  this->initial_constraint_jacobian_state = *H;
}

void Trajectory::set_initial_constraint_affine_term(const Eigen::VectorXd *h) {
  this->initial_constraint_affine_term = *h;
}

void Trajectory::set_terminal_constraint_dimension(unsigned int d) {
  this->terminal_constraint_dimension = d;
}

void Trajectory::set_terminal_constraint_jacobian_state(const Eigen::MatrixXd *H) {
  this->terminal_constraint_jacobian_state = *H;
}

void Trajectory::set_terminal_constraint_jacobian_terminal_projection(const Eigen::MatrixXd *H) {
  this->terminal_constraint_jacobian_terminal_projection = *H;
}

void Trajectory::set_terminal_constraint_affine_term(const Eigen::VectorXd *h) {
  this->terminal_constraint_affine_term = *h;
}

void Trajectory::set_dynamics_jacobian_state(unsigned int t, const Eigen::MatrixXd *A) {
  this->dynamics_jacobians_state[t] = *A;
}

void Trajectory::set_dynamics_jacobian_control(unsigned int t, const Eigen::MatrixXd *B) {
  this->dynamics_jacobians_control[t] = *B;
}

void Trajectory::set_dynamics_affine_term(unsigned int t, const Eigen::VectorXd *c) {
  this->dynamics_affine_terms[t] = *c;
}

void Trajectory::set_num_active_constraints(unsigned int t, unsigned int num_active_constraints) {
  this->num_active_constraints[t] = num_active_constraints;
}

void Trajectory::set_hamiltonian_hessians_state_state(unsigned int t, const Eigen::MatrixXd *Qxx) {
  this->hamiltonian_hessians_state_state[t] = *Qxx;
}

void Trajectory::set_hamiltonian_hessians_control_state(unsigned int t, const Eigen::MatrixXd *Qux) {
  this->hamiltonian_hessians_control_state[t] = *Qux;
}

void Trajectory::set_hamiltonian_hessians_control_control(unsigned int t, const Eigen::MatrixXd *Quu) {
  this->hamiltonian_hessians_control_control[t] = *Quu;
}

void Trajectory::set_hamiltonian_gradients_state(unsigned int t, const Eigen::VectorXd *Qx) {
  this->hamiltonian_gradients_state[t] = *Qx;
}

void Trajectory::set_hamiltonian_gradients_control(unsigned int t, const Eigen::VectorXd *Qu) {
  this->hamiltonian_gradients_control[t] = *Qu;
}

void Trajectory::set_terminal_cost_hessians_state_state(const Eigen::MatrixXd *Qxx) {
  this->terminal_cost_hessians_state_state = *Qxx;
}

void Trajectory::set_terminal_cost_gradient_state(const Eigen::VectorXd *Qx) {
  this->terminal_cost_gradient_state = *Qx;
}

// Getters

void Trajectory::get_state_dependencies_initial_state_projection(unsigned int t, Eigen::MatrixXd &Tx) const {
  Tx = this->state_dependencies_initial_state_projection[t];
}

void Trajectory::get_state_dependencies_terminal_state_projection(unsigned int t, Eigen::MatrixXd &Tz) const {
  Tz = this->state_dependencies_terminal_state_projection[t];
}

void Trajectory::get_state_dependencies_affine_term(unsigned int t, Eigen::VectorXd &T1) const {
  T1 = this->state_dependencies_affine_term[t];
}

void Trajectory::get_control_dependencies_initial_state_projection(unsigned int t, Eigen::MatrixXd &Tx) const {
  Tx = this->control_dependencies_initial_state_projection[t];
}

void Trajectory::get_control_dependencies_terminal_state_projection(unsigned int t, Eigen::MatrixXd &Tz) const {
  Tz = this->control_dependencies_terminal_state_projection[t];
}

void Trajectory::get_control_dependencies_affine_term(unsigned int t, Eigen::VectorXd &T1) const {
  T1 = this->control_dependencies_affine_term[t];
}

void Trajectory::get_dynamics_mult_initial_state_feedback_term(unsigned int t, Eigen::MatrixXd &Tx) const {
  Tx = this->dynamics_mult_initial_state_feedback_terms[t];
}

void Trajectory::get_dynamics_mult_terminal_state_feedback_term(unsigned int t, Eigen::MatrixXd &Tz) const {
  Tz = this->dynamics_mult_terminal_state_feedback_terms[t];
}

void Trajectory::get_dynamics_mult_feedforward_term(unsigned int t, Eigen::VectorXd &T1) const {
  T1 = this->dynamics_mult_feedforward_terms[t];
}

void Trajectory::get_running_constraint_mult_initial_state_feedback_term(unsigned int t, Eigen::MatrixXd &Tx) const {
  Tx = this->running_constraint_mult_initial_state_feedback_terms[t];
}

void Trajectory::get_running_constraint_mult_terminal_state_feedback_term(unsigned int t, Eigen::MatrixXd &Tz) const {
  Tz = this->running_constraint_mult_terminal_state_feedback_terms[t];
}

void Trajectory::get_running_constraint_mult_feedforward_term(unsigned int t, Eigen::VectorXd &T1) const {
  T1 = this->running_constraint_mult_feedforward_terms[t];
}

void Trajectory::get_terminal_constraint_mult_initial_state_feedback_term(Eigen::MatrixXd &Tx) const {
  Tx = this->terminal_constraint_mult_initial_state_feedback_term;
}

void Trajectory::get_terminal_constraint_mult_terminal_state_feedback_term(Eigen::MatrixXd &Tz) const {
  Tz = this->terminal_constraint_mult_terminal_state_feedback_term;
}

void Trajectory::get_terminal_constraint_mult_feedforward_term(Eigen::VectorXd &T1) const {
  T1 = this->terminal_constraint_mult_feedforward_term;
}

}  // namespace trajectory
