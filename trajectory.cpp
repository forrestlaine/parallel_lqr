//
// Created by Forrest Laine on 6/19/18.
//

#include "trajectory.h"

#include <iostream>
#include <vector>

namespace trajectory {

void Trajectory::populate_derivative_terms() {
  // Evaluate derivatives, numerically or analytically, and populate corresponding terms.
}

void Trajectory::compute_feedback_policies() {
  const unsigned int n = this->state_dimension;
  const unsigned int m = this->control_dimension;
  const unsigned int l = this->terminal_constraint_dimension;
//  const unsigned int j = this->initial_constraint_dimension;

  Eigen::MatrixXd Mxx = Eigen::MatrixXd::Zero(n, n);
  Eigen::MatrixXd Muu = Eigen::MatrixXd::Zero(m, m);
  Eigen::MatrixXd Mzz = Eigen::MatrixXd::Zero(l, l);
  Eigen::MatrixXd Mux = Eigen::MatrixXd::Zero(m, n);
  Eigen::MatrixXd Mzx = Eigen::MatrixXd::Zero(l, n);
  Eigen::MatrixXd Mzu = Eigen::MatrixXd::Zero(l, m);
  Eigen::VectorXd Mx1 = Eigen::MatrixXd::Zero(n, 1);
  Eigen::VectorXd Mu1 = Eigen::MatrixXd::Zero(m, 1);
  Eigen::VectorXd Mz1 = Eigen::MatrixXd::Zero(l, 1);
  Eigen::MatrixXd M11 = Eigen::MatrixXd::Zero(1, 1);

  Eigen::MatrixXd Vxx = (this->terminal_cost_hessians_state_state).eval();
  Eigen::MatrixXd Vzz = Eigen::MatrixXd::Zero(l, l);
  Eigen::MatrixXd Vzx = Eigen::MatrixXd::Zero(l, n);
  Eigen::VectorXd Vx1 = (this->terminal_cost_gradient_state).eval();
  Eigen::VectorXd Vz1 = Eigen::MatrixXd::Zero(l, 1);
  Eigen::MatrixXd V11 = Eigen::MatrixXd::Zero(1, 1);

  this->cost_to_go_hessians_state_state[this->trajectory_length - 1] = this->terminal_cost_hessians_state_state.eval();
  this->cost_to_go_gradients_state[this->trajectory_length - 1] = this->terminal_cost_gradient_state.eval();

  this->num_residual_constraints_to_go[this->trajectory_length - 1] = this->terminal_constraint_dimension;

  Eigen::MatrixXd Gx = this->terminal_constraint_jacobian_state.topLeftCorner(l, n).eval();
  Eigen::MatrixXd Gz = this->terminal_constraint_jacobian_terminal_projection.topLeftCorner(l, l).eval();
  Eigen::VectorXd G1 = this->terminal_constraint_affine_term.head(l).eval();

  Eigen::MatrixXd Nx = Eigen::MatrixXd::Zero(n, n);
  Eigen::MatrixXd Nu = Eigen::MatrixXd::Zero(n, m);
  Eigen::MatrixXd Nz = Eigen::MatrixXd::Zero(n, l);
  Eigen::VectorXd N1 = Eigen::MatrixXd::Zero(n, 1);

  Eigen::MatrixXd *constraint_jacobian_state_ptr;
  Eigen::MatrixXd *constraint_jacobian_control_ptr;
  Eigen::VectorXd *constraint_affine_term_ptr;

  for (int t = this->trajectory_length - 2; t >= 0; --t) {
    this->num_residual_constraints_to_go[t] =
        this->num_residual_constraints_to_go[t + 1] - m + this->num_active_constraints[t];

    constraint_jacobian_state_ptr = &this->running_constraint_jacobians_state[t];
    constraint_jacobian_control_ptr = &this->running_constraint_jacobians_control[t];
    constraint_affine_term_ptr = &this->running_constraint_affine_terms[t];
    this->perform_constrained_dynamic_programming_backup(&this->hamiltonian_hessians_state_state[t],
                                                         &this->hamiltonian_hessians_control_control[t],
                                                         &this->hamiltonian_hessians_control_state[t],
                                                         &this->hamiltonian_gradients_state[t],
                                                         &this->hamiltonian_gradients_control[t],
                                                         &this->dynamics_jacobians_state[t],
                                                         &this->dynamics_jacobians_control[t],
                                                         &this->dynamics_affine_terms[t],
                                                         constraint_jacobian_state_ptr,
                                                         constraint_jacobian_control_ptr,
                                                         constraint_affine_term_ptr,
                                                         this->num_active_constraints[t],
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
//  this->residual_initial_constraint_jacobian_initial_state.topRows(n - j) = Gx;
//  this->residual_initial_constraint_jacobian_terminal_projection.topLeftCorner(n - j, l) = Gz;
//  this->residual_initial_constraint_affine_term.head(n - j) = G1;
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
  Mz1 = Vzx * (*A1) + Vz1;
  Mxx = (*Ax).transpose() * Vxx * (*Ax) + *Qxx;
  Mux = (*Au).transpose() * Vxx * (*Ax) + *Qux;
  Muu = (*Au).transpose() * Vxx * (*Au) + *Quu;
  Mzx = Vzx * (*Ax);
  Mzu = Vzx * (*Au);
  Mzz = Vzz;
  //

  long constraint_size = num_active_constraints + G1.size();
  N1.resize(constraint_size);
  Nx.resize(constraint_size, this->state_dimension);
  Nu.resize(constraint_size, this->control_dimension);
  Nz.resize(constraint_size, this->terminal_constraint_dimension);

  N1 << (*D1).head(num_active_constraints), Gx * (*A1) + G1;
  Nx << (*Dx).topRows(num_active_constraints), Gx * (*Ax);
  Nu << (*Du).topRows(num_active_constraints), Gx * (*Au);
  Nz << Eigen::MatrixXd::Zero(num_active_constraints, this->terminal_constraint_dimension), Gz;

  if (constraint_size > 0) {
    Eigen::BDCSVD<Eigen::MatrixXd> svd(Nu, Eigen::ComputeFullU | Eigen::ComputeFullV);
    long constraint_rank = svd.rank();
    if (constraint_rank == 0) {
      Eigen::LDLT<Eigen::MatrixXd> decMuu(Muu);
      Lx = -decMuu.solve(Mux);
      Lz = -decMuu.solve(Mzu.transpose());
      L1 = -decMuu.solve(Mu1);
      Gx = Nx;
      Gz = Nz;
      G1 = N1;
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
      Nz = (U_Nu_t * Nz).eval();

      Gx.resize(constraint_size - constraint_rank, this->state_dimension);
      Gz.resize(constraint_size - constraint_rank, this->terminal_constraint_dimension);
      G1.resize(constraint_size - constraint_rank, 1);

      Gx << Nx.bottomRows(constraint_size - constraint_rank);
      Gz << Nz.bottomRows(constraint_size - constraint_rank);
      G1 << N1.tail(constraint_size - constraint_rank);

      N1.conservativeResize(constraint_rank);
      Nx.conservativeResize(constraint_rank, Eigen::NoChange);
      Nu.conservativeResize(constraint_rank, Eigen::NoChange);
      Nz.conservativeResize(constraint_rank, Eigen::NoChange);

      const Eigen::MatrixXd VlSi = -V_Nu_l * Si_Nu;
      //std::cout<<"Constraint rank: " << constraint_rank<<std::endl;
      if (constraint_rank >= this->control_dimension) { // constraint_rank == control_dimension
        Lx = VlSi * Nx;
        Lz = VlSi * Nz;
        L1 = VlSi * N1;
      } else { // 0 < constraint_rank < control_dimension
        Eigen::VectorXd T1 = VlSi * N1;
        Eigen::MatrixXd Tx = VlSi * Nx;
        Eigen::MatrixXd Tz = VlSi * Nz;
        Eigen::MatrixXd Tu = V_Nu_r;

        // Order of these operations matter
        Eigen::MatrixXd temp1, temp2, temp3;
        temp1 = Mu1 + Muu * T1;
        temp2 = Mux + Muu * Tx;
        temp3 = Muu * Tu;

        Eigen::MatrixXd tmuu, tmu1, tmux, tmzu;

        tmu1 = Tu.transpose() * temp1;
        tmux = Tu.transpose() * temp2;
        tmzu = Tz.transpose() * temp3 + Mzu * Tu;
        tmuu = Tu.transpose() * temp3;

        Eigen::LDLT<Eigen::MatrixXd> decMuu(tmuu);

        temp1 = -decMuu.solve(tmu1);
        temp2 = -decMuu.solve(tmux);
        temp3 = -decMuu.solve(tmzu.transpose());

        L1 = T1 + V_Nu_r * temp1;
        Lx = Tx + V_Nu_r * temp2;
        Lz = Tz + V_Nu_r * temp3;
      }
    }
  } else {
    Eigen::LDLT<Eigen::MatrixXd> decMuu(Muu);
    Lx = -decMuu.solve(Mux);
    Lz = -decMuu.solve(Mzu.transpose());
    L1 = -decMuu.solve(Mu1);
    Gx.resize(0, this->state_dimension);
    Gz.resize(0, this->terminal_constraint_dimension);
    G1.resize(0, 1);
  }

  const Eigen::VectorXd temp1 = Mu1 + Muu * L1;
  const Eigen::MatrixXd temp2 = Mux + Muu * Lx;

  V11 = M11 + Mu1.transpose() * L1 + L1.transpose() * temp1;
  Vx1 = Mx1 + Mux.transpose() * L1 + Lx.transpose() * temp1;
  Vz1 = Mz1 + Mzu * L1 + Lz.transpose() * temp1;
  Vxx = Mxx + Mux.transpose() * Lx + Lx.transpose() * temp2;
  Vzx = Mzx + Mzu * Lx + Lz.transpose() * temp2;
  Vzz = Mzz + Mzu * Lz + Lz.transpose() * (Mzu.transpose() + Muu * Lz);
}

void Trajectory::compute_multipliers() {
  const unsigned int n = this->state_dimension;
  const unsigned int l = this->terminal_constraint_dimension;
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
  Ez = U * this->state_dependencies_terminal_state_projection[T - 1].leftCols(l);
  E1 = U * this->state_dependencies_affine_term[T - 1] + H * q;
  Eigen::PartialPivLU<Eigen::MatrixXd> decomp;
  if (l > 0) {
    decomp.compute(-Omega);
    this->terminal_constraint_mult_dynamics_mult_feedback_term.topRows(l) = decomp.solve(H);
    this->terminal_constraint_mult_initial_state_feedback_term.topRows(l) = decomp.solve(Ex);
    this->terminal_constraint_mult_terminal_state_feedback_term.topLeftCorner(l, l) = decomp.solve(Ez);
    this->terminal_constraint_mult_feedforward_term.head(l) = decomp.solve(E1);
  } else {
    this->terminal_constraint_mult_dynamics_mult_feedback_term.topRows(l) = Eigen::MatrixXd::Zero(l, n);
    this->terminal_constraint_mult_initial_state_feedback_term.topRows(l) = Eigen::MatrixXd::Zero(l, n);
    this->terminal_constraint_mult_terminal_state_feedback_term.topLeftCorner(l, l) = Eigen::MatrixXd::Zero(l, l);
    this->terminal_constraint_mult_feedforward_term.head(l) = Eigen::VectorXd::Zero(l);
  }

  for (int t = T - 2; t >= 0; --t) {
    if (this->num_residual_constraints_to_go[t+1] > 0 or this->num_active_constraints[t] > 0) {

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

      Dz = V * this->state_dependencies_terminal_state_projection[t].leftCols(l) +
          Y * this->control_dependencies_terminal_state_projection[t].leftCols(l) +
          Qnext * this->state_dependencies_terminal_state_projection[t + 1].leftCols(l);

      D1 = V * this->state_dependencies_affine_term[t] +
          Y * this->control_dependencies_affine_term[t] +
          Qnext * this->state_dependencies_affine_term[t + 1] + qnext - A * q - B * r;

      Ex = U * this->state_dependencies_initial_state_projection[t] +
          W * this->control_dependencies_initial_state_projection[t];
      Ez = U * this->state_dependencies_terminal_state_projection[t].leftCols(l) +
          W * this->control_dependencies_terminal_state_projection[t].leftCols(l);
      E1 = U * this->state_dependencies_affine_term[t] +
          W * this->control_dependencies_affine_term[t] + H * q + G * r;

      LHS.resize(j + n, j + n);
      Plam.resize(j + n, n);
      Pproj.resize(j + n, l + n);
      P1.resize(j + n);

      if ((unsigned int) t < T - 2) {
        Sigma += (Hnext.transpose() * this->running_constraint_mult_dynamics_mult_feedback_terms[t + 1].topRows(nextj)
            - Anext.transpose() * this->dynamics_mult_dynamics_mult_feedback_terms[t + 2]).eval();
        Dx += (Hnext.transpose() * this->running_constraint_mult_initial_state_feedback_terms[t + 1].topRows(nextj)
            - Anext.transpose() * this->dynamics_mult_initial_state_feedback_terms[t + 2]
            + Snext.transpose() * this->control_dependencies_initial_state_projection[t + 1]).eval();
        Dz += (Hnext.transpose()
            * this->running_constraint_mult_terminal_state_feedback_terms[t + 1].topLeftCorner(nextj, l)
            - Anext.transpose() * this->dynamics_mult_terminal_state_feedback_terms[t + 2].leftCols(l)
            + Snext.transpose() * this->control_dependencies_terminal_state_projection[t + 1].leftCols(l)).eval();
        D1 += (Hnext.transpose() * this->running_constraint_mult_feedforward_terms[t + 1].topRows(nextj)
            - Anext.transpose() * this->dynamics_mult_feedforward_terms[t + 2]
            + Snext.transpose() * this->control_dependencies_affine_term[t + 1]).eval();
      } else {
        Sigma += (Hnext.transpose() * this->terminal_constraint_mult_dynamics_mult_feedback_term.topRows(l)).eval();
        Dx += (Hnext.transpose() * this->terminal_constraint_mult_initial_state_feedback_term.topRows(l)).eval();
        Dz +=
            (Hnext.transpose() * this->terminal_constraint_mult_terminal_state_feedback_term.topLeftCorner(l, l)).eval();
        D1 += (Hnext.transpose() * this->terminal_constraint_mult_feedforward_term.head(l)).eval();
      }
      Pproj.topLeftCorner(j, n) = Ex; 
      Pproj.topRightCorner(j, l) = Ez;
      Pproj.bottomLeftCorner(n, n) = Dx;
      Pproj.bottomRightCorner(n, l) = Dz;
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
      this->running_constraint_mult_terminal_state_feedback_terms[t].topLeftCorner(j, l) = Pproj.block(0, n, j, l);
      this->running_constraint_mult_dynamics_mult_feedback_terms[t].topRows(j) = Plam.topRows(j);
      this->running_constraint_mult_feedforward_terms[t].head(j) = P1.head(j);
      this->dynamics_mult_initial_state_feedback_terms[t + 1] = Pproj.block(j, 0, n, n);
      this->dynamics_mult_terminal_state_feedback_terms[t + 1].leftCols(l) = Pproj.block(j, n, n, l);
      this->dynamics_mult_dynamics_mult_feedback_terms[t + 1] = Plam.bottomRows(n);
      this->dynamics_mult_feedforward_terms[t + 1] = P1.tail(n);
    } else {
      Eigen::MatrixXd Vxx, Vzx;
      Eigen::VectorXd Vx1;
      Vxx = this->cost_to_go_hessians_state_state[t+1];
      Vzx = this->cost_to_go_hessians_terminal_state[t+1];
      Vx1 = this->cost_to_go_gradients_state[t+1];
      this->dynamics_mult_initial_state_feedback_terms[t+1] = -(Vxx * this->state_dependencies_initial_state_projection[t+1]);
      this->dynamics_mult_terminal_state_feedback_terms[t+1] = -(Vzx.transpose() + Vxx * this->state_dependencies_terminal_state_projection[t+1]);
      this->dynamics_mult_feedforward_terms[t+1] = -(Vx1 + Vxx * this->state_dependencies_affine_term[t+1]);
    }
  }
  if (this->num_residual_constraints_to_go[0] > 0) {
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
        S.transpose() * this->control_dependencies_terminal_state_projection[0].leftCols(l) +
        H.transpose() * this->running_constraint_mult_terminal_state_feedback_terms[0].topLeftCorner(j, l)
        - A.transpose() * this->dynamics_mult_terminal_state_feedback_terms[1].leftCols(l);
    D1 = Q * this->state_dependencies_affine_term[0] + S.transpose() * this->control_dependencies_affine_term[0] + q +
        H.transpose() * this->running_constraint_mult_feedforward_terms[0].head(j)
        - A.transpose() * this->dynamics_mult_feedforward_terms[1];
    Plam = Eigen::MatrixXd::Identity(n, n)
        + H.transpose() * this->running_constraint_mult_dynamics_mult_feedback_terms[0].topRows(j)
        - A.transpose() * this->dynamics_mult_dynamics_mult_feedback_terms[1];
    decomp.compute(-Plam);
    this->dynamics_mult_initial_state_feedback_terms[0] = decomp.solve(Dx);
    this->dynamics_mult_terminal_state_feedback_terms[0].leftCols(l) = decomp.solve(Dz);
    this->dynamics_mult_feedforward_terms[0] = decomp.solve(D1);
  } else {
    Eigen::MatrixXd Vxx, Vzx;
    Eigen::VectorXd Vx1;
    Vxx = this->cost_to_go_hessians_state_state[0];
    Vzx = this->cost_to_go_hessians_terminal_state[0];
    Vx1 = this->cost_to_go_gradients_state[0];
    this->dynamics_mult_initial_state_feedback_terms[0] = -(Vxx * this->state_dependencies_initial_state_projection[0]);
    this->dynamics_mult_terminal_state_feedback_terms[0] = -(Vzx.transpose() + Vxx * this->state_dependencies_terminal_state_projection[0]);
    this->dynamics_mult_feedforward_terms[0] = -(Vx1 + Vxx * this->state_dependencies_affine_term[0]);
  };

  // Forward solve for Multiplier dependencies
  for (unsigned int t = 0; t < T - 1; ++t) {
    j = (t < T - 1) ? this->num_active_constraints[t] : l;
    if (this->num_residual_constraints_to_go[t+1] > 0 or this->num_active_constraints[t] > 0) {
      this->running_constraint_mult_initial_state_feedback_terms[t].topRows(j) +=
          (this->running_constraint_mult_dynamics_mult_feedback_terms[t].topRows(j)
              * this->dynamics_mult_initial_state_feedback_terms[t]).eval();
      this->running_constraint_mult_terminal_state_feedback_terms[t].topLeftCorner(j, l) +=
          (this->running_constraint_mult_dynamics_mult_feedback_terms[t].topRows(j)
              * this->dynamics_mult_terminal_state_feedback_terms[t].leftCols(l)).eval();
      this->running_constraint_mult_feedforward_terms[t].topRows(j) +=
          (this->running_constraint_mult_dynamics_mult_feedback_terms[t].topRows(j)
              * this->dynamics_mult_feedforward_terms[t]).eval();
      this->dynamics_mult_initial_state_feedback_terms[t + 1] +=
          (this->dynamics_mult_dynamics_mult_feedback_terms[t + 1]
              * this->dynamics_mult_initial_state_feedback_terms[t]).eval();
      this->dynamics_mult_terminal_state_feedback_terms[t + 1].leftCols(l) +=
          (this->dynamics_mult_dynamics_mult_feedback_terms[t + 1]
              * this->dynamics_mult_terminal_state_feedback_terms[t].leftCols(l)).eval();
      this->dynamics_mult_feedforward_terms[t + 1] +=
          (this->dynamics_mult_dynamics_mult_feedback_terms[t + 1] * this->dynamics_mult_feedforward_terms[t]).eval();
    }
  }
  this->terminal_constraint_mult_initial_state_feedback_term.topRows(l) +=
      (this->terminal_constraint_mult_dynamics_mult_feedback_term.topRows(l)
          * this->dynamics_mult_initial_state_feedback_terms[T - 1]).eval();
  this->terminal_constraint_mult_terminal_state_feedback_term.topLeftCorner(l, l) +=
      (this->terminal_constraint_mult_dynamics_mult_feedback_term.topRows(l)
          * this->dynamics_mult_terminal_state_feedback_terms[T - 1].leftCols(l)).eval();
  this->terminal_constraint_mult_feedforward_term.head(l) +=
      (this->terminal_constraint_mult_dynamics_mult_feedback_term.topRows(l)
          * this->dynamics_mult_feedforward_terms[T - 1]).eval();
}

void Trajectory::compute_state_control_dependencies() {
  const unsigned int n = this->state_dimension;
  const unsigned int l = this->terminal_constraint_dimension;
  const unsigned int T = this->trajectory_length;

  Eigen::PartialPivLU<Eigen::MatrixXd> dec(this->initial_constraint_jacobian_state);
  this->state_dependencies_initial_state_projection[0] = -dec.solve(Eigen::MatrixXd::Identity(n, n));
  this->state_dependencies_affine_term[0] = Eigen::VectorXd::Zero(n);
  this->state_dependencies_terminal_state_projection[0] = Eigen::MatrixXd::Zero(n, l);

  for (unsigned int t = 0; t < T - 1; ++t) {
    // Create shortcuts for terms
    const auto &Ax = this->dynamics_jacobians_state[t];
    const auto &Au = this->dynamics_jacobians_control[t];
    const auto &A1 = this->dynamics_affine_terms[t];
    const auto &Kx = this->current_state_feedback_matrices[t];
    const auto &Kz = this->terminal_state_feedback_matrices[t].leftCols(l);
    const auto &K1 = this->feedforward_controls[t];
    const auto &Mx = this->state_dependencies_initial_state_projection[t];
    const auto &Mz = this->state_dependencies_terminal_state_projection[t].leftCols(l);
    const auto &M1 = this->state_dependencies_affine_term[t];
    // Recursively update control dependencies
    this->control_dependencies_initial_state_projection[t] = Kx * Mx;
    this->control_dependencies_terminal_state_projection[t].leftCols(l) = Kx * Mz + Kz;
    this->control_dependencies_affine_term[t] = Kx * M1 + K1;
    // Recursively update state dependencies
    this->state_dependencies_initial_state_projection[t + 1] =
        Ax * Mx + Au * this->control_dependencies_initial_state_projection[t];
    this->state_dependencies_terminal_state_projection[t + 1].leftCols(l) =
        Ax * Mz + Au * this->control_dependencies_terminal_state_projection[t].leftCols(l);
    this->state_dependencies_affine_term[t + 1] = Ax * M1 + Au * this->control_dependencies_affine_term[t] + A1;
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
