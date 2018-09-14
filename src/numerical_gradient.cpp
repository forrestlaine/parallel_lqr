//
// Created by Forrest Laine on 7/24/18.
//

#include "numerical_gradient.h"

#include <iostream>

namespace numerical_gradient {

const double tol = 1e-8;

void numerical_gradient_first_input(const std::function<double(const Eigen::VectorXd *, const Eigen::VectorXd *)> *f,
                                    const Eigen::VectorXd *v1,
                                    const Eigen::VectorXd *v2,
                                    Eigen::VectorXd &g1) {
  const long n = (*v1).size();
  Eigen::MatrixXd I = tol*Eigen::MatrixXd::Identity(n, n);
  Eigen::VectorXd temp1;
  Eigen::VectorXd temp2;

  for (int i = 0; i < n; ++i) {
    temp1 = (*v1) + I.col(i);
    temp2 = (*v1) - I.col(i);
    g1[i] = ((*f)(&temp1, v2) - (*f)(&temp2, v2)) / (2.*tol);
  }
}

void numerical_gradient_second_input(const std::function<double(const Eigen::VectorXd *, const Eigen::VectorXd *)> *f,
                                     const Eigen::VectorXd *v1,
                                     const Eigen::VectorXd *v2,
                                     Eigen::VectorXd &g2) {
  const long n = (*v2).size();
  Eigen::MatrixXd I = tol*Eigen::MatrixXd::Identity(n, n);
  Eigen::VectorXd temp1;
  Eigen::VectorXd temp2;

  for (int i = 0; i < n; ++i) {
    temp1 = (*v2) + I.col(i);
    temp2 = (*v2) - I.col(i);
    g2[i] = ((*f)(v1, &temp1) - (*f)(v1, &temp2)) / (2.*tol);
  }
}

void numerical_gradient(const std::function<double(const Eigen::VectorXd *)> *f,
                        const Eigen::VectorXd *v,
                        Eigen::VectorXd &g) {
  const long n = (*v).size();
  Eigen::MatrixXd I = tol*Eigen::MatrixXd::Identity(n, n);
  Eigen::VectorXd temp1;
  Eigen::VectorXd temp2;

  for (int i = 0; i < n; ++i) {
    temp1 = (*v) + I.col(i);
    temp2 = (*v) - I.col(i);
    g[i] = ((*f)(&temp1) - (*f)(&temp2)) / (2.*tol);
  }
}

void numerical_jacobian_first_input(const std::function<void(const Eigen::VectorXd *, const Eigen::VectorXd *, Eigen::VectorXd &)> *f,
                                    const Eigen::VectorXd *v1,
                                    const Eigen::VectorXd *v2,
                                    Eigen::MatrixXd &J1) {
  const long n = (*v1).size();
  Eigen::MatrixXd I = tol*Eigen::MatrixXd::Identity(n, n);
  Eigen::VectorXd temp1, temp2, temp3, temp4;

  for (int i = 0; i < n; ++i) {
    temp1 = (*v1) + I.col(i);
    temp2 = (*v1) - I.col(i);
    (*f)(&temp1, v2, temp3);
    (*f)(&temp2, v2, temp4);
    J1.col(i) = (temp3 - temp4) / (2.*tol);
  }
}

void numerical_jacobian_second_input(const std::function<void(const Eigen::VectorXd *, const Eigen::VectorXd *, Eigen::VectorXd &)> *f,
                                     const Eigen::VectorXd *v1,
                                     const Eigen::VectorXd *v2,
                                     Eigen::MatrixXd &J2) {
  const long n = (*v2).size();
  Eigen::MatrixXd I = tol*Eigen::MatrixXd::Identity(n, n);
  Eigen::VectorXd temp1, temp2, temp3, temp4;
  for (int i = 0; i < n; ++i) {
    temp1 = (*v2) + I.col(i);
    temp2 = (*v2) - I.col(i);
    (*f)(v1, &temp1, temp3);
    (*f)(v1, &temp2, temp4);
    J2.col(i) = (temp3 - temp4) / (2.*tol);
  }
}

void numerical_jacobian_first_input(const std::function<void(const Eigen::VectorXd *,
                                                             const Eigen::VectorXd *,
                                                             int,
                                                             Eigen::VectorXd &)> *f,
                                    const Eigen::VectorXd *v1,
                                    const Eigen::VectorXd *v2,
                                    int t,
                                    Eigen::MatrixXd &J1){
  const long n = (*v1).size();
  Eigen::MatrixXd I = tol*Eigen::MatrixXd::Identity(n, n);
  Eigen::VectorXd temp1, temp2, temp3, temp4;
  for (int i = 0; i < n; ++i) {
    temp1 = (*v1) + I.col(i);
    temp2 = (*v1) - I.col(i);
    (*f)(&temp1, v2, t, temp3);
    (*f)(&temp2, v2, t, temp4);
    J1.col(i) = (temp3 - temp4) / (2.*tol);
  }
}

void numerical_jacobian_second_input(const std::function<void(const Eigen::VectorXd *,
                                                              const Eigen::VectorXd *,
                                                              int,
                                                              Eigen::VectorXd &)> *f,
                                     const Eigen::VectorXd *v1,
                                     const Eigen::VectorXd *v2,
                                     int t,
                                     Eigen::MatrixXd &J2) {
  const long n = (*v2).size();
  Eigen::MatrixXd I = tol*Eigen::MatrixXd::Identity(n, n);
  Eigen::VectorXd temp1, temp2, temp3, temp4;

  for (int i = 0; i < n; ++i) {
    temp1 = (*v2) + I.col(i);
    temp2 = (*v2) - I.col(i);
    (*f)(v1, &temp1, t, temp3);
    (*f)(v1, &temp2, t, temp4);
    J2.col(i) = (temp3 - temp4) / (2.*tol);
  }
}

void numerical_jacobian(const std::function<void(const Eigen::VectorXd *, Eigen::VectorXd &)> *f,
                        const Eigen::VectorXd *v,
                        Eigen::MatrixXd &J) {
  const long n = (*v).size();
  Eigen::MatrixXd I = tol*Eigen::MatrixXd::Identity(n, n);
  Eigen::VectorXd temp1, temp2, temp3, temp4;

  for (int i = 0; i < n; ++i) {
    temp1 = (*v) + I.col(i);
    temp2 = (*v) - I.col(i);
    (*f)(&temp1, temp3);
    (*f)(&temp2, temp4);
    J.col(i) = (temp3 - temp4) / (2.*tol);
  }
}

void numerical_hessian_first_first(const std::function<double(const Eigen::VectorXd *, const Eigen::VectorXd *)> *f,
                                   const Eigen::VectorXd *v1,
                                   const Eigen::VectorXd *v2,
                                   Eigen::MatrixXd &H11) {
  const long n = (*v1).size();
  Eigen::MatrixXd I = tol*Eigen::MatrixXd::Identity(n, n);
  Eigen::VectorXd temp1(n), temp2(n), temp3(n), temp4(n);

  for (int i = 0; i < n; ++i) {
    temp1 = (*v1) + I.col(i);
    temp2 = (*v1) - I.col(i);
    numerical_gradient_first_input(f, &temp1, v2, temp3);
    numerical_gradient_first_input(f, &temp2, v2, temp4);
    H11.col(i) = (temp3 - temp4) / (2. * tol);
  }
}

void numerical_hessian_second_first(const std::function<double(const Eigen::VectorXd *, const Eigen::VectorXd *)> *f,
                                    const Eigen::VectorXd *v1,
                                    const Eigen::VectorXd *v2,
                                    Eigen::MatrixXd &H21) {
  const long n = (*v1).size();
  const long m = (*v2).size();
  Eigen::MatrixXd I = tol*Eigen::MatrixXd::Identity(n, n);
  Eigen::VectorXd temp1(n), temp2(n), temp3(m), temp4(m);

  for (int i = 0; i < n; ++i) {
    temp1 = (*v1) + I.col(i);
    temp2 = (*v1) - I.col(i);
    numerical_gradient_second_input(f, &temp1, v2, temp3);
    numerical_gradient_second_input(f, &temp2, v2, temp4);

    H21.col(i) = (temp3 - temp4) / (2. * tol);
  }
}

void numerical_hessian_second_second(const std::function<double(const Eigen::VectorXd *, const Eigen::VectorXd *)> *f,
                                     const Eigen::VectorXd *v1,
                                     const Eigen::VectorXd *v2,
                                     Eigen::MatrixXd &H22) {

  const long m = (*v2).size();
  Eigen::MatrixXd I = tol*Eigen::MatrixXd::Identity(m, m);
  Eigen::VectorXd temp1(m), temp2(m), temp3(m), temp4(m);

  for (int i = 0; i < m; ++i) {
    temp1 = (*v2) + I.col(i);
    temp2 = (*v2) - I.col(i);
    numerical_gradient_second_input(f, v1, &temp1, temp3);
    numerical_gradient_second_input(f, v1, &temp2, temp4);

    H22.col(i) = (temp3 - temp4) / (2. * tol);
  }
}

void numerical_hessian(const std::function<double(const Eigen::VectorXd *)> *f,
                       const Eigen::VectorXd *v,
                       Eigen::MatrixXd &H) {
  const long n = (*v).size();
  Eigen::MatrixXd I = tol*Eigen::MatrixXd::Identity(n, n);
  Eigen::VectorXd temp1(n), temp2(n), temp3(n), temp4(n);

  for (int i = 0; i < n; ++i) {
    temp1 = (*v) + I.col(i);
    temp2 = (*v) - I.col(i);
    numerical_gradient(f, &temp1, temp3);
    numerical_gradient(f, &temp2, temp4);

    H.col(i) = (temp3 - temp4) / (2. * tol);
  }
}

}  //namespace numerical_gradient