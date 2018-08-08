//
// Created by Forrest Laine on 7/24/18.
//

#ifndef MOTORBOAT_NUMERICAL_GRADIENT_H
#define MOTORBOAT_NUMERICAL_GRADIENT_H

#include "eigen3/Eigen/Dense"

namespace numerical_gradient {

void numerical_gradient_first_input(const std::function<double(const Eigen::VectorXd *, const Eigen::VectorXd *)> *f,
                                    const Eigen::VectorXd *v1,
                                    const Eigen::VectorXd *v2,
                                    Eigen::VectorXd &g1);

void numerical_gradient_second_input(const std::function<double(const Eigen::VectorXd *, const Eigen::VectorXd *)> *f,
                                     const Eigen::VectorXd *v1,
                                     const Eigen::VectorXd *v2,
                                     Eigen::VectorXd &g2);

void numerical_gradient(const std::function<double(const Eigen::VectorXd *)> *f,
                        const Eigen::VectorXd *v,
                        Eigen::VectorXd &g);

void numerical_jacobian_first_input(const std::function<void(const Eigen::VectorXd *,
                                                             const Eigen::VectorXd *,
                                                             Eigen::VectorXd &)> *f,
                                    const Eigen::VectorXd *v1,
                                    const Eigen::VectorXd *v2,
                                    Eigen::MatrixXd &J1);

void numerical_jacobian_second_input(const std::function<void(const Eigen::VectorXd *,
                                                              const Eigen::VectorXd *,
                                                              Eigen::VectorXd &)> *f,
                                     const Eigen::VectorXd *v1,
                                     const Eigen::VectorXd *v2,
                                     Eigen::MatrixXd &J2);

void numerical_jacobian_first_input(const std::function<void(const Eigen::VectorXd *,
                                                             const Eigen::VectorXd *,
                                                             int,
                                                             Eigen::VectorXd &)> *f,
                                    const Eigen::VectorXd *v1,
                                    const Eigen::VectorXd *v2,
                                    int t,
                                    Eigen::MatrixXd &J1);

void numerical_jacobian_second_input(const std::function<void(const Eigen::VectorXd *,
                                                              const Eigen::VectorXd *,
                                                              int,
                                                              Eigen::VectorXd &)> *f,
                                     const Eigen::VectorXd *v1,
                                     const Eigen::VectorXd *v2,
                                     int t,
                                     Eigen::MatrixXd &J2);

void numerical_jacobian(const std::function<void(const Eigen::VectorXd *, Eigen::VectorXd &)> *f,
                        const Eigen::VectorXd *v,
                        Eigen::MatrixXd &J);

void numerical_hessian_first_first(const std::function<double(const Eigen::VectorXd *, const Eigen::VectorXd *)> *f,
                                   const Eigen::VectorXd *v1,
                                   const Eigen::VectorXd *v2,
                                   Eigen::MatrixXd &H11);

void numerical_hessian_second_first(const std::function<double(const Eigen::VectorXd *, const Eigen::VectorXd *)> *f,
                                    const Eigen::VectorXd *v1,
                                    const Eigen::VectorXd *v2,
                                    Eigen::MatrixXd &H21);

void numerical_hessian_second_second(const std::function<double(const Eigen::VectorXd *, const Eigen::VectorXd *)> *f,
                                     const Eigen::VectorXd *v1,
                                     const Eigen::VectorXd *v2,
                                     Eigen::MatrixXd &H22);

void numerical_hessian(const std::function<double(const Eigen::VectorXd *)> *f,
                       const Eigen::VectorXd *v,
                       Eigen::MatrixXd &H);

}  //namespace numerical_gradient

#endif //MOTORBOAT_NUMERICAL_GRADIENT_H
