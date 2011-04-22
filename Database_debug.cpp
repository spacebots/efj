#include <iostream>
#include <Eigen/Core>
#include "Database.h"

void efj::Database::debug_print_pixels() {
#ifdef DEBUG
  std::cout << "PIXELS:\n";
  for (int i = 0; i < _pixels.rows(); i++) {
    for (int j = 0; j < _pixels.cols(); j++) {
      std::cout << _pixels(i, j) << " ";
    }
    std::cout << std::endl;
  }
#endif
}

void efj::Database::debug_print_mean() {
#ifdef DEBUG
  std::cout << "P2 640 480 255 \n";
  std::cout << _mean << std::endl;
#endif
}

void efj::Database::debug_print_centeredPixels() {
#ifdef DEBUG
  std::cout << "PIX:\n";
  for (int i = 0; i < _centeredPixels.rows(); i++) {
    for (int j = 0; j < _centeredPixels.cols(); j++) {
      std::cout << _centeredPixels(i, j) << " ";
    }
    std::cout << std::endl;
  }

#endif
}

void efj::Database::debug_print_eigenvectors(const eigenvalue_type &eigenValues,
                                             const eigenvectors_type &eigenVectors) {
#if DEBUG
  std::cout << "Eigenvalues:" << std::endl;
  std::cout << eigenValues;
  std::cout << "Eigenvectors:" << std::endl;
  std::cout << eigenVectors;
  std::cout << "eigenvectors done: " << _nImages << " images @ " << _nPixels << " features\n";
#endif
}

void efj::Database::debug_print_covariance_matrix(const Eigen::MatrixXd &covMatrix) {
#ifdef DEBUG
  std::cerr << "Covariance Matrix:\n";
  for (int i = 0; i < covMatrix.rows(); i++) {
    for (int j = 0; j < covMatrix.cols(); j++) {
      std::cout << covMatrix(i, j) << " ";
    }
    std::cout << std::endl;
  }
#endif
}
