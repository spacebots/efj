//$Id: Database_debug.cpp,v 1.3 2011/07/22 14:32:44 ferreira Exp $

// Copyright (C) 2008-2011 INESC ID Lisboa.
//
// This program is free software; you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation; either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program; if not, write to the Free Software
// Foundation, Inc., 59 Temple Place - Suite 330, Boston, MA 02111-1307, USA.
//

//$Log: Database_debug.cpp,v $
//Revision 1.3  2011/07/22 14:32:44  ferreira
//Added Copyright comment
//

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
