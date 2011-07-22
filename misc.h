//$Id: misc.h,v 1.5 2011/07/22 14:32:44 ferreira Exp $

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

//$Log: misc.h,v $
//Revision 1.5  2011/07/22 14:32:44  ferreira
//Added Copyright comment
//

#ifndef __EFJ_MISC_H__
#define __EFJ_MISC_H__

#include <Eigen/Core>
#include <QImage>

namespace efj {
  template<typename MatrixType>
  inline void initialize(int rows, int cols, MatrixType &matrix) {
#pragma omp parallel for
    for (int i = 0; i < rows; i++) {
#pragma omp parallel for
      for (int j = 0; j < cols; j++) {
        matrix(i, j) = 0;
      }
    }
  }
} // namespace efj

namespace efj {
  template<typename MatrixType>
  inline void initialize(MatrixType &matrix) {
    initialize(matrix.rows(), matrix.cols(), matrix);
  }
} // namespace efj

namespace efj {
  template<typename VectorType>
  inline void initialize(int size, VectorType &vector) {
#pragma omp parallel for
    for (int j = 0; j < size; j++) {
      vector[j] = 0;
    }
  }
} // namespace efj

#endif
