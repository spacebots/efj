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
