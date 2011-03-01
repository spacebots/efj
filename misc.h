#ifndef __EFJ_MISC_H__
#define __EFJ_MISC_H__

#include <Eigen/Core>
#include <QImage>

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

template<typename MatrixType>
inline void initialize(MatrixType &matrix) {
  initialize(matrix.rows(), matrix.cols(), matrix);
}

template<typename VectorType>
inline void initialize(int size, VectorType &vector) {
#pragma omp parallel for
  for (int j = 0; j < size; j++) {
    vector[j] = 0;
  }
}

inline void readSingleFile(std::string pathToFile, Eigen::VectorXi &testImg) {
  QImage img((QString)pathToFile.c_str());
  testImg.resize((int)(img.height() * (int)(img.width())));
#pragma omp parallel for
  for (int rx = 0; rx < img.height(); rx++) {
    int offset = rx * img.width();
#pragma omp parallel for
    for (int cx = 0; cx < img.width(); cx++) {
      testImg[offset + cx] = qGray(img.pixel(cx, rx));
    }
  }
}

#endif
