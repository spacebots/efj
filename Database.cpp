#include <vector>
#include <string>
#include <fstream>
#include <iostream>
#include <boost/filesystem.hpp>
#include <Eigen/Core>
#include <Eigen/QR> //Non-Stable part of Eigen API
//#include <Eigen/Geometry>
//#include <Eigen/LU>
#include <QImage>
#include <QString>

#include "Database.h"

namespace bf = boost::filesystem;

efj::Database::Database(const std::string &dir) :
  _pixels(new Eigen::MatrixXi()) {
  load_pixels(dir);
  debug_print_pixels();

  _nImages = _pixels->cols();
  _features = _pixels->rows(); //number of pixels, column size
  _mean.resize(_features);

#pragma omp parallel for
  for (int i = 0; i < _features; i++) {
    double accumulated = 0;
#pragma omp parallel for
    for (int j = 0; j < _nImages; j++) {
      accumulated += (*_pixels)(i, j);
    }
    _mean(i) = accumulated / _nImages;
  }
  debug_print_mean();

  _centeredPixels.resize(_features, _nImages);

  //Creating centeredPixels Matrix
#pragma omp parallel for
  for (int i = 0; i < _pixels->cols(); i++) {
#pragma omp parallel for
    for (int j = 0; j < _pixels->rows(); j++) {
      double px = (*_pixels)(j, i);
      _centeredPixels(j, i) = std::abs(px - _mean(j));
    }
  }

  std::cerr << "pixels centered\n";
  debug_print_centeredPixels();

  initialize(_eigenfaces); //initialize matriz to zeroes
}

/**
 * in this directory, seach for file with given extension
 * place paths in vector
 */
void efj::Database::find_files(const bf::path & dir_path, const std::string & extension,
                               std::vector<QString> &files) {
  if (!bf::exists(dir_path))
    return;
  bf::directory_iterator end_itr; // default construction yields past-the-end
  for (bf::directory_iterator itr(dir_path); itr != end_itr; ++itr) {
    if (bf::is_directory(itr->status())) {
      find_files(itr->path(), extension, files);
    } else if (itr->path().extension() == extension) {
      //files.push_back(itr->leaf());  //only the last part
      files.push_back(itr->path().string().c_str());
    }
  }
}

//const std::string trainDatabasePath = "/afs/l2f.inesc-id.pt/home/ferreira/FaceRec/ImageVault/train";
bool efj::Database::load_pixels(const std::string &trainDatabasePath) {
  std::vector<QString> files;
  find_files(trainDatabasePath, ".jpg", files);
  if (files.size() == 0)
    return false;
  //std::cout << "FILES FOUND: " << files.size() << std::endl;

  QImage img(files[0]);
  //std::cout << "1st FILE: " << files[0].toStdString().c_str() << std::endl;
  //std::cout << "rows x cols: " << (int)(img.height() * img.width()) << " x " << (int)files.size() << std::endl;
  _pixels->resize((int)(img.height() * img.width()), (int)files.size());

  // Construction of 2D matrix from 1D image vectors
#pragma omp parallel for
  for (size_t fx = 0; fx < files.size(); fx++) {
    //std::cout << "FILE: " << files[fx].toStdString().c_str() << std::endl;
    QImage img(QString(files[fx]));
    //std::cout << "VECTOR SIZE: " << (int)(img.height() * img.width()) << std::endl;
    Eigen::VectorXi pixs(img.height() * img.width());
#pragma omp parallel for
    for (int rx = 0; rx < img.height(); rx++) {
      int offset = rx * img.width();
#pragma omp parallel for
      for (int cx = 0; cx < img.width(); cx++) {
        pixs[offset + cx] = qGray(img.pixel(cx, rx));
      }
    }
    _pixels->col(fx) = pixs;

  }

  return true;
}

void efj::Database::compute_eigenfaces() {
  Eigen::MatrixXd covMatrix(_nImages, _nImages);
  covMatrix = _centeredPixels.transpose() * _centeredPixels;

  std::cerr << "covarianceMatrix done \n";
  std::cout << "COV:\n";
  for (int i = 0; i < covMatrix.rows(); i++) {
    for (int j = 0; j < covMatrix.cols(); j++) {
      std::cout << covMatrix(i, j) << " ";
    }
    std::cout << std::endl;
  }

  Eigen::EigenSolver<Eigen::MatrixXd> *solver2 = new Eigen::EigenSolver<Eigen::MatrixXd>(covMatrix);
  std::cerr << "eigensolver created \n";

  //UNUSED:
  Eigen::EigenSolver<Eigen::MatrixXd>::EigenvalueType eigenValues = solver2->eigenvalues();
  //UNUSED:
  std::cout << eigenValues;
  std::cerr << "eigenvalues done \n";

  Eigen::EigenSolver<Eigen::MatrixXd>::EigenvectorsType eigenVectors = solver2->eigenvectors();
  std::cerr << "eigenvectors done: " << _nImages << " images @ " << _features << " features\n";

#pragma omp parallel for
  for (int i = 0; i < _nImages; i++) {//i = col
#pragma omp parallel for
    for (int j = 0; j < _nImages; j++) { // j = row
      _eigenfaces.col(i) += _centeredPixels.col(j) * eigenVectors(j, i).real();
    }
  }

  std::cerr << "saving" << "\n";
  write("matrix2.out");
  std::cerr << "all done" << std::endl;

}

void efj::Database::write(const char *output_file) {
  std::ofstream mos(output_file, std::ios::binary);
  // store pixels
  int rows = _pixels->rows();
  int cols = _pixels->cols();
  mos.write((const char *)&rows, sizeof(int));
  mos.write((const char *)&cols, sizeof(int));
  for (int i = 0; i < _pixels->rows(); i++) {
    for (int j = 0; j < _pixels->cols(); j++) {
      int p = (*_pixels)(i, j);
      mos.write((const char *)&p, sizeof(int));
    }
  }
  // mean vector (face)
  int ms = _mean.size();
  mos.write((const char *)&ms, sizeof(int));
  for (int i = 0; i < _mean.size(); i++) {
    double m = _mean(i);
    mos.write((const char *)&m, sizeof(double));
  }
  // store eigenfaces
  rows = _eigenfaces.rows();
  cols = _eigenfaces.cols();
  mos.write((const char *)&rows, sizeof(int));
  mos.write((const char *)&cols, sizeof(int));
  for (int i = 0; i < _eigenfaces.rows(); i++) {
    for (int j = 0; j < _eigenfaces.cols(); j++) {
      double ef = _eigenfaces(i, j);
      mos.write((const char *)&ef, sizeof(double));
    }
  }
  mos.close();
}

void efj::Database::debug_print_pixels() {
#ifdef DEBUG
  std::cout << "PIXELS:\n";
  for (int i = 0; i < _pixels->rows(); i++) {
    for (int j = 0; j < _pixels->cols(); j++) {
      std::cout << (*_pixels)(i, j) << " ";
    }
    std::cout << std::endl;
  }
#endif
}

void efj::Database::debug_print_mean() {
#ifdef DEBUG
  std::cout << "P2 640 480 255 \n";
  std::cout << mean << std::endl;
#endif
}

void efj::Database::debug_print_centeredPixels() {
#ifdef DEBUG
  std::cout << "PIX:\n";
  for (int i = 0; i < centeredPixels.rows(); i++) {
    for (int j = 0; j < centeredPixels.cols(); j++) {
      std::cout << centeredPixels(i, j) << " ";
    }
    std::cout << std::endl;
  }

#endif
}
