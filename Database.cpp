#include <vector>
#include <string>
#include <utility>
#include <fstream>
#include <iostream>
#include <boost/filesystem.hpp>
#include <Eigen/Core>
#include <QImage>
#include <QString>

#include "Database.h"

namespace bf = boost::filesystem;

void efj::Database::readSingleFile(QString pathToFile, Eigen::VectorXd &testImg) {
  QImage img(pathToFile);
  testImg.resize((int)(img.height() * (int)(img.width())));
#pragma omp parallel for
  for (int rx = 0; rx < img.height(); rx++) {
    int offset = rx * img.width();
#pragma omp parallel for
    for (int cx = 0; cx < img.width(); cx++) {
      //DAVID testImg[offset + cx] = img.pixel(cx, rx);
      testImg[offset + cx] = qGray(img.pixel(cx, rx));
    }
  }
}

efj::Database::Database(const std::string &dir, int facesPerSubject) :
  _facesPerSubject(facesPerSubject) {
  load_pixels(dir);
  debug_print_pixels();

  _nImages = _pixels.cols();
  _nPixels = _pixels.rows(); //number of pixels, column size
  _nSubjects = _nImages / _facesPerSubject;

  _mean.resize(_nPixels);

#pragma omp parallel for
  for (int i = 0; i < _nPixels; i++) {
    double accumulated = 0;
#pragma omp parallel for
    for (int j = 0; j < _nImages; j++) {
      accumulated += _pixels(i, j);
    }
    _mean(i) = accumulated / _nImages;
  }
  debug_print_mean();

  _centeredPixels.resize(_nPixels, _nImages);

  //Creating centeredPixels Matrix
#pragma omp parallel for
  for (int i = 0; i < _pixels.cols(); i++) {
#pragma omp parallel for
    for (int j = 0; j < _pixels.rows(); j++) {
      double px = _pixels(j, i);
      _centeredPixels(j, i) = px - _mean(j);
      //DAVID _centeredPixels(j, i) = std::abs(px - _mean(j));
    }
  }

  std::cerr << "pixels centered\n";
  debug_print_centeredPixels();

  //_eigenfaces.resize(_features, _nImages);
  //initialize(_eigenfaces); //initialize matriz to zeroes
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
  find_files(trainDatabasePath, ".pgm", files);
  std::sort(files.begin(), files.end());

  if (files.size() == 0)
    return false;
  //std::cout << "FILES FOUND: " << files.size() << std::endl;

  QImage img(files[0]);
  _pixels.resize((int)(img.height() * img.width()), (int)files.size());

#pragma omp parallel for
  // Construction of 2D matrix from 1D image vectors
  for (size_t fx = 0; fx < files.size(); fx++) {
    Eigen::VectorXd pixs(img.height() * img.width());
    readSingleFile(files[fx], pixs);
    _pixels.col(fx) = pixs;
  }

  return true;
}

void efj::Database::compute_eigenfaces() {

  Eigen::MatrixXd covMatrix(_nImages, _nImages);
  covMatrix = _centeredPixels.transpose() * _centeredPixels;
  debug_print_covariance_matrix(covMatrix);

  Eigen::EigenSolver<Eigen::MatrixXd> *solver2 = new Eigen::EigenSolver<Eigen::MatrixXd>(covMatrix);
  const eigenvalue_type &eigenValues = solver2->eigenvalues();
  const eigenvectors_type &eigenVectors = solver2->eigenvectors();
  debug_print_eigenvectors(eigenValues, eigenVectors);
  filter_eigenvectors(eigenValues, eigenVectors);

  std::cerr << "all done" << std::endl;
}

/**
 * Initialize _eigenfaces and _nEigenFaces
 */
inline int operator<(std::complex<double> &c1, std::complex<double> &c2) {
  return c1.real() < c2.real();
}

void efj::Database::filter_eigenvectors(const eigenvalue_type &eigenvalues,
                                        const eigenvectors_type &eigenvectors) {

  double maxev = std::numeric_limits<double>::min();
  for (int i = 0; i < eigenvalues.size(); i++) {
    if (eigenvalues(i).real() > maxev) maxev = eigenvalues(i).real();
  }

  _nEigenFaces = 0;

  std::vector<int> eigenvalues_indexvector;
  for (int i = 0; i < eigenvalues.size(); i++) {
    if (eigenvalues(i).real() / maxev > 1e-3) {
      eigenvalues_indexvector.push_back(i);
      _nEigenFaces++;
    }
  }

  _eigenfaces.resize(_nPixels, _nEigenFaces);
  initialize(_eigenfaces);

#pragma omp parallel for
  for (int eigenface = 0; eigenface < _nEigenFaces; eigenface++) {
#pragma omp parallel for
    for (int image = 0; image < _nImages; image++) {
      _eigenfaces.col(eigenface) += _centeredPixels.col(image)
          * eigenvectors(image, eigenvalues_indexvector[eigenface]).real();
    }
  }

}

void efj::Database::project_clusters() {
  std::cerr << "Projecting Training Images" << std::endl;
  _clustersProjection.resize(_nEigenFaces, _nSubjects);

  for (int subject = 0; subject < _nSubjects; subject++) {
    Eigen::VectorXd meanFace(_nPixels);
    initialize(_nPixels, meanFace); //set vector to zeroes

    //calculate the face group and equalize it
    for (int i = subject * _facesPerSubject; i < (subject + 1) * _facesPerSubject; i++) {
      std::cout << "DOING " << i << std::endl;
      meanFace += _pixels.col(i) - _mean;
    }
    meanFace /= _facesPerSubject; //this gives the mean of images from the same group

    //project it to the eigenfaces space
    Eigen::VectorXd projection;
    project_single_image(meanFace, projection);
    _clustersProjection.col(subject) = projection;
  }
}

// "projection" will be resized to _nImages
void efj::Database::project_single_image(Eigen::VectorXd &image, Eigen::VectorXd &projection) {
  std::cerr << "Projecting Test Image" << std::endl;

  Eigen::VectorXd imageEqualized(_nPixels);
#if 0
#pragma omp parallel for
  for (int pix = 0; pix < _nPixels; pix++) {
    imageEqualized(pix) = std::abs(image(pix) - _mean(pix));
  }
#endif
  imageEqualized = image - _mean;

  projection.resize(_nEigenFaces);
#pragma omp parallel for
  for (int col = 0; col < _nEigenFaces; col++) {
    projection(col) = _eigenfaces.col(col).dot(imageEqualized);
  }
}

// "distances" will be resized to _nGroups
void efj::Database::compute_distance_to_groups(Eigen::VectorXd &projection,
                                               Eigen::VectorXd &distances) {
  std::cerr << "Calculating Distances" << std::endl;
  distances.resize(_nSubjects);
  for (int subject = 0; subject < _nSubjects; subject++) {
    Eigen::VectorXd aux = projection - _clustersProjection.col(subject);
    distances(subject) = aux.norm();
  }
}

void efj::Database::write(std::string output_file) {
  std::ofstream mos(output_file.c_str(), std::ios::binary);
  // store pixels
  int rows = _pixels.rows();
  int cols = _pixels.cols();
  mos.write((const char *)&rows, sizeof(int));
  mos.write((const char *)&cols, sizeof(int));
  for (int i = 0; i < _pixels.rows(); i++) {
    for (int j = 0; j < _pixels.cols(); j++) {
      double p = _pixels(i, j);
      mos.write((const char *)&p, sizeof(double));
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

  mos.write((const char *)&_facesPerSubject, sizeof(int));
  mos.write((const char *)&_nSubjects, sizeof(int));

  // clusters
  rows = _clustersProjection.rows();
  cols = _clustersProjection.cols();
  mos.write((const char *)&rows, sizeof(int));
  mos.write((const char *)&cols, sizeof(int));
  for (int i = 0; i < _clustersProjection.rows(); i++) {
    for (int j = 0; j < _clustersProjection.cols(); j++) {
      double cp = _clustersProjection(i, j);
      mos.write((const char *)&cp, sizeof(double));
    }
  }

  mos.write((char*)&_nEigenFaces, sizeof(int));

  mos.close();
}

void efj::Database::read(std::string input_file) {
  std::cerr << "reading" << "\n";
  std::ifstream mis(input_file.c_str(), std::ios::binary);

  // read pixels
  mis.read((char*)&_nPixels, sizeof(int));
  mis.read((char*)&_nImages, sizeof(int));

  std::cerr << _nPixels << "\n" << _nImages << "\n";
  _pixels.resize(_nPixels, _nImages);
  for (int i = 0; i < _nPixels; i++) {
    for (int j = 0; j < _nImages; j++) {
      double val;
      mis.read((char*)&val, sizeof(double));
      _pixels(i, j) = val;
    }
  }

  std::cerr << "Pixels read" << "\n";

  // mean vector (face)
  int features;
  mis.read((char*)&features, sizeof(int)); // (features)
  _mean.resize(features);
  for (int i = 0; i < features; i++) {
    double val;
    mis.read((char*)&val, sizeof(double));
    _mean(i) = val;
  }

  std::cerr << "Mean Vector read" << "\n";

  // read eigenfaces
  int rows, cols;
  mis.read((char*)&rows, sizeof(int));
  mis.read((char*)&cols, sizeof(int));
  _eigenfaces.resize(rows, cols);
  for (int i = 0; i < rows; i++) {
    for (int j = 0; j < cols; j++) {
      double val;
      mis.read((char*)&val, sizeof(double));
      _eigenfaces(i, j) = val;
    }
  }

  std::cerr << "Eigenfaces read" << "\n";

  mis.read((char*)&_facesPerSubject, sizeof(int)); // grouping
  mis.read((char*)&_nSubjects, sizeof(int)); // subjects

  // read clusters
  mis.read((char*)&rows, sizeof(int));
  mis.read((char*)&cols, sizeof(int));
  _clustersProjection.resize(rows, cols);
  for (int i = 0; i < rows; i++) {
    for (int j = 0; j < cols; j++) {
      double val;
      mis.read((char*)&val, sizeof(double));
      _clustersProjection(i, j) = val;
    }
  }

  std::cerr << "Clusters read" << "\n";

  mis.read((char*)&_nEigenFaces, sizeof(int));

  std::cout << _nEigenFaces;

  std::cerr << "Top eigenvalues read" << "\n";

  mis.close();

  std::cerr << "read done" << std::endl;
}

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
