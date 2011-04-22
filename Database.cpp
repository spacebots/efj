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
  for (int cx = 0; cx < _pixels.cols(); cx++) {
#pragma omp parallel for
    for (int rx = 0; rx < _pixels.rows(); rx++) {
      _centeredPixels(rx, cx) = _pixels(rx, cx) - _mean(rx);
    }
  }

  std::cerr << "pixels centered\n";
  debug_print_centeredPixels();
}

void efj::Database::compute_eigenfaces() {
  Eigen::MatrixXd covMatrix(_nImages, _nImages);
  covMatrix = _centeredPixels.transpose() * _centeredPixels; //[IxP]x[PxI]=[IxI]
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
    if (eigenvalues(i).real() > maxev)
      maxev = eigenvalues(i).real();
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

//#pragma omp parallel for
  for (int eigenface = 0; eigenface < _nEigenFaces; eigenface++) {
//#pragma omp parallel for
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
      meanFace += _pixels.col(i);
    }
    meanFace /= _facesPerSubject; //this gives the mean of images from the same group
    meanFace -= _mean;

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
