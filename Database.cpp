// $Id: Database.cpp,v 1.19 2012/02/12 02:05:23 ferreira Exp $
//
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
// $Log: Database.cpp,v $
// Revision 1.19  2012/02/12 02:05:23  ferreira
// Added CSUFaceIDEvalSystem compatible output
//
// Revision 1.18  2011/08/15 16:36:15  david
// Updated project files to be more compatible with building installation
// packages.
//
// Revision 1.17  2011/07/22 14:44:23  david
// Minor cleanup.
//
// Revision 1.16  2011/07/22 14:32:44  ferreira
// Added Copyright comment
//

#include <vector>
#include <string>
#include <utility>
#include <fstream>
#include <iostream>
#include <boost/filesystem.hpp>
#include <Eigen/Core>
#include <QImage>
#include <QString>

#include <efj/Database.h>

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

  _eigenValues = eigenValues;		//atribute filling for CSU compliance
  _eigenVectors = eigenVectors;		//atribute filling for CSU compliance

  std::cerr << "all done" << std::endl;
}

/**
 * Initialize _eigenfaces and _nEigenFaces
 */
inline int operator<(std::complex<double> &c1, std::complex<double> &c2) {
  return c1.real() < c2.real();
}

//#define EIGENVALUES_RATIO 1e-2
#define EIGENVALUES_RATIO 0  /* use all eigenvectors */
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
    if (eigenvalues(i).real() / maxev > EIGENVALUES_RATIO) {
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

    std::cout << "********** EIGENFACE " << eigenface << std::endl;
    //DAVID std::cout << "P2 172 244 255" << std::endl; // first tests
    std::cout << "P2 72 72 255" << std::endl; // camera
    std::cout << _eigenfaces.col(eigenface) << std::endl;
    std::cout << "********** EIGENFACE " << eigenface << " (END)" << std::endl;

    //DAVID: the following produces eigenfaces as images
    //DAVID: note that values as normalized as grayscale
    std::stringstream oefs;
    oefs << "efj-auto-eigenface-" << eigenface << ".pnm";
    std::ofstream oef(oefs.str().c_str());
    oef << "P2 " << sqrt(_eigenfaces.rows()) << " " << sqrt(_eigenfaces.rows()) << " 255"
        << std::endl; // camera
    Eigen::VectorXd pixels = _eigenfaces.col(eigenface);
    for (int px = 0; px < pixels.size(); px++)
      pixels(px) = std::abs(pixels(px));
    double efmax = pixels.maxCoeff();
    pixels *= 255.0 / efmax;
    for (int px = 0; px < pixels.size(); px++)
      oef << (int)pixels(px) << std::endl;
    oef.close();

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

    //project it to the eigenfaces space
    Eigen::VectorXd projection;
    project_single_image(meanFace, projection);
    _clustersProjection.col(subject) = projection;
  }
}

// "projection" will be resized to _nImages
void efj::Database::project_single_image(Eigen::VectorXd &image, Eigen::VectorXd &projection) const {
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

// "distances" will be resized to _nGroups
void efj::Database::compute_distance_to_groups(Eigen::VectorXd &projection,
                                               Eigen::VectorXd &distances, Eigen::VectorXd &results) {
  std::cerr << "Calculating Distances" << std::endl;
  distances.resize(_nSubjects);
  double average = 0;
  for (int subject = 0; subject < _nSubjects; subject++) {
    Eigen::VectorXd aux = projection - _clustersProjection.col(subject);
    distances(subject) = aux.norm();
    average += aux.norm();
  }
  average /= _nSubjects;

  results.resize(_nSubjects);
  for (int subject = 0; subject < _nSubjects; subject++) {
    if (distances(subject) > average)
      results(subject) = 0.0;
    else {
      double rate = std::abs(average - distances(subject)) / average;
      if (rate < 0.5)
        results(subject) = 0.0;
      else {
        if (rate > 0.95)
          results(subject) = 1.0;
        else
          results(subject) = rate;
      }
    }
  }

}

// "distances" will be resized to _nGroups
bool efj::Database::compute_single_match_with_confidence(Eigen::VectorXd &projection,
                                                         Eigen::VectorXd &distances, int &result,
                                                         double &confidence) const {
  std::cerr << "Calculating Distances" << std::endl;
  distances.resize(_nSubjects);
  double average = 0;
  double minimum = std::numeric_limits<double>::max();
  result = 0;
  for (int subject = 0; subject < _nSubjects; subject++) {
    Eigen::VectorXd aux = projection - _clustersProjection.col(subject);
    double norm = aux.norm();
    distances(subject) = norm;
    if (norm < minimum) {
      minimum = norm;
      result = subject;
    }
    average += aux.norm();
  }
  average /= _nSubjects;

  if (distances(result) > average) {
    confidence = 1.0;
    return false;
  } else {
    double rate = std::abs(average - distances(result)) / average;
    if (rate < 0.5 || rate < confidence) {
      confidence = rate;
      return false;
    } else {
      if (rate > 0.95)
        confidence = 1.0;
      else
        confidence = rate;
      return true;
    }
  }

}

