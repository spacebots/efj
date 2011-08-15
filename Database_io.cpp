// $Id: Database_io.cpp,v 1.6 2011/08/15 16:36:15 david Exp $
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
// $Log: Database_io.cpp,v $
// Revision 1.6  2011/08/15 16:36:15  david
// Updated project files to be more compatible with building installation
// packages.
//
// Revision 1.5  2011/07/22 14:44:23  david
// Minor cleanup.
//
// Revision 1.4  2011/07/22 14:32:44  ferreira
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

namespace bf = boost::filesystem;

void efj::Database::readSingleFile(QString imgFile, Eigen::VectorXd &imgVector) {
  QImage img(imgFile);
  imgVector.resize((int)(img.height() * (int)(img.width())));
#pragma omp parallel for
  for (int rx = 0; rx < img.height(); rx++) {
    int offset = rx * img.width();
#pragma omp parallel for
    for (int cx = 0; cx < img.width(); cx++) {
      //DAVID imgVector[offset + cx] = img.pixel(cx, rx);
      imgVector[offset + cx] = qGray(img.pixel(cx, rx));
    }
  }
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
  find_files(trainDatabasePath, ".png", files);
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
