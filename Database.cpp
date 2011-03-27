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

efj::Database::Database(const std::string &dir, int grouping, int topEigenValues) :
		  _grouping(grouping), _topEigenValues(topEigenValues){
	load_pixels(dir);
	debug_print_pixels();

	_nImages = _pixels.cols();
	_features = _pixels.rows(); //number of pixels, column size
	_nGroups = _nImages / _grouping;

	_mean.resize(_features);

#pragma omp parallel for
	for (int i = 0; i < _features; i++) {
		double accumulated = 0;
#pragma omp parallel for
		for (int j = 0; j < _nImages; j++) {
			accumulated += _pixels(i, j);
		}
		_mean(i) = accumulated / _nImages;
	}
	debug_print_mean();

	_centeredPixels.resize(_features, _nImages);

	//Creating centeredPixels Matrix
#pragma omp parallel for
	for (int i = 0; i < _pixels.cols(); i++) {
#pragma omp parallel for
		for (int j = 0; j < _pixels.rows(); j++) {
			double px = _pixels(j, i);
			_centeredPixels(j, i) = std::abs(px - _mean(j));
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
	//std::cout << "1st FILE: " << files[0].toStdString().c_str() << std::endl;
	//std::cout << "rows x cols: " << (int)(img.height() * img.width()) << " x " << (int)files.size() << std::endl;
	_pixels.resize((int)(img.height() * img.width()), (int)files.size());

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
		_pixels.col(fx) = pixs;

	}

	return true;
}

void efj::Database::compute_eigenfaces() {
	Eigen::MatrixXd covMatrix(_nImages, _nImages);
	covMatrix = _centeredPixels.transpose() * _centeredPixels;

	debug_print_covariance_matrix(covMatrix);

	Eigen::EigenSolver<Eigen::MatrixXd> *solver2 = new Eigen::EigenSolver<Eigen::MatrixXd>(covMatrix);
	std::cerr << "eigensolver created \n";

	Eigen::EigenSolver<Eigen::MatrixXd>::EigenvalueType eigenValues = solver2->eigenvalues();
	Eigen::EigenSolver<Eigen::MatrixXd>::EigenvectorsType eigenVectors = solver2->eigenvectors();

	debug_print_eigenvectors(eigenValues, eigenVectors);

	Eigen::MatrixXd usefullVectors;

	std::vector< std::pair<double,int> > selectedEigenValues;

	filter_eigenVectors(usefullVectors , eigenValues , eigenVectors, _topEigenValues, selectedEigenValues);

	if(_topEigenValues > 0){

		_eigenfaces.resize(_features, _topEigenValues);

		initialize(_eigenfaces); //initialize matriz to zeroes

		_centeredPixelsFiltered.resize(_features, _topEigenValues);
		for(int i = 0 ; i < _topEigenValues ; i++){
			_centeredPixelsFiltered.col(i) = _centeredPixels.col(selectedEigenValues.at(i).second);
		}


#pragma omp parallel for
		for (int i = 0; i < _topEigenValues; i++) {//i = col
#pragma omp parallel for
			for (int j = 0; j < _topEigenValues; j++) { // j = row
				_eigenfaces.col(i) += _centeredPixelsFiltered.col(j) * usefullVectors(j, i);
			}
		}
	}
	else if(_topEigenValues == 0){

#pragma omp parallel for
		for (int i = 0; i < _nImages; i++) {//i = col
#pragma omp parallel for
			for (int j = 0; j < _nImages; j++) { // j = row
				_eigenfaces.col(i) += _centeredPixels.col(j) * eigenVectors(j, i).real();
			}
		}

	}
	else{
		std::cerr << "UNKNOWN ERROR AT COMPUTE_EIGENFACES" << std::endl;
	}

  std::cerr << "all done" << std::endl;
}

void efj::Database::project_clusters() {
  std::cerr << "Projecting Training Images" << std::endl;

  _clustersProjection.resize(_topEigenValues, _nGroups);

  Eigen::VectorXd trainingFaceEqualized(_features);
  for (int image = 0, group = 0; group < _nGroups; group++) {
    initialize(_features, trainingFaceEqualized); //set vector to zeroes
    //calculate the face group and equalize it
    for (int i = 0; i < _grouping && image < _nImages; i++, image++) {
      trainingFaceEqualized += _pixels.col(image).cast<double>() - _mean;
    }
    trainingFaceEqualized /= _grouping; //this gives the mean of images from the same group

    //project it to the eigenfaces space
#pragma omp parallel for
    for (int i = 0; i < _topEigenValues; i++) {
      double a = _eigenfaces.col(i).dot(trainingFaceEqualized);
      _clustersProjection(i, group) = a;
    }

  }
  std::cerr << "Projecting Training Images - STARTING" << std::endl;
  Eigen::MatrixXd batata = _clustersProjection;

  std::cerr << batata.transpose() << std::endl;
  std::cerr << "Projecting Training Images - DONE" << std::endl;
}

// "projection" will be resized to _topEigenValues
void efj::Database::project_single_image(Eigen::VectorXi &image, Eigen::VectorXd &projection) {
  //std::cerr << "Projecting Test Image" << std::endl;

  Eigen::VectorXd imageEqualized(_features);
#pragma omp parallel for
  for (int i = 0; i < imageEqualized.size(); i++) {
    //imageEqualized(i) = std::abs(image(i) - _mean(i));
    imageEqualized(i) = image(i) - _mean(i);
  }

  projection.resize(_topEigenValues);
#pragma omp parallel for
  for (int i = 0; i < _topEigenValues; i++) {
    projection(i) = _eigenfaces.col(i).dot(imageEqualized);
  }
}

// "distances" will be resized to _topEigenValues
void efj::Database::compute_distance_to_groups(Eigen::VectorXd &projection, Eigen::VectorXd &distances) {
  //std::cerr << "Calculating Distances" << std::endl;

  distances.resize(_nGroups);

  for (int i = 0; i < _nGroups; i++) {
    Eigen::VectorXd aux(_topEigenValues);
#pragma omp parallel for
    for (int j = 0; j < _topEigenValues; j++) {
      aux(j) = projection(j) - _clustersProjection(j, i);
    }
    distances(i) = aux.norm();
  }

}

#if 0
void efj::Database::project_clusters() {
  std::cerr << "Projecting Training Images" << std::endl;

  _clustersProjection.resize(_nImages, _nGroups);

  Eigen::VectorXd trainingFaceEqualized(_features);
  for (int image = 0, group = 0; group < _nGroups; group++) {
    initialize(_features, trainingFaceEqualized); //set vector to zeroes
    //calculate the face group and equalize it
    for (int i = 0; i < _grouping && image < _nImages; i++, image++) {
      trainingFaceEqualized += _pixels.col(image).cast<double>() - _mean;
    }
    trainingFaceEqualized /= _grouping; //this gives the mean of images from the same group

    //project it to the eigenfaces space
#pragma omp parallel for
    for (int i = 0; i < _nImages; i++) {
      double a = _eigenfaces.col(i).dot(trainingFaceEqualized);
      _clustersProjection(i, group) = a;
    }
  }
}


// "projection" will be resized to _nImages
void efj::Database::project_single_image(Eigen::VectorXi &image, Eigen::VectorXd &projection) {
  std::cerr << "Projecting Test Image" << std::endl;

  Eigen::VectorXd imageEqualized(_features);
#pragma omp parallel for
  for (int i = 0; i < imageEqualized.size(); i++) {
    imageEqualized(i) = std::abs(image(i) - _mean(i));
  }

  projection.resize(_nImages);
#pragma omp parallel for
  for (int i = 0; i < _nImages; i++) {
    projection(i) = _eigenfaces.col(i).dot(imageEqualized);
  }
}


// "distances" will be resized to _nGroups
void efj::Database::compute_distance_to_groups(Eigen::VectorXd &projection, Eigen::VectorXd &distances) {
  std::cerr << "Calculating Distances" << std::endl;

  distances.resize(_nGroups);

  for (int i = 0; i < _nGroups; i++) {
    Eigen::VectorXd aux(_nImages);
#pragma omp parallel for
    for (int j = 0; j < _nImages; j++) {
      aux(j) = projection(j) - _clustersProjection(j, i);
    }
    distances(i) = aux.norm();
  }

}
#endif

//"usefullVectors" will be resized to _topEigenValues
void efj::Database::filter_eigenVectors(Eigen::MatrixXd &usefulVectors, eigenvalue_type &eigenValues,
    										 eigenvectors_type &eigenVectors,
    										 std::vector< std::pair<double,int> > &selectedEigenValues){

	if(_topEigenValues == 0){
		_topEigenValues = _nImages;
	}

	int eigenVectorsRows = eigenVectors.rows();
	usefulVectors.resize(eigenVectorsRows , _topEigenValues);

	//std::cout << eigenVectors.rows();

	//std::vector< std::pair<double,int> > selectedEigenValues;

	for(int i = 0 ; i < eigenValues.size() ; i++){
		selectedEigenValues.push_back(std::make_pair(eigenValues(i).real() , i));
	}
	//Eigen::VectorXd eigVal = eigenValues;
	std::sort(selectedEigenValues.begin(), selectedEigenValues.end());

#if 0
	for(int k = 0 ; k < _nImages ; k++){
		std::cout << selectedEigenValues.at(k).first << std::endl;
	}

	std::cerr << "PRINTED";
#endif

#pragma omp parallel for
  for (int i = 0; i < _topEigenValues; i++) {//i = col
#pragma omp parallel for
    for (int j = 0; j < eigenVectorsRows; j++) { // j = row
    	usefulVectors(j,i) = eigenVectors(j, selectedEigenValues.back().second).real();
    }
    selectedEigenValues.pop_back();
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
      int p = _pixels(i, j);
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

  mos.write((const char *)&_grouping, sizeof(int));
  mos.write((const char *)&_nGroups, sizeof(int));

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

  mos.write((char*)&_topEigenValues, sizeof(int));

  mos.close();
}

void efj::Database::read(std::string input_file) {
  std::cerr << "reading" << "\n";
  std::ifstream mis(input_file.c_str(), std::ios::binary);

  // read pixels
  mis.read((char*)&_features, sizeof(int));
  mis.read((char*)&_nImages, sizeof(int));

  std::cerr << _features << "\n" << _nImages << "\n";
  _pixels.resize(_features, _nImages);
  for (int i = 0; i < _features; i++) {
    for (int j = 0; j < _nImages; j++) {
      int val;
      mis.read((char*)&val, sizeof(int));
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

  mis.read((char*)&_grouping, sizeof(int)); // grouping
  mis.read((char*)&_nGroups, sizeof(int)); // subjects

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

  mis.read((char*)&_topEigenValues, sizeof(int));

  std::cout << _topEigenValues;

  std::cerr << "Top eigenvalues read" << "\n";

  mis.close();

  std::cerr << "read done" << std::endl;
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

void efj::Database::debug_print_eigenvectors(eigenvalue_type &eigenValues,
                                             eigenvectors_type &eigenVectors) {
#if DEBUG
  std::cout << "Eigenvalues:" << std::endl;
  std::cout << eigenValues;
  std::cout << "Eigenvectors:" << std::endl;
  std::cout << eigenVectors;
  std::cout << "eigenvectors done: " << _nImages << " images @ " << _features << " features\n";
#endif
}

void efj::Database::debug_print_covariance_matrix(Eigen::MatrixXd &covMatrix) {
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
