#include <time.h>
#include <stdlib.h>
#include <vector>
#include <string>
#include <fstream>
#include <iostream>
#include <Eigen/Core>
#include <Eigen/QR> //Non-Stable part of Eigen API
//#include <Eigen/Geometry>
//#include <Eigen/LU>
#include <QImage>
#include <QString>

void readSingleFile(std::string pathToFile, Eigen::VectorXi &testImg) {
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

template<typename MatrixType>
void initialize(int rows, int cols, MatrixType &matrix) {
#pragma omp parallel for
  for (int i = 0; i < rows; i++) {
#pragma omp parallel for
    for (int j = 0; j < cols; j++) {
      matrix(i, j) = 0;
    }
  }
}

template<typename MatrixType>
void initialize(MatrixType &matrix) {
  initialize(matrix.rows(), matrix.cols(), matrix);
}

template<typename VectorType>
void initialize(int size, VectorType &vector) {
#pragma omp parallel for
  for (int j = 0; j < size; j++) {
    vector[j] = 0;
  }
}

int main(int argc, char *argv[]) {

  std::cerr << "reading" << "\n";

  std::ifstream mis("matrix2.out", std::ios::binary);

  // read pixels
  int rows, cols;
  mis.read((char*)&rows, sizeof(int));
  mis.read((char*)&cols, sizeof(int));

  std::cerr << rows << "\n" << cols << "\n";
  Eigen::MatrixXi pixels(rows, cols);
  for (int i = 0; i < rows; i++) {
    for (int j = 0; j < cols; j++) {
      int val;
      mis.read((char*)&val, sizeof(int));
      pixels(i, j) = val;
    }
  }

  std::cerr << "Pixels read" << "\n";

  // mean vector (face)
  int features;
  mis.read((char*)&features, sizeof(int)); // (features)
  Eigen::VectorXd mean(features);
  for (int i = 0; i < features; i++) {
    double val;
    mis.read((char*)&val, sizeof(double));
    mean(i) = val;
  }

  std::cerr << "Mean Vector read" << "\n";

  // read eigenfaces
  mis.read((char*)&rows, sizeof(int));
  mis.read((char*)&cols, sizeof(int));
  Eigen::MatrixXd eigenfaces(rows, cols);
  for (int i = 0; i < rows; i++) {
    for (int j = 0; j < cols; j++) {
      double val;
      mis.read((char*)&val, sizeof(double));
      eigenfaces(i, j) = val;
    }
  }
  mis.close();

  int nImages = cols;

  std::cerr << "read done" << std::endl;

  //******************************* Recognition ********************************//

  std::cerr << "Recognition Phase" << std::endl;
  std::cerr << "Projecting Training Images" << std::endl;

  int grouping; //number of images per group/subject
  std::cerr << "Images per Subject:" << std::endl;
  std::cin >> grouping;
  int nGroups = nImages / grouping;

  Eigen::VectorXd trainingFaceEqualized(features);
  //Eigen::MatrixXd imageClustersProjection(nImages, nGroups);
  Eigen::MatrixXd *imageClustersProjection = new Eigen::MatrixXd(nImages, nGroups);

  //initialize(nImages, nGroups , *imageClustersProjection);
  for (int image = 0, group = 0; group < nGroups; group++) {
    initialize(features, trainingFaceEqualized); //set vector to zeroes
    //calculate the face group and equalize it
    for (int i = 0; i < grouping && image < nImages; i++, image++) {
      trainingFaceEqualized += pixels.col(image) - mean;
    }

    trainingFaceEqualized /= grouping; //this gives the mean of images from the same group

    //project it to the eigenfaces space
#pragma omp parallel for
    for (int i = 0; i < nImages; i++) {
      //imageClustersProjection.col(group)[i] = eigenfaces.row(i).cast<int>().dot(trainingFaceEqualized);
      double a = eigenfaces.col(i).dot(trainingFaceEqualized);
      (*imageClustersProjection)(i, group) = a;
    }
  }
  //std::cout << *imageClustersProjection << std::endl;
  //We now have all images projected into the eigenface space

  while (true) {

    std::cerr << "Test image path:" << std::endl;
    Eigen::VectorXi testImg;
    std::string testImgPath;
    std::cin >> testImgPath;
    readSingleFile(testImgPath, testImg);

    //std::cout << "P2 640 480 255 \n";
    //std::cout << testImg;

    std::cerr << "Projecting Test Image" << std::endl;

    Eigen::VectorXd testFaceEqualized(features);
#pragma omp parallel for
    for (int i = 0; i < testFaceEqualized.size(); i++) {
      testFaceEqualized(i) = std::abs(testImg(i) - mean(i));
    }

    //std::cout << "P2 640 480 255 \n";
    //std::cout << testFaceEqualized;

    Eigen::VectorXd testFaceProjection(nImages);
#pragma omp parallel for
    for (int i = 0; i < nImages; i++) {
      testFaceProjection(i) = eigenfaces.col(i).dot(testFaceEqualized);
      //std::cout << eigenfaces.row(i) * testFaceEqualized;
    }

    std::cerr << "Calculating Distances" << std::endl;

    Eigen::VectorXd distances(nGroups);

    for (int i = 0; i < nGroups; i++) {
      Eigen::VectorXd aux(nImages);
#pragma omp parallel for
      for (int j = 0; j < nImages; j++) {
        //std::cerr << testFaceProjection[j] << std::endl;
        //std::cerr << imageClustersProjection.row(i)[j] << std::endl;
        aux(j) = testFaceProjection(j) - (*imageClustersProjection)(j, i);
      }
      distances(i) = aux.norm();
      //std::cerr << aux << std::endl;

    }

    /*for(int i = 0 ; i < nGroups ; i++){
     std::cerr << distances[i] << std::endl;
     }*/

    std::cerr << distances;
    std::cerr << "Done" << std::endl;
  }
}

