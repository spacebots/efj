#include <iostream>
#include "Database.h"

int main(int argc, char *argv[]) {

  efj::Database efjdb;
  efjdb.read("batata.dat");

  while (true) {

    std::cerr << "Test image path:" << std::endl;
    std::string testImgPath;
    std::cin >> testImgPath;

    Eigen::VectorXi testImg;
    efj::readSingleFile(testImgPath, testImg);

    Eigen::VectorXd projection;
    efjdb.project_single_image(testImg, projection);

    Eigen::VectorXd distances;
    efjdb.compute_distance_to_groups(projection, distances);

    std::cerr << distances;
    std::cerr << "Done" << std::endl;

  }
}

