#include <iostream>
#include "Database.h"

int main(int argc, char *argv[]) {

  if (argc != 3) {
    std::cerr << "Usage: " << argv[0] << "  database image" << std::endl;
    exit(1);
  }

  std::string database = argv[1], image = argv[2];

  efj::Database efjdb;
  efjdb.read(database);

  Eigen::VectorXd testImg;
  efj::Database::readSingleFile(image.c_str(), testImg);

  Eigen::VectorXd projection, distances;
  efjdb.project_single_image(testImg, projection);
  efjdb.compute_distance_to_groups(projection, distances);

  std::cerr << distances << std::endl;
}

