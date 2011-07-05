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
  efjdb.readSingleFile(image.c_str(), testImg);

  Eigen::VectorXd projection, distances, results;
  efjdb.project_single_image(testImg, projection);
  efjdb.compute_distance_to_groups(projection, distances, results);

  std::cerr << "DISTANCES: \n" << distances << std::endl;
  std::cerr << "RESULTS: \n";
  for (int rx = 0; rx < results.rows(); rx++)
    if (results(rx) != 0) std::cout << rx << ": " << results(rx) << std::endl;

}

