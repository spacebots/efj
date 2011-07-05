#include <cstdlib>
#include <iostream>
#include "Database.h"

int main(int argc, char *argv[]) {

  if (argc != 4) {
    std::cerr << "Usage: " << argv[0] << "  database image confidence" << std::endl;
    exit(1);
  }

  std::string database = argv[1], image = argv[2];
  double confidence = strtod(argv[3], NULL);

  efj::Database efjdb;
  efjdb.read(database);

  Eigen::VectorXd testImg;
  efjdb.readSingleFile(image.c_str(), testImg);

  Eigen::VectorXd projection, distances;
  int result;
  efjdb.project_single_image(testImg, projection);
  bool recognized = efjdb.compute_single_match_with_confidence(projection, distances, result, confidence);

  std::cerr << "DISTANCES: \n" << distances << std::endl;
  std::cerr << "RESULT: \n";
  if (recognized) std::cout << result << ": " << confidence << std::endl;

}

