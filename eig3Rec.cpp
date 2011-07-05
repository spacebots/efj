#include <iostream>
#include "Database.h"

int main(int argc, char *argv[]) {

	std::string database;

	if (argc == 2) {
		database = argv[1];
	}
	else {
		std::cerr << "Usage: " << argv[0] << "  database" << std::endl;
		exit(1);
	}

	efj::Database efjdb;
	efjdb.read(database);

  while (true) {
    std::cerr << "Test image path:" << std::endl;
    std::string testImgPath;
    std::cin >> testImgPath;

    Eigen::VectorXd testImg;
    efjdb.readSingleFile(testImgPath.c_str(), testImg);

    Eigen::VectorXd projection;
    efjdb.project_single_image(testImg, projection);

    std::cerr << "IMAGE:" << std::endl;
    std::cerr << testImg << std::endl;
    std::cerr << "PROJECTION:" << std::endl;
    std::cerr << projection << std::endl;
    std::cerr << "OOOK!" << std::endl;

    Eigen::VectorXd distances;
    efjdb.compute_distance_to_groups(projection, distances);

    std::cerr << distances << std::endl;
    std::cerr << "Done" << std::endl;
  }

}

