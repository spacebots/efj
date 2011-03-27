#include <iostream>
#include "Database.h"

int main(int argc, char *argv[]) {

	std::string readDir;

	if (argc == 2) {
		readDir = argv[1];
	}
	else {
		std::cerr << "Usage: " << argv[0] << "  read_database_directory" << std::endl;
		exit(1);
	}

	//"/ofs/tmp/eigDataBase1.dat"

	efj::Database efjdb;
	efjdb.read(readDir);
	//efjdb.read("ofs/tmp/eigDataBaseYaleBExtendedCroped.dat");

  while (true) {
    std::cerr << "Test image path:" << std::endl;
    std::string testImgPath;
    std::cin >> testImgPath;

    Eigen::VectorXi testImg;
    efj::readSingleFile(testImgPath, testImg);

    Eigen::VectorXd projection;
    efjdb.project_single_image(testImg, projection);

    std::cerr << "IMAGE:" << std::endl;
    std::cerr << testImg << std::endl;
    std::cerr << "PROJECTION:" << std::endl;
    std::cerr << projection << std::endl;
    std::cerr << "OOOK!" << std::endl;

    Eigen::VectorXd distances;
    efjdb.compute_distance_to_groups(projection, distances);

    std::cerr << distances;
    std::cerr << "Done" << std::endl;
  }

}

