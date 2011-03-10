#include <vector>
#include <string>
#include <fstream>
#include <iostream>
#include <QString>
#include <boost/filesystem.hpp>
#include <QDebug>

#include <Eigen/Core>
#include "Database.h"


/**
 *
 * This function DOES NOT get the files sequentially..
 *
 */

namespace bf = boost::filesystem;

/**
 * in this directory, seach for file with given extension
 * place paths in vector
 */
void find_files(const bf::path & dir_path, const std::string & extension,
                std::vector<QString> &files) {
  if (!bf::exists(dir_path))
    return;
  bf::directory_iterator end_itr; // default construction yields past-the-end
  for (bf::directory_iterator itr(dir_path); itr != end_itr; ++itr) {
    if (bf::is_directory(itr->status())) {
      find_files(itr->path(), extension, files);
    } else if (itr->path().extension() == extension) {
      //files.push_back(itr->leaf());  //only the last part
      // std::cout << itr->path().string().c_str();
      files.push_back(itr->path().string().c_str());
    }
  }
}

//finds minimum value of the vector returning its index
int find_min(Eigen::VectorXd &vec){
	int min = vec[0];
	std::cout << min << std::endl;
	int index = 0;
	for(int i = 1; i < vec.size() ; i++){
		std::cout << min << std::endl;
		std::cout << vec[i] << std::endl;
		if(vec[i] < min){
			min = vec[i];
			index = i;
		}
		std::cout << index << std::endl;
	}
	return index;
}

int main(int argc, char *argv[]) {

#if 0
	Eigen::VectorXd teste(8);
	teste << 10,20,3,4,5,6,31,20;

	std::cout << "MinIndex: " << find_min(teste) << "   " << teste(find_min(teste));

#endif

	std::string databaseDir;
	std::string testDir;

	if (argc == 3) {
		databaseDir = argv[1];
		testDir = argv[2];
	}
	else {
		std::cerr << "Usage: " << argv[0] << "  read_database_directory  read_test_directory" << std::endl;
		exit(1);
	}


	//const std::string dir = "/afs/l2f.inesc-id.pt/home/ferreira/face-recognition/ImageVault/train";
	//const std::string dir = "/afs/l2f.inesc-id.pt/home/ferreira/face-recognition/ImageVault/test";
	//const std::string dir = "/afs/l2f.inesc-id.pt/home/ferreira/face-recognition/MyTrainDatabase";
	std::vector<QString> files;

	find_files(testDir, ".jpg", files);
	std::sort(files.begin(), files.end());

	//load the matrix
	efj::Database efjdb;
	//efjdb.read("/ofs/tmp/david/batata.dat");
	efjdb.read(databaseDir);

	int grouping = efjdb.get_grouping();
	int nGroups = efjdb.get_nGroups();

	int certas = 0;
	int erradas = 0;

	int aux = 0;

	for(int i = 0 , subject = 0; i < (int)files.size() && subject < nGroups ; i++ , aux++) {


		//std::cerr << "Test image path:" << std::endl;
		//std::string testImgPath;
		//std::cin >> testImgPath;

		if(aux == grouping){
			subject++;
			aux = 0;
		}

		Eigen::VectorXi testImg;
		efj::readSingleFile(files[i].toStdString(), testImg);

		Eigen::VectorXd projection;
		efjdb.project_single_image(testImg, projection);

		Eigen::VectorXd distances;
		efjdb.compute_distance_to_groups(projection, distances);


		if(find_min(distances) == subject){
			certas++;
		}
		else{
			erradas++;
		}





		//std::cerr << distances;
		//std::cerr << "Done" << std::endl;




	}

	std::cout << "Certas: " << certas << std::endl;
	std::cout << "Erradas: " << erradas << std::endl;

#if 0
  for (int i = 0; i < 500; i++) {
    //std::cout << files[i];
    qDebug("%s", qPrintable(files[i]));
  }
#endif

}
