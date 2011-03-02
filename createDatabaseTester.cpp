#include <vector>
#include <string>
#include <fstream>
#include <iostream>
#include <QString>
#include <boost/filesystem.hpp>
#include <QDebug>

#include <Eigen/Core>
#include "Database.h"

//CONSEGUES LER ISTO???

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
int find_min(Eigen::VectorXi vec){
	int min = vec[0];
	int index = 0;
	for(int i = 1, aux = 0 ; i < vec.size() ; i++){
		if(aux < min){
			index = i;
		}
	}
	return index;
}

int main() {

  const std::string dir = "/afs/l2f.inesc-id.pt/home/ferreira/face-recognition/ImageVault/train";

  std::vector<QString> files;

  find_files(dir, ".jpg", files);
  std::sort(files.begin(), files.end());

  //load the matrix
  efj::Database efjdb;
  efjdb.read("batata.dat");

  int certas;
  int erradas;


  for(int i = 0 , subject ; i < files.size() ; i++) {


     //std::cerr << "Test image path:" << std::endl;
     //std::string testImgPath;
     //std::cin >> testImgPath;

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


     std::cerr << "Certas: " << certas << std::endl;
     std::cerr << "Erradas: " << erradas << std::endl;


     //std::cerr << distances;
     //std::cerr << "Done" << std::endl;




  }

#if 0
  for (int i = 0; i < 500; i++) {
    //std::cout << files[i];
    qDebug("%s", qPrintable(files[i]));
  }
#endif

}
