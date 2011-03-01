#include <iostream>
#include "Database.h"

int main(int argc, char *argv[]) {
  //const std::string dir = "/afs/l2f/home/ferreira/face-recognition/ImageVault/train";
  std::string dir;

  if (argc == 1) {
    dir = "/afs/l2f/home/ferreira/face-recognition/MyTrainDatabase";
  } else if (argc == 2) {
    dir = argv[1];
  } else {
    std::cerr << "Usage: " << argv[0] << " [ directory_name ]" << std::endl;
    exit(1);
  }

  efj::Database db(dir);
  db.compute_eigenfaces();
  db.write("batata.dat");
}
