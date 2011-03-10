#include <iostream>
#include "Database.h"

int main(int argc, char *argv[]) {
  std::string dir;
  std::string saveDir;
  int grouping = 1;
  int topEigenValues = 0;

  ///ofs/tmp/eigDataBase1.dat

  if (argc == 5) {
    dir = argv[3];
    saveDir = argv[4];
    grouping = strtol(argv[1], NULL, 10);
    topEigenValues = strtol(argv[2], NULL, 10);
  } else {
    std::cerr << "Usage: " << argv[0] << " grouping topEigenValues directory_name save_database_directory" << std::endl;
    exit(1);
  }

  efj::Database db(dir, grouping, topEigenValues);
  db.compute_eigenfaces();
  db.project_clusters();
  db.write(saveDir);
  //db.write("ofs/tmp/eigDataBaseYaleBExtendedCroped.dat");
}
