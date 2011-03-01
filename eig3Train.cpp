#include <iostream>
#include "Database.h"

int main(int argc, char *argv[]) {
  std::string dir;
  int grouping = 1;

  if (argc == 3) {
    dir = argv[2];
    grouping = strtol(argv[1], NULL, 10);
  } else {
    std::cerr << "Usage: " << argv[0] << " grouping directory_name" << std::endl;
    exit(1);
  }

  efj::Database db(dir, grouping);
  db.compute_eigenfaces();
  db.project_clusters();
  db.write("batata.dat");
}
