#include <iostream>
#include "Database.h"

int main(int argc, char *argv[]) {
  std::string dir;
  std::string database;
  int nSubjects = 1;

  if (argc == 4) {
    dir = argv[2];
    database = argv[3];
    nSubjects = strtol(argv[1], NULL, 10);
  } else {
    std::cerr << "Usage: " << argv[0] << " images_per_subject directory_name database_name" << std::endl;
    exit(1);
  }

  efj::Database db(dir, nSubjects);
  db.compute_eigenfaces();
  db.project_clusters();
  db.write(database);
}
