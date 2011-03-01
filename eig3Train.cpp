#include "Database.h"

int main(int argc, char *argv[]) {
  //const std::string dir = "/afs/l2f/home/ferreira/FaceRec/ImageVault/train";
  const std::string dir = "/afs/l2f/home/ferreira/face-recognition/MyTrainDatabase";
  efj::Database db(dir);
  db.compute_eigenfaces();
  db.write("batata.dat");
}
