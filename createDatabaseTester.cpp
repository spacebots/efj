#include <vector>
#include <string>
#include <fstream>
#include <iostream>
#include <QString>
#include <boost/filesystem.hpp>
#include <QDebug>

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

int main() {

  const std::string dir = "/afs/l2f.inesc-id.pt/home/ferreira/face-recognition/ImageVault/train";

  std::vector<QString> files;

  find_files(dir, ".jpg", files);

  for (int i = 0; i < 500; i++) {
    //std::cout << files[i];
    qDebug("%s", qPrintable(files[i]));
  }

}
