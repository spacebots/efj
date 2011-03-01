#ifndef __EFJ_DATABASE_H__
#define __EFJ_DATABASE_H__

#include <vector>
#include <string>
#include <boost/filesystem.hpp>
#include <Eigen/Core>
#include <QString>

#include "misc.h"

namespace bf = boost::filesystem;

namespace efj {

  class Database {

    // this matrix may become so large it won't fit in the stack
    // so, we allocate it on the heap
    Eigen::MatrixXi *_pixels; // M*N x P

    int _nImages; // typically, _pixels->cols()
    int _features; // typically, _pixels->rows()

    Eigen::VectorXd _mean;
    Eigen::MatrixXd _centeredPixels; //[M*N x P] - [mean] ///usar new

    Eigen::MatrixXd _eigenfaces;

  public:

    //dir = "/afs/l2f.inesc-id.pt/home/ferreira/face-recognition/MyTrainDatabase"
    //dir = "/afs/l2f.inesc-id.pt/home/ferreira/FaceRec/ImageVault/train";
    Database(const std::string &dir);

    /**
     * in this directory, seach for file with given extension
     * place paths in vector
     */
    void find_files(const bf::path & dir_path, const std::string & extension,
                    std::vector<QString> &files);

    bool load_pixels(const std::string &trainDatabasePath);

    void readSingleFile(std::string pathToFile, Eigen::VectorXi &testImg);

    void compute_eigenfaces();

    void debug_print_pixels();
    void debug_print_mean();
    void debug_print_centeredPixels();

    void write(const char *output_file = "matrix.out");

  };

} // namespace efj

#endif

