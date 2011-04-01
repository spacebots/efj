#ifndef __EFJ_DATABASE_H__
#define __EFJ_DATABASE_H__

#include <vector>
#include <string>
#include <boost/filesystem.hpp>
#include <Eigen/Core>
#include <Eigen/Eigenvalues>
#include <QString>

#include "misc.h"

namespace bf = boost::filesystem;

namespace efj {

  class Database {

  public:
    typedef Eigen::EigenSolver<Eigen::MatrixXd>::EigenvalueType eigenvalue_type;
    typedef Eigen::EigenSolver<Eigen::MatrixXd>::EigenvectorsType eigenvectors_type;

  private:
    // this matrix may become so large it won't fit in the stack
    // so, we allocate it on the heap
    Eigen::MatrixXd _pixels; // M*N x P

    int _nImages; // typically, _pixels->cols()
    int _nPixels; // typically, _pixels->rows()

    int _facesPerSubject; // how many faces per subject
    int _nSubjects; // number of subjects

    Eigen::VectorXd _mean; // uppercase psi
    Eigen::MatrixXd _centeredPixels; //[M*N x P] - [mean]
    Eigen::MatrixXd _centeredPixelsFiltered;

    Eigen::MatrixXd _eigenfaces;
    int _nEigenFaces; //top n most significative eigenValues

    Eigen::MatrixXd _clustersProjection;

  public:

    // empty database: should use "read" to fill up data.
    inline Database() {
    }

    //dir = "/afs/l2f.inesc-id.pt/home/ferreira/face-recognition/MyTrainDatabase"
    //dir = "/afs/l2f.inesc-id.pt/home/ferreira/FaceRec/ImageVault/train";
    Database(const std::string &dir, int facesPerSubject);

    /**
     * in this directory, seach for file with given extension
     * place paths in vector
     */
    void find_files(const bf::path & dir_path, const std::string & extension,
                    std::vector<QString> &files);

    bool load_pixels(const std::string &trainDatabasePath);

    void compute_eigenfaces();

    void project_clusters();
    void project_single_image(Eigen::VectorXd &image, Eigen::VectorXd &projection);
    void compute_distance_to_groups(Eigen::VectorXd &projection, Eigen::VectorXd &distances);

    void debug_print_pixels();
    void debug_print_mean();
    void debug_print_centeredPixels();
    void debug_print_eigenvectors(const eigenvalue_type &eigenValues,
                                  const eigenvectors_type &eigenVectors);
    void debug_print_covariance_matrix(const Eigen::MatrixXd &covMatrix);

    void read(std::string input_file);
    void write(std::string output_file);

    int get_nGroups() {
      return _nSubjects;
    }

    int get_grouping() {
      return _facesPerSubject;
    }

    static void readSingleFile(QString imageFileName, Eigen::VectorXd &pixels);

  protected:
    void filter_eigenvectors(const eigenvalue_type &eigenvalues, const eigenvectors_type &eigenvectors);

  };

} // namespace efj

#endif

