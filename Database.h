// $Id: Database.h,v 1.16 2012/02/16 23:43:36 ferreira Exp $
//
// Copyright (C) 2008-2011 INESC ID Lisboa.
//
// This program is free software; you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation; either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program; if not, write to the Free Software
// Foundation, Inc., 59 Temple Place - Suite 330, Boston, MA 02111-1307, USA.
//
// $Log: Database.h,v $
// Revision 1.16  2012/02/16 23:43:36  ferreira
// added a new computeEigenfaces that does not filter the eigenvalues and eigenvectors
//
// Revision 1.15  2012/02/12 02:05:23  ferreira
// Added CSUFaceIDEvalSystem compatible output
//
// Revision 1.14  2011/08/15 16:36:14  david
// Updated project files to be more compatible with building installation
// packages.
//
// Revision 1.13  2011/07/22 14:44:23  david
// Minor cleanup.
//
// Revision 1.12  2011/07/22 14:32:44  ferreira
// Added Copyright comment
//

#ifndef __EFJ_DATABASE_H__
#define __EFJ_DATABASE_H__

#include <Eigen/Core>
#include <Eigen/Eigenvalues>

#include <vector>
#include <string>
#include <boost/filesystem.hpp>

#include <QtCore/QString>

#include <efj/misc.h>

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

    Eigen::MatrixXd _eigenfaces;
    int _nEigenFaces; //top n most significative eigenValues

    eigenvalue_type _eigenValues;		//needed for CSU compliance
    eigenvectors_type _eigenVectors;	//needed for CSU compliance

    Eigen::MatrixXd _clustersProjection;

  public:

    // empty database: should use "read" to fill up data.
    inline Database() {
    }

    /**
     * Constructor for training.
     */
    Database(const std::string &dir, int facesPerSubject);

    /**
     * Constructor for recognizing.
     */
    inline Database(const std::string &database) {
      read(database);
    }

    /**
     * in this directory, seach for file with given extension
     * place paths in vector
     */
    void find_files(const bf::path & dir_path, const std::string & extension,
                    std::vector<QString> &files);

    bool load_pixels(const std::string &trainDatabasePath);

    void compute_eigenfaces();
    void compute_eigenfaces_NO_FILTERING();

    void project_clusters();
    void project_single_image(Eigen::VectorXd &image, Eigen::VectorXd &projection) const;
    void compute_distance_to_groups(Eigen::VectorXd &projection, Eigen::VectorXd &distances);
    void compute_distance_to_groups(Eigen::VectorXd &projection, Eigen::VectorXd &distances,
                                    Eigen::VectorXd &results);

    bool compute_single_match_with_confidence(Eigen::VectorXd &projection,
                                              Eigen::VectorXd &distances, int &result,
                                              double &confidence) const;

    bool id(Eigen::VectorXd &image, double confidence, int &result) const {
      Eigen::VectorXd projection, distances;
      project_single_image(image, projection);
      bool recognized = compute_single_match_with_confidence(projection, distances, result,
                                                             confidence);
      //DEBUG std::cout << "** DISTANCES: " << distances << "\n";
      return recognized;
    }

    void debug_print_pixels();
    void debug_print_mean();
    void debug_print_centeredPixels();
    void debug_print_eigenvectors(const eigenvalue_type &eigenValues,
                                  const eigenvectors_type &eigenVectors);
    void debug_print_covariance_matrix(const Eigen::MatrixXd &covMatrix);

    void read(std::string input_file);
    void write(std::string output_file);

    void writeSubspace(std::string output_file);  //CSU compatible write


    int get_nGroups() {
      return _nSubjects;
    }

    void readSingleFile(QString imageFileName, Eigen::VectorXd &pixels);

  protected:
    void filter_eigenvectors(const eigenvalue_type &eigenvalues,
                             const eigenvectors_type &eigenvectors);

  };

} // namespace efj

#endif

