// Copyright (C) 2008-2012 INESC ID Lisboa.
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
// $Log: Database_csuOutput.cpp,v $
// Revision 1.1  2012/02/16 17:21:27  david
// Correcção de alguns bugs (David + Jaime).
//
// Revision 1.1  2012/02/12 02:05:23  ferreira
// Added CSUFaceIDEvalSystem compatible output
//


#include <efj/Database.h>
#include <cstdio>

//########################### Auxiliary ##############################

int isMachineLittleEndian() {
  int a = 0x12345678;
  unsigned char *c = (unsigned char*)(&a);
  if (*c == 0x78) {
    return 1;
  }//little endian
  else
    return 0; //big endian
}

/******************************************************************************
 *                               FILE UTILITIES                                *
 ******************************************************************************/

/*
 The following six functions are used to read and write binary information to file
 in a way that is platform independant.  Each function checks the endianness of
 the current architecture and reverses the byte order if nessessary.
 */

typedef struct {
  char a, b, c, d;
} bytes4;
typedef union {
  float n;
  bytes4 elem;
} float4;
typedef union {
  int n;
  bytes4 elem;
} int4;

typedef struct {
  char a, b, c, d, e, f, g, h;
} bytes8;
typedef union {
  double n;
  bytes8 elem;
} double8;

void writeInt(FILE* f, int n) {
  int4 tmp;
  tmp.n = n;
  if (isMachineLittleEndian()) {
    fwrite(&(tmp.elem.d), 1, 1, f);
    fwrite(&(tmp.elem.c), 1, 1, f);
    fwrite(&(tmp.elem.b), 1, 1, f);
    fwrite(&(tmp.elem.a), 1, 1, f);
  } else {
    fwrite(&(tmp.elem.a), 1, 1, f);
    fwrite(&(tmp.elem.b), 1, 1, f);
    fwrite(&(tmp.elem.c), 1, 1, f);
    fwrite(&(tmp.elem.d), 1, 1, f);
  }
}

void writeFloat(FILE* f, float n) {
  float4 tmp;
  tmp.n = n;
  if (isMachineLittleEndian()) {
    fwrite(&(tmp.elem.d), 1, 1, f);
    fwrite(&(tmp.elem.c), 1, 1, f);
    fwrite(&(tmp.elem.b), 1, 1, f);
    fwrite(&(tmp.elem.a), 1, 1, f);
  } else {
    fwrite(&(tmp.elem.a), 1, 1, f);
    fwrite(&(tmp.elem.b), 1, 1, f);
    fwrite(&(tmp.elem.c), 1, 1, f);
    fwrite(&(tmp.elem.d), 1, 1, f);
  }
}

void writeDouble(FILE* f, double n) {
  double8 tmp;
  tmp.n = n;
  if (isMachineLittleEndian()) {
    fwrite(&(tmp.elem.h), 1, 1, f);
    fwrite(&(tmp.elem.g), 1, 1, f);
    fwrite(&(tmp.elem.f), 1, 1, f);
    fwrite(&(tmp.elem.e), 1, 1, f);
    fwrite(&(tmp.elem.d), 1, 1, f);
    fwrite(&(tmp.elem.c), 1, 1, f);
    fwrite(&(tmp.elem.b), 1, 1, f);
    fwrite(&(tmp.elem.a), 1, 1, f);
  } else {
    fwrite(&(tmp.elem.a), 1, 1, f);
    fwrite(&(tmp.elem.b), 1, 1, f);
    fwrite(&(tmp.elem.c), 1, 1, f);
    fwrite(&(tmp.elem.d), 1, 1, f);
    fwrite(&(tmp.elem.e), 1, 1, f);
    fwrite(&(tmp.elem.f), 1, 1, f);
    fwrite(&(tmp.elem.g), 1, 1, f);
    fwrite(&(tmp.elem.h), 1, 1, f);
  }

}

//########## My code #############


void efj::Database::writeSubspace(std::string output_file) {

  FILE* file = fopen(output_file.c_str(), "wb");

  time_t ttt = time(0);
  const char *cutOffModeStr = "SIMPLE";
  const char *useLDA = "NO";
  float cutOff = 100;
  int dropNVectors = 0;

  if (!file) {
    printf("Error: could not open file <%s>\n", output_file.c_str());
    exit(1);
  }

  fprintf(file, "TRAINING_COMMAND =  ----IGNORED----");

  const char *imageList = "IGNORED";
  fprintf(file, "\n");
  fprintf(file, "DATE          = %s", ctime(&ttt));
  fprintf(file, "FILE_LIST     = %s\n", imageList);
  fprintf(file, "VECTOR_LENGTH = %d\n", efj::Database::_nPixels); /* numPixels */

  fprintf(file, "USE_LDA       = %s\n", useLDA);
  fprintf(file, "CUTOFF_MODE   = %s\n", cutOffModeStr);
  fprintf(file, "CUTOFF_PERCENTAGE  = %f\n", cutOff);

  fprintf(file, "BASIS_VALUE_COUNT  = %d\n", int(efj::Database::_eigenValues.rows())); /* basisDim */
  fprintf(file, "BASIS_VECTOR_COUNT = %d\n", int(efj::Database::_eigenVectors.cols()));
  fprintf(file, "DROPPED_FROM_FRONT = %d\n", dropNVectors);

  for (int i = 11; i < 256; i++) {
    fprintf(file, "\n");
  }

  /* write out the pixel count */

  writeInt(file, _mean.size());

  /* write out the mean vector */
  for (int i = 0; i < _mean.size(); i++) {
    double m = _mean(i);
    writeDouble(file, m);
  }

  /* write out the number of eigen values */
  writeInt(file, _eigenValues.rows());

  /* write out the eigen values */
  for (int i = 0; i < _eigenValues.size(); i++) {
    double m = _eigenValues(i).real();
    writeDouble(file, m);
  }

  /* write out the number of basis vectors */
  writeInt(file, _eigenVectors.cols());

  /* write out the eigen basis.  the size is "pixelcount"X"number of vectors"*/
  for (int i = 0; i < _eigenVectors.cols(); i++) {
    for (int j = 0; j < _eigenVectors.rows(); j++) {
      writeDouble(file, _eigenVectors(j, i).real());
    }
  }
  fclose(file);

}

