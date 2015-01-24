/*

    Copyright (C) 2011  Abhinav Jauhri (abhinav.jauhri@gmail.com), Carnegie Mellon University - Silicon Valley 

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

#include <iostream>
#include <algorithm>
#include <omp.h>
#include <stdlib.h>
#include <memory.h>
#include <cassert>
#include "matrix_mul.h"

using std::cout;
using std::endl;

typedef unsigned int uint;

namespace omp
{
  void
  matrix_multiplication(
      float *sq_matrix_1,
      float *sq_matrix_2,
      float *sq_matrix_result,
      unsigned int sq_dimension ) {
    // seems we can't modify sq_matrix_1 and sq_matrix_2....

    uint mm_size = sq_dimension*sq_dimension*sizeof(float);
    float * tmp = (float *) malloc (mm_size);

    for (uint i = 0; i < sq_dimension; i++) {
      for (uint j = 0; j < sq_dimension; j++) {
        tmp[j*sq_dimension+i] = sq_matrix_2[i*sq_dimension+j];
      }
    }
#pragma omp parallel for
    for (uint i = 0; i < sq_dimension; i++) {
      for (uint j = 0; j < sq_dimension; j++) {
        float sum = 0.0f;
        for (uint k = 0; k < sq_dimension; k++) {
          sum += sq_matrix_1[i*sq_dimension+k] * tmp[j*sq_dimension+k];
        }
        sq_matrix_result[i*sq_dimension+j] = sum;
      }
    }
    free(tmp);
  }
} //namespace omp
