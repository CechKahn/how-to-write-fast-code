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
#include <stdlib.h>
#include <memory.h>
#include <cassert>
#include <omp.h>
#include <inttypes.h>
#include "matrix_mul.h"

extern "C"
{
#include <pmmintrin.h>
}

using std::cout;
using std::endl;

typedef unsigned int uint;

inline uint greatestBitValue(uint x) {
  x |= (x >> 1);
  x |= (x >> 2);
  x |= (x >> 4);
  x |= (x >> 8);
  x |= (x >> 16);
  return x & ~(x>>1);
}

namespace omp
{

  void matrix_multiplication(
      float *sq_matrix_1,
      float *sq_matrix_2,
      float *sq_matrix_result,
      unsigned int sq_dimension ) {
    // seems we can't modify sq_matrix_1 and sq_matrix_2....

    bool aligned = true;
    uint mask4 = 0x3;
    // use bit operation to determin whether sq_dimension is multiple of 4
    if ((sq_dimension & mask4) > 0) {
      // dimension that's a multiple of 4
      uint dim4 = sq_dimension + 4 - (sq_dimension & mask4);

      // sq1 is the aligned version of sq_matrix_1
      float *sq1 = (float *)calloc(dim4*dim4, sizeof(float));
      // sq2_t is the transform of sq_matrix_2(also aligned)
      float *sq2_t = (float *)calloc(dim4*dim4, sizeof(float));
      for (uint i = 0; i < sq_dimension; i++) {
        uint base_i4 = i*dim4;
        uint base_i = i*sq_dimension;
        for (uint j = 0; j < sq_dimension; j++) {
          // align sq_matrix_1 data
          sq1[base_i4 + j] = sq_matrix_1[base_i+j];
          // Transform sq_matrix_2
          sq2_t[j*dim4 + i] = sq_matrix_2[base_i+j];
        }
      }
      aligned = false;
#pragma omp parallel for
      for (uint i = 0; i < sq_dimension; i++) {
        for (uint j = 0; j < sq_dimension; j++) {
          float sum = 0;
          uint base_i4 = i * dim4;
          uint base_j4 = j * dim4;
          for (uint k = 0; k < sq_dimension; k += 4) {
            sum += sq1[base_i4+0] * sq2_t[base_j4+0]
            sum += sq1[base_i4+1] * sq2_t[base_j4+1]
            sum += sq1[base_i4+2] * sq2_t[base_j4+2]
            sum += sq1[base_i4+3] * sq2_t[base_j4+3]
          }
          sq_matrix_result[i*sq_dimension+j] = sum;
        }
      }
      free(sq1);
      free(sq2_t);
    } else {
      float *sq1 = sq_matrix_1;
      // transform of sq_matrix_2
      float *sq2_t = (float *)malloc(sq_dimension*sq_dimension*sizeof(float));
      for (uint i = 0; i < sq_dimension; i++) {
        uint base_i = i * sq_dimension;
        for (uint j = 0; j < sq_dimension; j++) {
          sq2_t[j*sq_dimension+i] = sq_matrix_2[base_i+j];
        }
      }
#pragma omp parallel for
      for (uint i = 0; i < sq_dimension; i++) {
        for (uint j = 0; j < sq_dimension; j++) {
          float sum = 0;
          uint base_i = i * sq_dimension;
          uint base_j = j * sq_dimension;
          for (uint k = 0; k < sq_dimension; k+=4) {
            sum += sq1[base_i+0] * sq2_t[base_j+0]
            sum += sq1[base_i+1] * sq2_t[base_j+1]
            sum += sq1[base_i+2] * sq2_t[base_j+2]
            sum += sq1[base_i+3] * sq2_t[base_j+3]
          }
          sq_matrix_result[i*sq_dimension + j] = sum;
        }
      }
      free(sq2_t);
    }
  }
} //namespace omp
