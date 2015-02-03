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




void
    matrix_multiplication(
        float *sq_matrix_1,
        float *sq_matrix_2,
        float *sq_matrix_result,
        unsigned int sq_dimension ) {
      // seems we can't modify sq_matrix_1 and sq_matrix_2....

      cout << "\n" << uintptr_t(sq_matrix_1) << endl;
      assert(uintptr_t(sq_matrix_1) % 16 == 0);
      uint mask4 = 0x3;
      if ((sq_dimension & mask4) > 0) {
        uint dim4 = sq_dimension + 4 - (sq_dimension & mask4);
        float *sq1 = (float *)malloc(dim4*dim4*sizeof(float));
        float *sq2_t = (float *)malloc(dim4*dim4*sizeof(float));
        assert(uintptr_t(sq1) % 16 == 0);
        assert(uintptr_t(sq2_t) % 16 == 0);
        for (uint i = 0; i < sq_dimension; i++) {
          uint base_i4 = i*dim4;
          uint base_i = i*sq_dimension;
          for (uint j = 0; j < sq_dimension; j++) {
            sq1[base_i4 + j] = sq_matrix_1[base_i+j];
            sq2_t[j*dim4 + i] = sq_matrix_2[base_i+j];
          }
          for (uint j = sq_dimension; j < dim4; j++) {
            sq1[base_i4 + j] = 0;
            sq2_t[j*dim4 + i] = 0;
          }
        }
        for (uint i = sq_dimension; i < dim4; i++) {
          uint base_i4 = i*dim4;
          uint base_i = i*sq_dimension;
          for (uint j = 0; j < sq_dimension; j++) {
            sq1[base_i4 + j] = sq_matrix_1[base_i+j];
            sq2_t[j*dim4 + i] = sq_matrix_2[base_i+j];
          }
          for (uint j = sq_dimension; j < dim4; j++) {
            sq1[base_i4 + j] = 0;
            sq2_t[j*dim4 + i] = 0;
          }
        }
#pragma omp parallel for
        for (uint i = 0; i < sq_dimension; i++) {
          for (uint j = 0; j < sq_dimension; j++) {
            __m128 sum128 = _mm_setzero_ps();
            __m128 X, Y;
            uint base_i4 = i * dim4;
            uint base_j4 = j * dim4;
            for (uint k = 0; k < sq_dimension; k+=4) {
              X = _mm_load_ps(&sq1[base_i4+k]);
              Y = _mm_load_ps(&sq2_t[base_j4+k]);
              sum128 = _mm_add_ps(sum128, _mm_mul_ps(X, Y));
            }
            float sum_temp[4];
            _mm_store_ps(sum_temp, sum128);
            float sum_32 = sum_temp[0] + sum_temp[1] + sum_temp[2] + sum_temp[3];
            sq_matrix_result[i*sq_dimension+j] = sum_32;
          }
        }
        free(sq1);
        free(sq2_t);
      } else {
        float *sq2_t = (float *)malloc(sq_dimension*sq_dimension*sizeof(float));
        for (uint i = 0; i < sq_dimension; i++) {
          uint base_i = i * sq_dimension;
          for (uint j = 0; j < sq_dimension; j++) {
            sq2_t[j*sq_dimension+i ] = sq_matrix_2[base_i+j];
          }
        }
#pragma omp parallel for
        for (uint i = 0; i < sq_dimension; i++) {
          for (uint j = 0; j < sq_dimension; j++) {
            __m128 sum128 = _mm_setzero_ps();
            __m128 X, Y;
            uint base_i = i * sq_dimension;
            uint base_j = j * sq_dimension;
            for (uint k = 0; k < sq_dimension; k+=4) {
              X = _mm_load_ps(&sq_matrix_1[base_i+k]);
              Y = _mm_load_ps(&sq2_t[base_j+k]);
              sum128 = _mm_add_ps(sum128, _mm_mul_ps(X,Y));
            }
            float sum_temp[4];
            _mm_store_ps(sum_temp, sum128);
            float sum32 = sum_temp[0] + sum_temp[1] + sum_temp[2] + sum_temp[3];
            sq_matrix_result[i*sq_dimension + j] = sum32;
          }
        }
        free(sq2_t);
      }
    }
} //namespace omp
