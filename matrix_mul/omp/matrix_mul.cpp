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

#include <omp.h>
#include "matrix_mul.h"
#include <stdio.h>
#include <string.h>
#include <xmmintrin.h>

#define NUM_OF_THREADS 8

namespace omp
{

inline void matrix_multiplication_subblock(float *m1, float *m2, float *result,unsigned int &block_size,unsigned int &sq_dimension);

void matrix_multiplication(float *sq_matrix_1, float *sq_matrix_2, float *sq_matrix_result, unsigned int sq_dimension ){
	memset(sq_matrix_result,0,sizeof(float) * sq_dimension * sq_dimension);
	bool enable_block_mul = false;
	if(sq_dimension >= 256)
		enable_block_mul = true;
	else
		enable_block_mul = false;
	if(enable_block_mul)
	{
		unsigned blk_range = 128;
		
		while(sq_dimension % blk_range != 0)
			blk_range--;
#pragma omp parallel for \
			shared(sq_matrix_1,sq_matrix_2,sq_matrix_result)\
			schedule(static)
		for(unsigned j = 0;j < sq_dimension;j+=blk_range)
		{
			for(unsigned i = 0;i < sq_dimension;i+=blk_range)
			{
				for(unsigned k = 0;k < sq_dimension;k+=blk_range)
				{
					matrix_multiplication_subblock(&sq_matrix_1[i * sq_dimension + k]\ 
													,&sq_matrix_2[k * sq_dimension + j]\
													,&sq_matrix_result[i * sq_dimension + j]\
													,blk_range,sq_dimension);
				}
			}
		}
	}
	else
	{
#pragma omp parallel for
	for (unsigned int i = 0; i < sq_dimension; i++)
	{
		for(unsigned int j = 0; j < sq_dimension; j++) 
		{
			//sq_matrix_result[i*sq_dimension + j] = 0;
			for (unsigned int k = 0; k < sq_dimension; k++)
				sq_matrix_result[i*sq_dimension + j] += sq_matrix_1[i*sq_dimension + k] * sq_matrix_2[k*sq_dimension + j];
		}
	}// End of parallel region
	}
}

	/*
	 * @Desc: This function will do muplication on two matrix, according to its parameters. 
	 * 			Generally each thread should be executing this function parallelly to get the final result.
	 * @Para m1,m2: The first matrix to be multiplied, it might be the pointer diverging from the 
	 * 				original sq_matrix_1 or sq_matrix_2. E.g , m1 = sq_matrix_1 + sq_dimension and m2 = sq_matrix_2 + sq_dimension 
	 * @Para result: Where the result should be stored, the actual return value
	 * @Para block_size: The size of the sub block where the actual multiplcaition is performed
	 * @Para sq_dimension: The dimension of the square matrix
	 *
	 */
inline void matrix_multiplication_subblock(float *m1, float *m2, float *result,unsigned int &block_size,unsigned int &sq_dimension)
{
	//result[row][col] would be the corresponding element in the resulting matrix
	unsigned int row, col, count;
	for(row = 0;row < block_size;row++)
		for(col = 0;col < block_size;col++)
			for(count = 0;count < block_size;)
			{
				if(count + 4 < block_size)
				{
					result[row * sq_dimension + col] += (m1[row * sq_dimension + count] * m2[count * sq_dimension + col]\
																+m1[row * sq_dimension + count + 1] * m2[(count + 1) * sq_dimension + col]\
																+m1[row * sq_dimension + count + 2] * m2[(count + 2) * sq_dimension + col]\
																+m1[row * sq_dimension + count + 3] * m2[(count + 3) * sq_dimension + col]);
					count+=4;
				}
				else
				{
					result[row * sq_dimension + col] += m1[row * sq_dimension + count] * m2[count * sq_dimension + col];
					count++;
				}
			}
	return ;
}

} //namespace omp
