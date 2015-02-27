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


/*
  The optimization is done by Edward ZILONG HUANG
*/
#include <cuda.h>
#include <cuda_runtime.h>
#include "matrix_mul.h"
#include "stdio.h"

#define TILE_WIDTH 2
#define MAX_BLK_DIM 32

#define SUBMAT_BLK_SIZE 32

#define SHARED_MEM_SIZE 49152
#define OPTIMIZE_OPT 1
namespace cuda
{
	//this function generate a transposed matrix 
	float* transposeMatrix(float *m, int sq_dim)
	{
		//transpose the matrix
		int i = 0, j = 0;
		float *result_m = (float*) malloc(sq_dim*sq_dim * sizeof(float));
		for(i = 0; i < sq_dim;i++)
		{
			for(j = 0;j < sq_dim;j++)
			{
				result_m[j * sq_dim + i] = m[i * sq_dim + j];
			}
		}
		return result_m;
	}

	void showMatrix(float *m, int sq_dim)
	{
		int i = 0, j = 0;
		printf("\n");
		for(i = 0; i < sq_dim;i++){
			printf("row %d\n",i);
			for(j = 0; j < sq_dim;j++)
				printf("%f\t",m[i*sq_dim + j]);
			printf("\n");
		}
		printf("\n");
	}
	__global__ 
	void 
	matrix_mul_kernel(float *sq_matrix_1, float *sq_matrix_2, float *sq_matrix_result, int sq_dimension)
	{
		
		int tx = threadIdx.x;
		int ty = threadIdx.y;
		
		float sum = 0.0f;
		
		for(int k = 0; k < sq_dimension; k++)
		{
			sum += sq_matrix_1[ty*sq_dimension + k] * sq_matrix_2[k*sq_dimension + tx];
		}
		sq_matrix_result[ty*sq_dimension + tx] = sum;
		
	}

	__global__
	void
	matrixMulV1(float *sq_m1, float *sq_m2, float *sq_m3, int sq_dim)
	{
		//memory coaleasing because each thread is accessing sq_m2 in different memory bank
		//because threadIdx.x is typically different
		int row = blockIdx.y * blockDim.y + threadIdx.y;
		int col = blockIdx.x * blockDim.x + threadIdx.x;
		
		float sum = 0.0f;
		if(row >= sq_dim || col >= sq_dim)
			return ;
		int i = 0;
		for(i = 0;i < sq_dim;i++)
		{
			sum+=sq_m1[row * sq_dim + i] * sq_m2[i*sq_dim + col];
		}
		sq_m3[row * sq_dim + col] = sum;
	}

	__global__
	void
	matrixMulV3(float *sq_m1, float *sq_m2, float *sq_m3, int sq_dim)
	{
		int row = blockIdx.y * blockDim.y + threadIdx.y * TILE_WIDTH;
		int col = blockIdx.x * blockDim.x + threadIdx.x * TILE_WIDTH;
		
		float sum = 0.0f;
		if(row >= sq_dim || col >= sq_dim)
			return ;
		int i = 0, j = 0, k = 0;
		for(j = col; j < col + TILE_WIDTH;j++)
			for(i = row; i < row + TILE_WIDTH;i++)
			{
				sum = 0.0f;
				for(k = 0; k < sq_dim;k++)
					sum+=sq_m1[i * sq_dim + k] * sq_m2[k*sq_dim + j];
				sq_m3[i * sq_dim + j] = sum;
			}
	}

	//this one try to transpose sq_m1 rather than transpose sq_m2
	__global__
	void
	matrixMulV4(float *sq_m1, float *sq_m2, float *sq_m3, int sq_dim)
	{
		//memory coaleasing because each thread is accessing sq_m2 in different memory bank
		//because threadIdx.x is typically different
		int row = blockIdx.y * blockDim.y + threadIdx.y;
		int col = blockIdx.x * blockDim.x + threadIdx.x;
		
		float sum = 0.0f;
		if(row >= sq_dim || col >= sq_dim)
			return ;
		int i = 0;
		for(i = 0;i < sq_dim;i++)
		{
			sum+=sq_m1[row * sq_dim + i] * sq_m2[col*sq_dim + i];
		}
		sq_m3[row * sq_dim + col] = sum;
	}


	//try to use block matrix multiplication and shared memory
	__global__
	void
	matrixMulV5(float *sq_m1, float *sq_m2, float *sq_m3, int sq_dim, int submat_blk_size)
	{
		int blk_cnt = 0;
		//m should be of size submat_blk_size * submat_blk_size * 2
		
		__shared__ float m[SUBMAT_BLK_SIZE][SUBMAT_BLK_SIZE];
		__shared__ float m2[SUBMAT_BLK_SIZE][SUBMAT_BLK_SIZE];
		//float *m2 = m + submat_blk_size * submat_blk_size;
		int row = blockIdx.y * blockDim.y + threadIdx.y;
		int col = blockIdx.x * blockDim.x + threadIdx.x;
		
		float sum = 0.0f;
		
		int max_blk_cnt;
		int idx1, idx2, size = sq_dim * sq_dim;
		if(sq_dim % submat_blk_size == 0)
			max_blk_cnt = sq_dim / submat_blk_size;
		else
			max_blk_cnt = sq_dim / submat_blk_size+1;
		
		for(blk_cnt = 0; blk_cnt < max_blk_cnt;blk_cnt++)
		{
			//block matrix multiplication with shared memory	
			//each thread load one element into sub matrix in sq_m1 and one element into submatrix into sq_m2
			//for matrix non-multiple of block_size,
			//bring zero for those un-unified position
			idx1 = row * sq_dim + threadIdx.x + blk_cnt * submat_blk_size;
			idx2 = (threadIdx.y + blk_cnt * submat_blk_size) * sq_dim + col;
			if(idx1 < size && row < sq_dim) 
				m[threadIdx.y][threadIdx.x] = sq_m1[idx1];
			else
				m[threadIdx.y][threadIdx.x] = 0.0f;
			if(idx2 < size && col < sq_dim)
				m2[threadIdx.y][threadIdx.x] = sq_m2[idx2];
			else
				m2[threadIdx.y][threadIdx.x] = 0.0f;
			__syncthreads();
		
			//do computation based on that shared memory content
			for(int k = 0; k < submat_blk_size;k++)
					sum += m[threadIdx.y][k] * m2[k][threadIdx.x];
			__syncthreads();
		}
		if(row < sq_dim && col < sq_dim)
			sq_m3[row * sq_dim + col] = sum;
		return ;
	}
	
	//try to use block matrix multiplication, shared memory and transposed sq_m1
	__global__
	void
	matrixMulV6(float *sq_m1, float *sq_m2, float *sq_m3, int sq_dim, int submat_blk_size)
	{
		int blk_cnt = 0;
//		__shared__ float m[SUBMAT_BLK_SIZE][SUBMAT_BLK_SIZE];
//		__shared__ float m2[SUBMAT_BLK_SIZE][SUBMAT_BLK_SIZE];
		int row = blockIdx.x * blockDim.x + threadIdx.x;
		int col = blockIdx.y * blockDim.y + threadIdx.y;

		float sum = 0.0f;
/*
		int max_blk_cnt;
		int idx1, size = sq_dim * sq_dim;
		if(sq_dim % submat_blk_size == 0)
			max_blk_cnt = sq_dim / submat_blk_size;
		else
			max_blk_cnt = sq_dim / submat_blk_size+1;
	
		for(blk_cnt = 0; blk_cnt < max_blk_cnt;blk_cnt++)
		{
			//block matrix multiplication with shared memory	
			//each thread load one element into sub matrix in sq_m1 and one element into submatrix into sq_m2
			//for matrix non-multiple of block_size,
			//bring zero for those un-unified position
			idx1 = row * sq_dim + (blk_cnt * SUBMAT_BLK_SIZE + threadIdx.y);
			if(idx1 < size && row < sq_dim) 
			{
				m[threadIdx.x][threadIdx.y] = sq_m1[idx1];
				m2[threadIdx.x][threadIdx.y] = sq_m2[idx1];
			}
			else
			{
				m[threadIdx.x][threadIdx.y] = 0.0f;
				m2[threadIdx.x][threadIdx.y] = 0.0f;
			}
			__syncthreads();
		
			//do computation based on that shared memory content
			for(int k = 0; k < submat_blk_size;k++)
					sum += (m[threadIdx.x][k] * m2[threadIdx.x][k]);
			__syncthreads();
		}
	*/

		for(int k = 0; k < sq_dim;k++)
			sum+= (sq_m1[row * sq_dim + k] * sq_m2[col * sq_dim + k]);
		if(row < sq_dim && col < sq_dim)
			sq_m3[row * sq_dim + col] = sum;
		return ;
	}

	//try to use block matrix multiplication and shared memory
	__global__
	void
	matrixMulV7(float *input_m1, float *input_m2, float *output_m3, int dim, int blk_size)
	{
		int blk_cnt = 0;
		//m should be of size submat_blk_size * submat_blk_size * 2
		
		__shared__ float *sq_m1;
		__shared__ float *sq_m2;
		__shared__ float *sq_m3;
		__shared__ int sq_dim;
		__shared__ int submat_blk_size;

		sq_m1 = input_m1;
		sq_m2 = input_m2;
		sq_m3 = output_m3;
		sq_dim = dim;
		submat_blk_size = blk_size;
		__shared__ float m[SUBMAT_BLK_SIZE][SUBMAT_BLK_SIZE];
		__shared__ float m2[SUBMAT_BLK_SIZE][SUBMAT_BLK_SIZE];
		//float *m2 = m + submat_blk_size * submat_blk_size;
		int row = blockIdx.y * blockDim.y + threadIdx.y;
		int col = blockIdx.x * blockDim.x + threadIdx.x;
		
		float sum = 0.0f;
		
		int max_blk_cnt;
		int idx1, idx2, size = sq_dim * sq_dim;
		if(sq_dim % submat_blk_size == 0)
			max_blk_cnt = sq_dim / submat_blk_size;
		else
			max_blk_cnt = sq_dim / submat_blk_size+1;
		
		for(blk_cnt = 0; blk_cnt < max_blk_cnt;blk_cnt++)
		{
			//block matrix multiplication with shared memory	
			//each thread load one element into sub matrix in sq_m1 and one element into submatrix into sq_m2
			//for matrix non-multiple of block_size,
			//bring zero for those un-unified position
			idx1 = row * sq_dim + threadIdx.x + blk_cnt * submat_blk_size;
			idx2 = (threadIdx.y + blk_cnt * submat_blk_size) * sq_dim + col;
			if(idx1 < size && row < sq_dim) 
				m[threadIdx.y][threadIdx.x] = sq_m1[idx1];
			else
				m[threadIdx.y][threadIdx.x] = 0.0f;
			if(idx2 < size && col < sq_dim)
				m2[threadIdx.y][threadIdx.x] = sq_m2[idx2];
			else
				m2[threadIdx.y][threadIdx.x] = 0.0f;
			__syncthreads();
		
			//do computation based on that shared memory content
			for(int k = 0; k < submat_blk_size;k++)
					sum += m[threadIdx.y][k] * m2[k][threadIdx.x];
			__syncthreads();
		}
		if(row < sq_dim && col < sq_dim)
			sq_m3[row * sq_dim + col] = sum;
		return ;
	}

	void 
	matrix_multiplication(float *sq_matrix_1, float *sq_matrix_2, float *sq_matrix_result, unsigned int sq_dimension)
	{
		int size = sq_dimension * sq_dimension * sizeof(float);
		float *sq_matrix_1_d, *sq_matrix_2_d, *sq_matrix_result_d;
		/***************************************************
	1st Part: Allocation of memory on device memory  
		****************************************************/
		
		/* copy sq_matrix_1 and sq_matrix_2 to device memory */
		cudaMalloc((void**) &sq_matrix_1_d, size);
		
		cudaMalloc((void**) &sq_matrix_2_d, size);
		
		/*allocate sq_matrix_result on host */
		cudaMalloc((void**) &sq_matrix_result_d, size);
		
		/***************************************************
	 2nd Part: Inovke kernel 
		****************************************************/
		if(sq_dimension <= 32)
		{
			cudaMemcpy(sq_matrix_1_d, sq_matrix_1, size, cudaMemcpyHostToDevice);
			cudaMemcpy(sq_matrix_2_d, sq_matrix_2, size, cudaMemcpyHostToDevice);
			dim3 dimBlock(sq_dimension,sq_dimension);
			dim3 dimGrid(1,1);
			matrix_mul_kernel<<<dimGrid,dimBlock>>>(sq_matrix_1_d, sq_matrix_2_d, sq_matrix_result_d, sq_dimension);
		}
		else
		{

#if OPTIMIZE_OPT == 1
			//CUDA Version 1, only correctness is ensured here
			cudaMemcpy(sq_matrix_1_d, sq_matrix_1, size, cudaMemcpyHostToDevice);
			cudaMemcpy(sq_matrix_2_d, sq_matrix_2, size, cudaMemcpyHostToDevice);
			int blockSizeX = MAX_BLK_DIM, blockSizeY = MAX_BLK_DIM;
			dim3 dimBlock(blockSizeX, blockSizeY);
			int gridSizeX = (sq_dimension / blockSizeX) + 1, gridSizeY = (sq_dimension/blockSizeY) + 1;
			dim3 dimGrid(gridSizeX,gridSizeY);
			matrixMulV1<<<dimGrid, dimBlock, sq_dimension * sizeof(float)>>>(sq_matrix_1_d, sq_matrix_2_d, sq_matrix_result_d, sq_dimension);
#endif


#if OPTIMIZE_OPT == 3
		dim3 dimBlock(MAX_BLK_DIM/TILE_WIDTH, MAX_BLK_DIM/TILE_WIDTH);
		dim3 dimGrid((sq_dimension/dimBlock.x)+1,(sq_dimension/dimBlock.y)+1);
		cudaMemcpy(sq_matrix_1_d, sq_matrix_1, size, cudaMemcpyHostToDevice);
		cudaMemcpy(sq_matrix_2_d, sq_matrix_2, size, cudaMemcpyHostToDevice);
		matrixMulV3<<<dimGrid, dimBlock>>>(sq_matrix_1_d, sq_matrix_2_d, sq_matrix_result_d,sq_dimension);
#endif

#if OPTIMIZE_OPT == 4
			dim3 dimBlock(MAX_BLK_DIM,MAX_BLK_DIM);
			int gridSizeX = (sq_dimension / dimBlock.x) + 1, gridSizeY = (sq_dimension/dimBlock.y) + 1;
			dim3 dimGrid(gridSizeX, gridSizeY);
			
			float *m_t = transposeMatrix(sq_matrix_2,sq_dimension);
			cudaMemcpy(sq_matrix_1_d, sq_matrix_1, size, cudaMemcpyHostToDevice);
			cudaMemcpy(sq_matrix_2_d, m_t, size, cudaMemcpyHostToDevice);
			free(m_t);
			matrixMulV4<<<dimGrid, dimBlock>>>(sq_matrix_1_d, sq_matrix_2_d, sq_matrix_result_d, sq_dimension);
#endif

#if OPTIMIZE_OPT == 5
		int blk_size = 32;
		dim3 dimBlock(blk_size,blk_size);
		int gridX, gridY;
		if(sq_dimension % dimBlock.x == 0)
			gridX = sq_dimension/dimBlock.x;
		else 
			gridX = sq_dimension/dimBlock.x + 1;
		if(sq_dimension % dimBlock.y == 0)
			gridY = sq_dimension/dimBlock.y;
		else
			gridY = sq_dimension/dimBlock.y + 1;

		dim3 dimGrid(gridX, gridY);
		//printf("\ndimGridX %d, dimGridY %d\n",dimGrid.x,dimGrid.y);
		cudaMemcpy(sq_matrix_1_d,sq_matrix_1,size,cudaMemcpyHostToDevice);
		cudaMemcpy(sq_matrix_2_d,sq_matrix_2,size,cudaMemcpyHostToDevice);
		int submat_blk_size = blk_size;
		matrixMulV5<<<dimGrid, dimBlock>>>(sq_matrix_1_d, sq_matrix_2_d, sq_matrix_result_d, sq_dimension, submat_blk_size);
#endif


#if OPTIMIZE_OPT == 6
		int blk_size = 32;
		dim3 dimBlock(blk_size,blk_size);
		int gridX, gridY;
		if(sq_dimension % dimBlock.x == 0)
			gridX = sq_dimension/dimBlock.x;
		else 
			gridX = sq_dimension/dimBlock.x + 1;
		if(sq_dimension % dimBlock.y == 0)
			gridY = sq_dimension/dimBlock.y;
		else
			gridY = sq_dimension/dimBlock.y + 1;

		dim3 dimGrid(gridX, gridY);

		float *m_t = transposeMatrix(sq_matrix_2,sq_dimension);
		cudaMemcpy(sq_matrix_1_d,sq_matrix_1,size,cudaMemcpyHostToDevice);
		cudaMemcpy(sq_matrix_2_d,m_t,size,cudaMemcpyHostToDevice);
		int submat_blk_size = SUBMAT_BLK_SIZE;
		matrixMulV6<<<dimGrid, dimBlock>>>(sq_matrix_1_d, sq_matrix_2_d, sq_matrix_result_d, sq_dimension, submat_blk_size);
		free(m_t);
#endif

#if OPTIMIZE_OPT == 7
		int blk_size = 32;
		dim3 dimBlock(blk_size,blk_size);
		int gridX, gridY;
		if(sq_dimension % dimBlock.x == 0)
			gridX = sq_dimension/dimBlock.x;
		else 
			gridX = sq_dimension/dimBlock.x + 1;
		if(sq_dimension % dimBlock.y == 0)
			gridY = sq_dimension/dimBlock.y;
		else
			gridY = sq_dimension/dimBlock.y + 1;

		dim3 dimGrid(gridX, gridY);
		//printf("\ndimGridX %d, dimGridY %d\n",dimGrid.x,dimGrid.y);
		cudaMemcpy(sq_matrix_1_d,sq_matrix_1,size,cudaMemcpyHostToDevice);
		cudaMemcpy(sq_matrix_2_d,sq_matrix_2,size,cudaMemcpyHostToDevice);
		int submat_blk_size = blk_size;
		matrixMulV5<<<dimGrid, dimBlock, sizeof(float) * submat_blk_size * submat_blk_size * 2>>>(sq_matrix_1_d, sq_matrix_2_d, sq_matrix_result_d, sq_dimension, submat_blk_size);
#endif

		}
		/***************************************************
	 3rd Part: Transfer result from device to host 
		****************************************************/
		cudaMemcpy(sq_matrix_result, sq_matrix_result_d, size, cudaMemcpyDeviceToHost);
		cudaFree(sq_matrix_1_d);
		cudaFree(sq_matrix_2_d);
		cudaFree(sq_matrix_result_d);
	}  
} // namespace cuda
