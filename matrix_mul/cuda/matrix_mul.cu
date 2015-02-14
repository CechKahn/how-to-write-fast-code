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

#define SHARED_MEM_SIZE 49152
#define OPTIMIZE_OPT 5
namespace cuda
{
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

	//this one does tranpose on sq_m2 for a quicker matrix multiplication
	__global__
	void
	matrixMulV2(float *sq_m1, float *sq_m2, float *sq_m3, int sq_dim)
	{
		int row = blockIdx.y * blockDim.y + threadIdx.y;
		int col = blockIdx.x * blockDim.x + threadIdx.x;
		float sum = 0.0f;
		if(row >= sq_dim || col >= sq_dim)
			return ;
		int i = 0;
		for(i = 0; i < sq_dim;i++)
		{
			sum += sq_m1[row*sq_dim + i] * sq_m2[col*sq_dim + i];
		}
		sq_m3[row * sq_dim + col] = sum;
		return ;
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
			sum+=sq_m1[i * sq_dim + row] * sq_m2[i*sq_dim + col];
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


	//try to use block matrix multiplication and shared memory
	__global__
	void
	matrixMulV5(float *sq_m1, float *sq_m2, float *sq_m3, int sq_dim, int submat_blk_size)
	{
		int blk_cnt = 0;
		//m should be of size submat_blk_size * submat_blk_size * 2
		extern __shared__ float m[];
		float *m2 = m + submat_blk_size * submat_blk_size;
		int row = blockIdx.y * blockDim.y + threadIdx.y;
		int col = blockIdx.x * blockDim.x + threadIdx.x;
		
		if(row >= sq_dim || col >= sq_dim)
			return ;
		float sum = 0.0f;
		for(blk_cnt = 0; blk_cnt < sq_dim / submat_blk_size;blk_cnt++)
		{
			//block matrix multiplication with shared memory	
			//each thread load one element into sub matrix in sq_m1 and one element into submatrix into sq_m2
			m[threadIdx.y * submat_blk_size + threadIdx.x] = sq_m1[row * sq_dim + col + blk_cnt * submat_blk_size];
			m2[threadIdx.y * submat_blk_size + threadIdx.x] = sq_m2[(row + blk_cnt * submat_blk_size) * sq_dim + col];
			__syncthreads();
			
			for(int k = 0; k < submat_blk_size;k++)
					sum += m[threadIdx.y * submat_blk_size + k] * m2[k * submat_blk_size + threadIdx.x];
			__syncthreads();
		
		/*
			//block matrix multiplication without shared memory
			int max_k = (blk_cnt+1) * submat_blk_size;
			for(int k = blk_cnt * submat_blk_size; k < max_k;k++)
				sum += sq_m1[row * sq_dim + k] * sq_m2[k * sq_dim + col];
		*/
		}
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

#if OPTIMIZE_OPT == 2
			cudaMemcpy(sq_matrix_1_d, sq_matrix_1, size, cudaMemcpyHostToDevice);
			dim3 dimBlock(MAX_BLK_DIM,MAX_BLK_DIM);
			int gridSizeX = (sq_dimension / dimBlock.x) + 1, gridSizeY = (sq_dimension/dimBlock.y) + 1;
			dim3 dimGrid(gridSizeX, gridSizeY);
			//transpose the second matrix
			int i = 0, j = 0;
			float *sq_matrix_2t = (float*) malloc(size * sizeof(float));
			float tmp = 0.0f;
			for(i = 0; i < sq_dimension;i++)
			{
				for(j = 0;j < sq_dimension;j++)
				{
					sq_matrix_2t[j * sq_dimension + i] = sq_matrix_2[i * sq_dimension + j];
				}
			}
			cudaMemcpy(sq_matrix_2_d, sq_matrix_2t, size, cudaMemcpyHostToDevice);
			free(sq_matrix_2t);
			matrixMulV2<<<dimGrid, dimBlock>>>(sq_matrix_1_d, sq_matrix_2_d, sq_matrix_result_d, sq_dimension);
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
			//transpose the 1st matrix
			int i = 0, j = 0;
			float *sq_matrix_1t = (float*) malloc(size * sizeof(float));
			float tmp = 0.0f;
			for(i = 0; i < sq_dimension;i++)
			{
				for(j = 0;j < sq_dimension;j++)
				{
					sq_matrix_1t[j * sq_dimension + i] = sq_matrix_1[i * sq_dimension + j];
				}
			}
			cudaMemcpy(sq_matrix_1_d, sq_matrix_1t, size, cudaMemcpyHostToDevice);
			cudaMemcpy(sq_matrix_2_d, sq_matrix_2, size, cudaMemcpyHostToDevice);
			free(sq_matrix_1t);
			matrixMulV4<<<dimGrid, dimBlock>>>(sq_matrix_1_d, sq_matrix_2_d, sq_matrix_result_d, sq_dimension);
#endif

#if OPTIMIZE_OPT == 5
		dim3 dimBlock(MAX_BLK_DIM,MAX_BLK_DIM);
		dim3 dimGrid(sq_dimension/dimBlock.x+1, sq_dimension/dimBlock.y+1);
		cudaMemcpy(sq_matrix_1_d,sq_matrix_1,size,cudaMemcpyHostToDevice);
		cudaMemcpy(sq_matrix_2_d,sq_matrix_2,size,cudaMemcpyHostToDevice);
		int submat_blk_size = 32;
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
