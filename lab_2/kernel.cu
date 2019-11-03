#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "lodepng.h"
#include "wm.h"
#include "A_10.h"
#include "B_10.h"
#include "A_32.h"
#include "B_32.h"
#include <stdio.h>
#include <time.h>

__global__ void convolution(unsigned char* image, unsigned char* new_image, unsigned type, float* weight, unsigned size, unsigned w, unsigned num_threads)
{
	// here, index is the starting index in the new image that we are modifying
	for (int index = (blockIdx.x * num_threads + threadIdx.x) * 4; index < size; index += num_threads * 4)
	{
		unsigned x = 0, y = 0;
		float rgb[3] = { 0, 0, 0 };

		// we want the x, y to be the coordinates in the theoretical matrix
		x = index / 4 % w + 1;
		y = index / (w * 4) + 1;

		if (type == 5)
		{
			x++;
			y++;
		}
		if (type == 7)
		{
			x += 2;
			y += 2;
		}

		// we want the ixy to be the starting index in the original image that we use to compute convolution
		for (int i = 0; i < type; i++)
		{
			for (int j = 0; j < type; j++)
			{
				unsigned tmp_index = 4 * (w - 1 + type) * (y + j - (type - 1) / 2) + 4 * (x + i - (type - 1) / 2);
				unsigned weight_index = type * j + i;
				rgb[0] += image[tmp_index++] * weight[weight_index];
				rgb[1] += image[tmp_index++] * weight[weight_index];
				rgb[2] += image[tmp_index] * weight[weight_index];
			}
		}

		if (rgb[0] < 0)
			rgb[0] = 0;
		else if (rgb[0] > 255)
			rgb[0] = 255;

		if (rgb[1] < 0)
			rgb[1] = 0;
		else if (rgb[1] > 255)
			rgb[1] = 255;

		if (rgb[2] < 0)
			rgb[2] = 0;
		else if (rgb[2] > 255)
			rgb[2] = 255;

		new_image[index] = rgb[0];
		new_image[index + 1] = rgb[1];
		new_image[index + 2] = rgb[2];
		new_image[index + 3] = 255;
	}
}

void do_convolution(unsigned num_blocks, unsigned num_threads)
{
	char* input_filename = "test.png";
	char* output_filename = "test_convolution.png";
	unsigned error;

	unsigned char* image, * shared_image, * new_image_convolution;
	unsigned w, h;
	unsigned type = 5;

	float* weight;

	error = lodepng_decode32_file(&image, &w, &h, input_filename);
	if (error) printf("error %u: %s\n", error, lodepng_error_text(error));

	unsigned new_w = w - type + 1, new_h = h - type + 1;

	cudaMallocManaged((void**)& shared_image, (unsigned long long)w * (unsigned long long)h * 4 * sizeof(unsigned char));
	cudaMallocManaged((void**)& new_image_convolution, ((unsigned long long)new_w) * ((unsigned long long)new_h) * 4 * sizeof(unsigned char));
	cudaMallocManaged((void**)& weight, type * type * sizeof(float));

	for (int i = 0; i < h; i++)
	{
		for (int j = 0; j < w; j++)
		{
			for (int a = 0; a < 4; a++)
			{
				int p = 4 * w * i + 4 * j + a;
				shared_image[p] = image[p];
			}
		}
	}

	if (type == 3)
	{
		for (int i = 0; i < type; i++)
		{
			for (int j = 0; j < type; j++)
			{
				weight[type * i + j] = w3[i][j];
			}
		}
	}
	else if (type == 5)
	{
		for (int i = 0; i < type; i++)
		{
			for (int j = 0; j < type; j++)
			{
				weight[type * i + j] = w5[i][j];
			}
		}
	}
	else if (type == 7)
	{
		for (int i = 0; i < type; i++)
		{
			for (int j = 0; j < type; j++)
			{
				weight[type * i + j] = w7[i][j];
			}
		}
	}
	convolution << < num_blocks, num_threads >> > (shared_image, new_image_convolution, type, weight, new_w * new_h * 4, new_w, num_threads);

	cudaDeviceSynchronize();

	lodepng_encode32_file(output_filename, new_image_convolution, new_w, new_h);

	cudaFree(shared_image);
	cudaFree(new_image_convolution);
	free(image);
}

__global__ void simplify_matrix(float* matrix, unsigned target, unsigned size, unsigned num_threads)
{
	for (int index = blockIdx.x * num_threads + threadIdx.x; index < size; index += num_threads)
	{
		float divisor = matrix[(size)* target + target];
		matrix[(size)* target + index] /= divisor;
	}
}

__global__ void reduce_matrix(float* matrix, unsigned target, unsigned size, unsigned num_threads)
{
	for (int index = blockIdx.x * num_threads + threadIdx.x; index < size; index += num_threads)
	{
		if (index == target)
			continue;

		float multiple = matrix[(size + 1) * index + target];

		for (int i = 0; i < size + 1; i++)
			matrix[(size + 1) * index + i] -= multiple * matrix[(size + 1) * target + i];

	}
}

void print_matrix(float* matrix, unsigned size)
{

	for (int i = 0; i < size; i++)
	{
		for (int j = 0; j < size + 1; j++)
		{
			printf("%.0f|", matrix[(size + 1) * i + j]);

		}
		printf("\n");
	}
	printf("\n");
}

float* do_find_solution(unsigned num_blocks, unsigned num_threads)
{
	unsigned size = 32;

	float* matrix;
	cudaMallocManaged((void**)& matrix, size * (size + 1) * sizeof(unsigned char));

	/* we need to fill the matrix with actual values, this needs to be done when changing A,b matrices */
	for (int i = 0; i < size; i++)
	{
		for (int j = 0; j < size; j++)
		{
			matrix[(size + 1) * i + j] = A_32[i][j];
		}
		matrix[(size + 1) * i + size] = b_32[i][0];
	}

	for (int i = 0; i < size; i++)
	{
		simplify_matrix << < num_blocks, num_threads >> > (matrix, i, size + 1, num_threads);
		cudaDeviceSynchronize();

		reduce_matrix << <num_blocks, num_threads >> > (matrix, i, size, num_threads);
		cudaDeviceSynchronize();
	}
	print_matrix(matrix, size);
	for (int i = 0; i < size; i++)
	{
		printf("x%d = %.4f\n", i, matrix[(size + 1) * (i + 1) - 1]);
	}
	cudaFree(matrix);

	return 0;
}

int main()
{
	unsigned num_threads = 2000;
	unsigned num_blocks = 1;

	double time_spent = 0.0;
	clock_t begin = clock();

	if (num_threads > 1024)
	{
		num_blocks = 1 + num_threads / 1024;
		num_threads = num_threads / num_blocks + 1;
	}

	//do_convolution(num_blocks, num_threads);
	do_find_solution(num_blocks, num_threads);

	clock_t end = clock();

	time_spent += (double)(end - begin) / CLOCKS_PER_SEC;

	printf("Execution time: %f\n", time_spent);

	return 0;
}