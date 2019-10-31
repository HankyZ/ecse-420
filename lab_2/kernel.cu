#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "lodepng.h"
#include "wm.h"
#include <stdio.h>
#include <time.h>

__global__ void convolution(unsigned char* image, unsigned char* new_image, unsigned type, unsigned size, unsigned w, unsigned num_threads)
{
	// here, index is the starting index in the new image that we are modifying
	for (int index = (blockIdx.x * num_threads + threadIdx.x) * 4; index < size; index += num_threads * 4)
	{
		unsigned x, y, rgba[4];

		// we want the x, y to be the coordinates in the theoretical matrix
		x = index % (w * 4) + 1;
		y = index / (w * 4) + 1;

		// we want the ixy to be the starting index in the original image that we use to compute convolution
		for (int i = 0; i < type; i++)
		{
			for (int j = 0; j < type; j++)
			{
				for (int a = 0; a < 4; a++)
				{
					rgba[a] += image[4 * w * (y + j - 1) + 4 * (x + i - 1) + a] * w3[i][j];
				}
			}
		}

		for (int a = 0; a < 4; a++)
		{
			new_image[index + a] = rgba[a];
		}
	}
}


void do_convolution(unsigned char* image, unsigned w, unsigned h, char* output_filename)
{
	unsigned char* shared_image, * new_image_convolution;

	unsigned num_threads = 256;
	unsigned num_blocks = 1;

	double time_spent = 0.0;
	clock_t begin = clock();

	cudaMallocManaged((void**)& shared_image, (unsigned long long)w * (unsigned long long)h * 4 * sizeof(unsigned char));
	cudaMallocManaged((void**)& new_image_convolution, ((unsigned long long)w - 2) * ((unsigned long long)h - 2) * 4 * sizeof(unsigned char));

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

	if (num_threads > 1024)
	{
		num_blocks = 2;
		num_threads = 1024;
	}

	convolution << < num_blocks, num_threads >> > (shared_image, new_image_convolution, (w - 2) * (h - 2) * 4, w - 2, num_threads);

	cudaDeviceSynchronize();

	lodepng_encode32_file(output_filename, new_image_convolution, w - 2, h - 2);

	cudaFree(shared_image);
	cudaFree(new_image_convolution);
	free(image);

	clock_t end = clock();

	time_spent += (double)(end - begin) / CLOCKS_PER_SEC;

	printf("Execution time: %f\n", time_spent);
}

__global__ void pool(unsigned char* image, unsigned char* new_image, unsigned size, unsigned w, unsigned num_threads)
{
}

int main()
{
	char* input_filename = "mango.png";
	char* output_filename = "new_image.png";

	unsigned error;

	unsigned char* image, * shared_image, * new_image_rectify, * new_image_pooling;
	unsigned w, h;

	unsigned num_threads = 256;
	unsigned num_blocks = 1;

	double time_spent = 0.0;
	clock_t begin = clock();


	error = lodepng_decode32_file(&image, &w, &h, input_filename);
	if (error) printf("error %u: %s\n", error, lodepng_error_text(error));

	cudaMallocManaged((void**)& shared_image, (unsigned long long)w * (unsigned long long)h * 4 * sizeof(unsigned char));
	cudaMallocManaged((void**)& new_image_rectify, (unsigned long long)w * (unsigned long long)h * 4 * sizeof(unsigned char));
	cudaMallocManaged((void**)& new_image_pooling, (unsigned long long)w / 2 * (unsigned long long)h / 2 * 4 * sizeof(unsigned char));

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

	if (num_threads > 1024)
	{
		num_blocks = 2;
		num_threads = 1024;
	}

	//rectify << <num_blocks, num_threads >> > (shared_image, new_image_rectify, w * h * 4, num_threads);
	pool << < num_blocks, num_threads >> > (shared_image, new_image_pooling, w / 2 * h / 2 * 4, w, num_threads);

	cudaDeviceSynchronize();

	//lodepng_encode32_file(output_filename, new_image_rectify, w, h);
	lodepng_encode32_file(output_filename, new_image_pooling, w / 2, h / 2);

	cudaFree(shared_image);
	cudaFree(new_image_rectify);
	cudaFree(new_image_pooling);
	free(image);

	clock_t end = clock();

	time_spent += (double)(end - begin) / CLOCKS_PER_SEC;

	printf("Execution time: %f\n", time_spent);

	return 0;
}