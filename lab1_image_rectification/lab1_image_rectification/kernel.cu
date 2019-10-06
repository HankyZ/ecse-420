#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "lodepng.h"
#include "lodepng_process.h"
#include <stdio.h>

__global__ void process(unsigned char* image, unsigned char* new_image, unsigned size, unsigned num_threads)
{
	for (int i = blockIdx.x * num_threads + threadIdx.x; i < size; i += num_threads)
	{
		for (int a = 0; a < 4; a++)
		{
			i += a;
			int value = image[i] - 127;
			if (value < 0)
				value = 0;

			new_image[i] = value + 127;
		}
	}
}

void printArray(int* a, int n)
{
	for (int i = 0; i < n; i++)
	{
		printf("c[%d] = %d\n", i, a[i]);
	}
}

int compareArray(unsigned char* a, unsigned char* b, int size)
{
	for (int i = 0; i < size; i++)
	{
		if (a[i] != b[i])
		{
			printf("a[%d] is %d, b[%d] is %d\n", i, a[i], i, b[i]);
			return -1;
		}
	}
	printf("the two arrays are the same");
	return 0;
}

int main()
{
	char* input_filename = "test.png";
	char* output_filename = "new_image.png";

	unsigned error;

	unsigned char* image, * shared_image, * new_image;
	unsigned w, h;

	unsigned num_threads = 2048;
	unsigned num_blocks = 1;

	error = lodepng_decode32_file(&image, &w, &h, input_filename);
	if (error) printf("error %u: %s\n", error, lodepng_error_text(error));

	cudaMallocManaged((void**)& shared_image, (unsigned long long)w * (unsigned long long)h * 4 * sizeof(unsigned char));
	cudaMallocManaged((void**)& new_image, (unsigned long long)w * (unsigned long long)h * 4 * sizeof(unsigned char));

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

	printf("w = %d, h = %d\n", w, h);

	unsigned char* expected_new_image;
	expected_new_image = (unsigned char*)malloc((unsigned long long)w * (unsigned long long)h * 4 * sizeof(unsigned char));

	for (int i = 0; i < 4 * w * h; i++)
	{
		int value = image[i] - 127;

		if (value < 0)
			value = 0;

		expected_new_image[i] = value + 127;
	}

	if (num_threads > 1024)
	{
		num_blocks = 2;
		num_threads = 1024;
	}

	process << <num_blocks, num_threads >> > (shared_image, new_image, w * h * 4, num_threads);

	cudaDeviceSynchronize();

	lodepng_encode32_file(output_filename, new_image, w, h);

	cudaFree(shared_image);
	cudaFree(new_image);
	free(image);

	return 0;
}