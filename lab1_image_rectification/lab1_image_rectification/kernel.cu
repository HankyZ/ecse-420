#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "lodepng.h"
#include "lodepng_process.h"
#include <stdio.h>

__global__ void process(unsigned char* image, unsigned char* new_image, unsigned* w)
{
	int i = *w * 4 * blockIdx.x + 4 * threadIdx.x;

	for (int a = 0; a < 4; a++)
	{
		int p = i + a;
		int value = image[p] - 127;
		if (value < 0)
			new_image[p] = 0;
		else
			new_image[p] = value + 127;
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
	return 0;
}

int main()
{
	char* input_filename = "test1.png";
	char* output_filename = "new_image.png";

	unsigned error;

	unsigned char* image, * shared_image, * new_image;
	unsigned w, h;
	unsigned* d_w;

	error = lodepng_decode32_file(&image, &w, &h, input_filename);
	if (error) printf("error %u: %s\n", error, lodepng_error_text(error));

	cudaMalloc((void**)& d_w, sizeof(unsigned));

	cudaMemcpy(d_w, &w, sizeof(unsigned), cudaMemcpyHostToDevice);

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

	unsigned char* expected_new_image;
	expected_new_image = (unsigned char*)malloc((unsigned long long)w * (unsigned long long)h * 4 * sizeof(unsigned char));

	for (int i = 0; i < h; i++)
	{
		for (int j = 0; j < w; j++)
		{
			for (int a = 0; a < 4; a++)
			{
				int p = 4 * w * i + 4 * j + a;
				int value = image[p] - 127;

				if (value < 0)
					expected_new_image[p] = 0;
				else
					expected_new_image[p] = value + 127;
			}
		}
	}

	process << <h, w >> > (shared_image, new_image, d_w);

	cudaDeviceSynchronize();

	lodepng_encode32_file(output_filename, new_image, w, h);

	cudaFree(d_w);
	cudaFree(shared_image);
	cudaFree(new_image);
	free(image);

	return 0;
}