#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "lodepng.h"
#include "lodepng_process.h"
#include <stdio.h>

__global__ void process(unsigned char* image, unsigned char* new_image, unsigned* w, unsigned* h)
{
	int i = *w * 4 * blockIdx.x + 4 * threadIdx.x;

	for (int a = 0; a < 4; a++)
	{
		unsigned value = image[i + a] - 0;
		//printf("value is %d\n", value);
		if (value < 0)
			new_image[i + a] = 0;
		else
			new_image[i + a] = value + 0;
	}
}

void printArray(int* a, int n)
{
	for (int i = 0; i < n; i++)
	{
		printf("c[%d] = %d\n", i, a[i]);
	}
}

int compareArray(unsigned char *a, unsigned char* b, int size)
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
	unsigned* d_w, * d_h;

	unsigned num_threads = 5;


	error = lodepng_decode32_file(&image, &w, &h, input_filename);
	if (error) printf("error %u: %s\n", error, lodepng_error_text(error));

	cudaMalloc((void**)& d_w, sizeof(unsigned));
	cudaMalloc((void**)& d_h, sizeof(unsigned));

	cudaMemcpy(d_w, &w, sizeof(unsigned), cudaMemcpyHostToDevice);
	cudaMemcpy(d_h, &h, sizeof(unsigned), cudaMemcpyHostToDevice);

	cudaMallocManaged((void**)& shared_image, (unsigned long long)w * (unsigned long long)h * 4 * sizeof(unsigned char));
	cudaMallocManaged((void**)& new_image, (unsigned long long)w * (unsigned long long)h * 4 * sizeof(unsigned char));

	int counter = 0;
	for (int i = 0; i < w; i++)
	{
		for (int j = 0; j < h; j++)
		{
			for (int a = 0; a < 4; a++)
			{
				int p = 4 * w * i + 4 * j + a;
				printf("p = %d\n", p);
				if (p == 2005984)
				{
					printf("image[%d] = %d\n", p, image[p]);
				}
				shared_image[p] = image[p];
				counter++;
			}
		}
	}

	printf("counter is %d\n", counter);
	printf("h * w * 4 = %d\n", h * w * 4);

	if (compareArray(image, shared_image, w * h * 4) == 0)
		printf("same\n");
	else
		printf("not same\n");

	printf("w is %d, h is %d\n", w, h);

	process << <w, h >> > (shared_image, new_image, d_w, d_h);

	cudaDeviceSynchronize();

	lodepng_encode32_file(output_filename, new_image, w, h);

	if (compareArray(image, new_image, w * h * 4) == 0)
		printf("same\n");
	else
		printf("not same\n");

	cudaFree(d_w);
	cudaFree(d_h);
	cudaFree(shared_image);
	cudaFree(new_image);
	free(image);


	return 0;
}