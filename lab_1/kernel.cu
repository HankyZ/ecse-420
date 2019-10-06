#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "lodepng.h"
#include <stdio.h>

__global__ void rectify(unsigned char* image, unsigned char* new_image, unsigned size, unsigned num_threads)
{
	for (int i = (blockIdx.x * num_threads + threadIdx.x) * 4; i < size; i += num_threads * 4)
	{
		for (int a = 0; a < 4; a++)
		{
			int value = image[i + a];
			if (value < 127)
				value = 127;
			new_image[i + a] = value;
		}
	}
}

__global__ void pool(unsigned char* image, unsigned char* new_image, unsigned size, unsigned w, unsigned num_threads)
{
	for (int i = (blockIdx.x * num_threads + threadIdx.x) * 4; i < size; i += num_threads * 4)
	{
		unsigned x, y, i0, i1, i2, i3, r_max, g_max, b_max, a_max;

		x = i % (w * 2) * 2;
		y = i / (w * 2);

		i0 = 8 * w * y + x;
		i1 = i0 + 4;
		i2 = i0 + 4 * w;
		i3 = i2 + 4;

		unsigned r[] = { image[i0], image[i1], image[i2], image[i3] };
		unsigned g[] = { image[i0 + 1], image[i1 + 1], image[i2 + 1], image[i3 + 1] };
		unsigned b[] = { image[i0 + 2], image[i1 + 2], image[i2 + 2], image[i3 + 2] };
		unsigned a[] = { image[i0 + 3], image[i1 + 3], image[i2 + 3], image[i3 + 3] };

		r_max = r[0];
		g_max = g[0];
		b_max = b[0];
		a_max = a[0];

		for (int c = 1; c < 4; c++)
		{
			if (r[c] > r_max)
			{
				r_max = r[c];
			}
			if (g[c] > g_max)
			{
				g_max = g[c];
			}
			if (b[c] > b_max)
			{
				b_max = b[c];
			}
			if (a[c] > a_max)
			{
				a_max = a[c];
			}
		}

		new_image[i] = r_max;
		new_image[i + 1] = g_max;
		new_image[i + 2] = b_max;
		new_image[i + 3] = a_max;

		/*if (y <= 1)
			printf("i = %d, x = %d, y = %d, i0 = %d, i1 = %d, i2 = %d, i3 = %d. stored pixel value at %d, %d, %d, %d\n", i, x, y, i0, i1, i2, i3, i, i + 1, i + 2, i + 3);*/
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

	unsigned char* image, * shared_image, * new_image_rectify, * new_image_pooling;
	unsigned w, h;

	unsigned num_threads = 2048;
	unsigned num_blocks = 1;

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

	rectify << <num_blocks, num_threads >> > (shared_image, new_image_rectify, w * h * 4, num_threads);
	//pool << < num_blocks, num_threads >> > (shared_image, new_image_pooling, w / 2 * h / 2 * 4, w, num_threads);

	cudaDeviceSynchronize();

	printf("w = %d, h = %d\n", w, h);

	lodepng_encode32_file(output_filename, new_image_rectify, w, h);
	//lodepng_encode32_file(output_filename, new_image_pooling, w / 2, h / 2);

	cudaFree(shared_image);
	cudaFree(new_image_rectify);
	cudaFree(new_image_pooling);
	free(image);

	return 0;
}