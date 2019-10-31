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
		unsigned x = 0, y = 0;
		float rgb[3] = { 0, 0, 0 };

		// we want the x, y to be the coordinates in the theoretical matrix
		x = index / 4 % w + 1;
		y = index / (w * 4) + 1;

		float w3[3][3] =
		{
		  1,	2,		-1,
		  2,	0.25,	-2,
		  1,	-2,		-1
		};

		// we want the ixy to be the starting index in the original image that we use to compute convolution
		for (int i = 0; i < type; i++)
		{
			for (int j = 0; j < type; j++)
			{
				for (int a = 0; a < 3; a++)
				{
					unsigned tmp_index = 4 * (w + 2) * (y + j - 1) + 4 * (x + i - 1) + a;

					rgb[a] += image[tmp_index] * w3[j][i]; if (index == 3952124 && a == 2)
					{
						printf("i = %d, image[i] = %d, w = %f, rbg[a] = %f\n", tmp_index, image[tmp_index], w3[j][i], rgb[a]);
					}
				}
			}
		}

		if (index == 0) {
			printf("index = %d, x = %d, y = %d, r = %f, g = %f, b = %f\n", index, x, y, rgb[0], rgb[1], rgb[2]);
		}

		for (int a = 0; a < 3; a++)
		{
			if (index == 3952124 && a == 2)
			{
				printf("b = %f\n", rgb[2]);
			}
			if (rgb[a] < 0)
				rgb[a] = 0;
			else if (rgb[a] > 255)
				rgb[a] = 255;
			new_image[index + a] = rgb[a];
		}
		new_image[index + 3] = 255;
	}
}


void do_convolution(unsigned char* image, unsigned w, unsigned h, unsigned num_blocks, unsigned num_threads)
{
	char* output_filename = "test_convolution.png";

	unsigned char* shared_image, * new_image_convolution;

	unsigned new_w = w - 2, new_h = h - 2;

	cudaMallocManaged((void**)& shared_image, (unsigned long long)w * (unsigned long long)h * 4 * sizeof(unsigned char));
	cudaMallocManaged((void**)& new_image_convolution, ((unsigned long long)new_w) * ((unsigned long long)new_h) * 4 * sizeof(unsigned char));

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

	printf("new_w = %d, new_h = %d\n", new_w, new_h);
	printf("%d | %d | %d\n", shared_image[0 + 2], shared_image[4 + 2], shared_image[8 + 2]);
	printf("%d | %d | %d\n", shared_image[3976 + 2], shared_image[3980 + 2], shared_image[3984 + 2]);
	printf("%d | %d | %d\n", shared_image[7952 + 2], shared_image[7956 + 2], shared_image[7960 + 2]);

	convolution << < num_blocks, num_threads >> > (shared_image, new_image_convolution, 3, new_w * new_h * 4, new_w, num_threads);

	cudaDeviceSynchronize();

	lodepng_encode32_file(output_filename, new_image_convolution, new_w, new_h);

	cudaFree(shared_image);
	cudaFree(new_image_convolution);
}

int main()
{
	char* input_filename = "test.png";

	unsigned error;

	unsigned char* image, * shared_image;
	unsigned w, h;

	unsigned num_threads = 21;
	unsigned num_blocks = 1;

	double time_spent = 0.0;
	clock_t begin = clock();

	error = lodepng_decode32_file(&image, &w, &h, input_filename);
	if (error) printf("error %u: %s\n", error, lodepng_error_text(error));

	if (num_threads > 1024)
	{
		num_blocks = 1 + num_threads / 1024;
		num_threads = num_threads / num_blocks + 1;
	}

	printf("w = %d, h = %d\n", w, h);

	do_convolution(image, w, h, num_blocks, num_threads);

	free(image);

	clock_t end = clock();

	time_spent += (double)(end - begin) / CLOCKS_PER_SEC;

	printf("Execution time: %f\n", time_spent);

	return 0;
}