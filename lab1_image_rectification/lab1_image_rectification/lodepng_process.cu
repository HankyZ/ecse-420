/* Example of using lodepng to load, process, save image */
#include "lodepng.h"
#include <stdio.h>
#include <stdlib.h>
#include "lodepng_process.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"


void process(char* input_filename, char* output_filename)
{
	unsigned error;
	unsigned char* image, * new_image;
	unsigned width, height;

	error = lodepng_decode32_file(&image, &width, &height, input_filename);
	if (error) printf("error %u: %s\n", error, lodepng_error_text(error));
	new_image = (unsigned char*)malloc((unsigned char*)width * height * 4 * sizeof(unsigned char));

	// process image
	unsigned char value;
	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {


			new_image[4 * width * i + 4 * j + 0] = image[4 * width * i + 4 * j + 0]; // R
			new_image[4 * width * i + 4 * j + 1] = image[4 * width * i + 4 * j + 1]; // G
			new_image[4 * width * i + 4 * j + 2] = image[4 * width * i + 4 * j + 2]; // B

			/*
			value = image[4 * width * i + 4 * j];
			new_image[4*width*i + 4*j + 0] = value; // R
			new_image[4*width*i + 4*j + 1] = value; // G
			new_image[4*width*i + 4*j + 2] = value; // B
			*/
			new_image[4 * width * i + 4 * j + 3] = image[4 * width * i + 4 * j + 3]; // A
		}
	}

	lodepng_encode32_file(output_filename, new_image, width, height);

	free(image);
	free(new_image);
}