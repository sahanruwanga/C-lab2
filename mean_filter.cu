#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <iostream>
#include <cuda.h>

__global__ void mean_filter_gpu(unsigned char* input_image, int img_height, int img_width,unsigned char* filtered_image, int window_size){
	
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	int row = blockIdx.y * blockDim.y + threadIdx.y;

	// Check window size to avoid errors at edges
    int window_checker = (window_size-1)/2;
    if( row < img_height && col < img_width){
    		
    		int top_margin, bottom_margin, left_margin, right_margin;

            // Check top margin
            if((row-window_checker) < 0)
                top_margin = 0;
            else
                top_margin = row - window_checker;

            // Check bottom margin
            if((row+window_checker) <= img_height - 1)
                bottom_margin = row + window_checker;
            else
                bottom_margin = img_height - 1;

            // Check left margin
    		if((col-window_checker) >= 0)
    			left_margin = col - window_checker;
    		else
				left_margin = 0;

			// Check right margin
			if((col+window_checker) <= img_width - 1)
				right_margin = col + window_checker;
			else
				right_margin = img_width - 1;

			double val;
			for (int m = top_margin; m <= bottom_margin; m++){
				for(int n = left_margin; n <= right_margin; n++){
					val += input_image[(m*img_height)+n];
				}
			}	
			int cal_window_size = (bottom_margin - top_margin + 1) * (right_margin - left_margin + 1);
			filtered_image[row*img_height+col] = val/cal_window_size;       // Find mean and replace
		}
}

void mean_filter_cpu(unsigned char* input_image, int img_height,int img_width,unsigned char* filtered_image, int window_size){

    // Check window size to avoid errors at edges
    int window_checker = (window_size - 1)/2;

    // For all raws
    for (int row = 0; row < img_height; row++){
        int top_margin, bottom_margin;

        // Check top margin
        if((row - window_checker) < 0)
	        top_margin = 0;
		else
            top_margin = row - window_checker;

        // Check bottom margin
		if((row + window_checker) <= img_height - 1)
			bottom_margin = row + window_checker;
		else
            bottom_margin = img_height - 1;

        // For all columns
    	for(int col = 0; col < img_width; col++){
    		int left_margin, right_margin;

            // Check left margin
    		if((col - window_checker) >= 0)
    			left_margin = col - window_checker;
    		else
				left_margin = 0;

            // Check right margin
			if((col + window_checker) <= img_width - 1)
				right_margin = col+window_checker;
			else
				right_margin = img_width - 1;

			double val;
			for (int m = top_margin; m <= bottom_margin ; m++){
				for(int n=left_margin;n<=right_margin;n++){
					val+=input_image[(m*img_height)+n];
				}
			}
			int cal_window_size = (bottom_margin-top_margin+1) * (right_margin-left_margin+1);
			filtered_image[row*img_height+col] = val/cal_window_size;   // Find mean and replace
		}
	}
}

int main(int argc, char *argv[]) {
	
	int img_width, img_height, img_size;
	img_width = 1280;
	img_height = 1280;
	img_size = img_width * img_height;
	int window_size = 5;    // Assign variables as required for image resolution, and window size (640x640, 1280x1280, 3, 5)
	
	// Read the image
	FILE* f = fopen("image_1280x1280.bmp", "rb");       // Provide an image out of two (image_640x640.bmp, image_1280x1280.bmp)

	// Initializing parameters (host)
    unsigned char* input_img_cpu = new unsigned char[img_size];
	unsigned char* filtered_output_img_cpu = new unsigned char[img_size];

	// Initializing parameters (device)
    unsigned char* mean_device_image = new unsigned char[img_size];
    unsigned char* input_img_gpu;
    unsigned char* filtered_output_img_gpu;

    // Read image in host
    fread(input_img_cpu, sizeof(unsigned char), img_size, f);
    fclose(f);      // Close image reading
	
	// Assign block and grid sizes for maximum utilization
    int block_size = 32;
    int grid_size = img_width/block_size;
    dim3 dimBlock(block_size, block_size, 1);
    dim3 dimGrid(grid_size, grid_size, 1);
	
	int current_iteration;
	int total_iterations = 30;      // Run code 30 times and get average time
    
    double total_cpu_time = 0;
    double total_gpu_time = 0;
    
    for(current_iteration=0;current_iteration<total_iterations;current_iteration++){
		// Allocate memory in the device for raw input image and filtered image
    	cudaMalloc((void **)&input_img_gpu,img_size*sizeof(unsigned char));
		cudaMalloc((void **)&filtered_output_img_gpu,img_size*sizeof(unsigned char));

		// Copy raw input image GPU
		cudaMemcpy(input_img_gpu, input_img_cpu,img_size*sizeof(unsigned char),cudaMemcpyHostToDevice);

        // Execute in GPU
		clock_t start_d = clock();
        mean_filter_gpu <<< dimGrid, dimBlock >>> (input_img_gpu, img_height,img_width, filtered_output_img_gpu, window_size);
        cudaThreadSynchronize();
		clock_t end_d = clock();
        
        // Execute in CPU
        clock_t start_h = clock();
        mean_filter_cpu(input_img_cpu, img_height, img_width, filtered_output_img_cpu, window_size);
        clock_t end_h = clock();
        
        cudaMemcpy(mean_device_image, filtered_output_img_gpu, img_size*sizeof(unsigned char), cudaMemcpyDeviceToHost);

        total_gpu_time += (double)(end_d - start_d)/CLOCKS_PER_SEC;
        total_cpu_time += (double)(end_h - start_h)/CLOCKS_PER_SEC;

		// Free device memory
        cudaFree(input_img_gpu);
        cudaFree(filtered_output_img_gpu);

	}
    
    printf("Average GPU Time: %f\n",(total_gpu_time/total_iterations));
    printf("Average CPU Time: %f\n",(total_cpu_time/total_iterations));
    
	return 0;
}
