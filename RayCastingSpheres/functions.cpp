#include "Functions.hpp"
#include <cstdlib>
#include <time.h>
#include <iostream>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

void CreateCircles(circles* h_circles)
{
	int n = h_circles->n;
	srand(time(NULL));
	h_circles->xs = (float*)malloc(n * sizeof(float));
	h_circles->ys = (float*)malloc(n * sizeof(float));
	h_circles->zs = (float*)malloc(n * sizeof(float));
	h_circles->rs = (float*)malloc(n * sizeof(float));

	if (n == 1)
	{
		*h_circles->xs = 0;
		*h_circles->ys = 0;
		*h_circles->zs = 40;
		*h_circles->rs = 20;
	}

	for (int i = 0; i < n; i++)
	{
		*(h_circles->xs + i) = (float)(rand() % 2001 - 1000) / 10;
		*(h_circles->ys + i) = (float)(rand() % 2001 - 1000) / 10;
		*(h_circles->zs + i) = (float)(rand() % 2001 - 1000) / 10;
		*(h_circles->rs + i) = (float)(rand() % 101) / 10;
	}
}

void PrepareCircles(circles h_circles, circles* d_circles)
{
	int n = h_circles.n;
	d_circles->n = n;
	cudaError_t cudaStatus;
	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&(d_circles->xs), n * sizeof(float));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}
	
	cudaStatus = cudaMalloc((void**)&(d_circles->ys), n * sizeof(float));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}
	
	cudaStatus = cudaMalloc((void**)&(d_circles->zs), n * sizeof(float));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}
	
	cudaStatus = cudaMalloc((void**)&(d_circles->rs), n * sizeof(float));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMemcpy(d_circles->xs, h_circles.xs, n * sizeof(float), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}
		
	cudaStatus = cudaMemcpy(d_circles->ys, h_circles.ys, n * sizeof(float), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}
		
	cudaStatus = cudaMemcpy(d_circles->zs, h_circles.zs, n * sizeof(float), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}
		
	cudaStatus = cudaMemcpy(d_circles->rs, h_circles.rs, n * sizeof(float), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}


Error:
	cudaFree(d_circles->xs);
	cudaFree(d_circles->ys);
	cudaFree(d_circles->zs);
	cudaFree(d_circles->rs);
}

void DisplayCircles(circles h_circles)
{
	for (int i = 0; i < h_circles.n; i++)
	{
		std::cout << *(h_circles.xs + i) << " " << *(h_circles.ys + i) << " " << *(h_circles.zs + i) << " " << *(h_circles.rs + i) << std::endl;
	}
}

void CreateLights(lights* h_lights)
{
	int n = h_lights->n;
	h_lights->xs = (float*)malloc(n * sizeof(float));
	h_lights->ys = (float*)malloc(n * sizeof(float));
	h_lights->zs = (float*)malloc(n * sizeof(float));
	for (int i = 0; i < n; i++)
	{
		*(h_lights->xs + i) = (float)(rand() % 2001 - 1000) / 10;
		*(h_lights->ys + i) = (float)(rand() % 2001 - 1000) / 10;
		*(h_lights->zs + i) = (float)(rand() % 2001 - 1000) / 10;
	}
}

void PrepareLights(lights h_lights, lights* d_lights)
{
	int n = h_lights.n;
	cudaError_t cudaStatus;
	
	cudaStatus = cudaMalloc((void**)&(d_lights->xs), n * sizeof(float));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&(d_lights->ys), n * sizeof(float));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&(d_lights->zs), n * sizeof(float));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMemcpy(d_lights->xs, h_lights.xs, n * sizeof(float), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	cudaStatus = cudaMemcpy(d_lights->ys, h_lights.ys, n * sizeof(float), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	cudaStatus = cudaMemcpy(d_lights->zs, h_lights.zs, n * sizeof(float), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}



Error:
	cudaFree(d_lights->xs);
	cudaFree(d_lights->ys);
	cudaFree(d_lights->zs);
}

void DisplayLights(lights h_lights)
{
	for (int i = 0; i < h_lights.n; i++)
	{
		std::cout << *(h_lights.xs + i) << " " << *(h_lights.ys + i) << " " << *(h_lights.zs + i) << std::endl;
	}
}
