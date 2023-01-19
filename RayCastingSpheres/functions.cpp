#include "Functions.hpp"
#include <cstdlib>
#include <time.h>
#include <iostream>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "helper_math.h"

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
		*h_circles->zs = 0;
		*h_circles->rs = 10;
		return;
	}

	for (int i = 0; i < n; i++)
	{
		*(h_circles->xs + i) = (float)(rand() % 4001 - 2000) / 10;
		*(h_circles->ys + i) = (float)(rand() % 4001 - 2000) / 10;
		*(h_circles->zs + i) = (float)(rand() % 4001 - 2000) / 10;
		*(h_circles->rs + i) = (float)(rand() % 101) / 10;
	}

	return;
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

	return;


Error:
	cudaFree(d_circles->xs);
	cudaFree(d_circles->ys);
	cudaFree(d_circles->zs);
	cudaFree(d_circles->rs);

	return;
}

void DisplayCircles(circles h_circles)
{
	for (int i = 0; i < h_circles.n; i++)
	{
		std::cout << *(h_circles.xs + i) << " " << *(h_circles.ys + i) << " " << *(h_circles.zs + i) << " " << *(h_circles.rs + i) << std::endl;
	}
	return;
}

void CreateLights(lights* h_lights)
{
	int n = h_lights->n;
	h_lights->xs = (float*)malloc(n * sizeof(float));
	h_lights->ys = (float*)malloc(n * sizeof(float));
	h_lights->zs = (float*)malloc(n * sizeof(float));
	if (n == 1)
	{
		*(h_lights->xs) = 100;
		*(h_lights->ys) = 0;
		*(h_lights->zs) = 0;
		return;

	}
	for (int i = 0; i < n; i++)
	{
		*(h_lights->xs + i) = (float)(rand() % 2001 - 1000) / 10;
		*(h_lights->ys + i) = (float)(rand() % 2001 - 1000) / 10;
		*(h_lights->zs + i) = (float)(rand() % 2001 - 1000) / 10;
	}

	return;
}

void PrepareLights(lights h_lights, lights* d_lights)
{
	int n = h_lights.n;
	d_lights->n = n;
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

	return;


Error:
	cudaFree(d_lights->xs);
	cudaFree(d_lights->ys);
	cudaFree(d_lights->zs);

	return;
}

void DisplayLights(lights h_lights)
{
	for (int i = 0; i < h_lights.n; i++)
	{
		std::cout << *(h_lights.xs + i) << " " << *(h_lights.ys + i) << " " << *(h_lights.zs + i) << std::endl;
	}
	return;
}

void PrepareTexture(unsigned char** d_texture, int size)
{
	cudaError_t cudaStatus;
	cudaStatus = cudaMalloc(d_texture, size);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	return;

Error:
	cudaFree(*d_texture);

	return;
}

void CopyTexture(unsigned char** h_texture, unsigned char** d_texture, int size, bool toDevice)
{
	cudaError_t cudaStatus;
	if (toDevice)
	{
		cudaStatus = cudaMemcpy(*d_texture, *h_texture, size * sizeof(char), cudaMemcpyHostToDevice);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMemcpy failed!");
			goto Error;
		}
	}
	else
	{
		cudaStatus = cudaMemcpy(*h_texture, *d_texture, size, cudaMemcpyDeviceToHost);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMemcpy failed!");
			goto Error;
		}

	}

	return;

Error:
	cudaFree(*d_texture);

	return;
}

void PrepareCamera(camera* h_camera)
{
	float3 up = make_float3(0, 1, 0);
	float3 at = normalize(make_float3(0, 0, 0) - h_camera->pos);
	float3 v = normalize(cross(up, at));
	float3 u = normalize(cross(at, v));
	float3 corner = h_camera->pos + at*100 - v * h_camera->fovH / 2 - u * h_camera->fovV / 2;
	h_camera->lowerLeft = corner;
	h_camera->horizontalStep = v * h_camera->fovH / h_camera->width;
	h_camera->verticalStep = u * h_camera->fovV / h_camera->height;
	return;
}
