#include "cudaFunctions.cuh"

#include <stdio.h>
#include <iostream>
#include "functions.hpp"



__global__ void rayKernel(scene d_scene, float3 start, int width, int height, int fovW, int fovH, unsigned char* texture)
{
	int x, y = blockIdx.x;
	x = blockIdx.y * 1024 + threadIdx.x;
	//printf("%d %d\n", y, x);
	if (y * width + x >= width * height) return;
	unsigned char* pixel = texture + y * width * 3 + x * 3;
	ray light, out;
	light.origin = make_float3(0, 0, -200);
	light.direction = normalize(start + make_float3(((float)x / width) * fovW, ((float)y / height) * fovH, 0) - light.origin);
	float3 aim = start + make_float3(((float)x / width) * fovW, ((float)y / height) * fovH, 0);
	if ((x == 0 && y == 0) || (x == width - 1 && y == height - 1))
	{
		printf("%f %f %f\n", aim.x, aim.y, aim.z);
	}
	if (findHit(light, d_scene._circles, &out))
	{
		*pixel = 255;
		*(pixel + 1) = 0;
		*(pixel + 2) = 0;
	}
	else
	{
		*pixel = 0;
		*(pixel + 1) = 0;
		*(pixel + 2) = 255;
	}

}

void rayTrace(scene d_scene, int width, int height, unsigned char* texture)
{
	int fovW = 80;
	int fovH = 60;
	float3 start = make_float3(-fovW/ 2, -fovH / 2, -100);
	if (width > 1024)
	{
		dim3 dim(height, width / 1024 + 1);
		rayKernel << < dim, 1024 >> > (d_scene, start, width, height, fovW, fovH, texture);
	}
	else
	{
		rayKernel << < height, width >> > (d_scene, start, width, height, fovW, fovH, texture);
	}
	cudaError_t cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
	}
}

__device__ __host__ bool findHit(ray light, circles d_circles, ray* out)
{
	float closest = 0;
	bool hitSomething = false;
	float3 centre;
	//Vector hit, point;
	float3 hit, point;
	float t;
	float d;
	float a;
	for (int i = 0; i < d_circles.n; i++)
	{
		centre = make_float3(*(d_circles.xs + i), *(d_circles.ys + i), *(d_circles.zs + i));
		t = dot(light.direction, (centre - light.origin));
		point = light.origin + light.direction * t;
		d = length(point - centre);
		if (d > *(d_circles.rs + i)) continue;

		else return true;
	}
	return hitSomething;
}