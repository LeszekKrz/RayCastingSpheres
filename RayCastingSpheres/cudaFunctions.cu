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
		//if (out.origin.x > 0) printf("%f %f %f %f %f %f\n", out.origin.x, out.origin.y, out.origin.z, out.direction.x, out.direction.y, out.direction.z);
		unsigned int color = calculateColor(out, d_scene._lights, 120 << 16);
		//printf("%d %d %d\n", (color >> 16)&255, (color >> 8) & 255, color & 255);
		*pixel = (color >> 16) & 255;
		*(pixel + 1) = (color >> 8) & 255;
		*(pixel + 2) = color & 255;

		/**pixel = 0;
		*(pixel + 1) = 255;
		*(pixel + 2) = 0;*/
	}
	else
	{
		*pixel = 0;
		*(pixel + 1) = 0;
		*(pixel + 2) = 0;
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
		else
		{
			a = sqrt(*(d_circles.rs + i) * *(d_circles.rs + i) - d * d);
			hit = light.origin + (t - a) * light.direction;
			if (!hitSomething || length(hit - light.origin) < closest)
			{
				hitSomething = true;
				closest = length(hit - light.origin);
				out->origin = hit;
				out->direction = hit - centre;
			}
		}
	}
	return hitSomething;
}

__device__ __host__ unsigned int calculateColor(ray point, lights d_lights, int color)
{
	unsigned char r, g, b = color & 255;
	g = (color >> 8) & 255;
	r = (color >> 16) & 255;
	float ri = (float)r / 255, gi = (float)g / 255, bi = (float)b / 255;
	unsigned char ro, go, bo;
	float rf = 0.1, gf = 0.1, bf = 0.1;
	float kd = 0.5f, ks = 0.5f;
	float3 L;
	float3 N = normalize(point.direction);
	float3 V = normalize(make_float3(0, 0, -200) - point.origin);
	float3 R;
	int m = 10;
	for (int i = 0; i < d_lights.n; i++)
	{
		L = normalize(make_float3(*(d_lights.xs + i), *(d_lights.ys + i), *(d_lights.zs + i)) - point.origin);
		//if (point.origin.x < 0)
		//{
		//	//printf("%f %f %f %f\n", point.origin.x, point.origin.y, point.origin.z, dot(N, L));
		//	return 0;
		//}
		//if (dot(N, L) < 0) continue;
		R = 2 * dot(N, L) * N - L;
		rf += kd * 1 * ri * clamp(dot(N, L), 0.0f, 1.0f) + ks * 1 * ri * pow(clamp(dot(V, R), 0.0f, 1.0f), 10);
		gf += kd * 1 * gi * clamp(dot(N, L), 0.0f, 1.0f) + ks * 1 * gi * pow(clamp(dot(V, R), 0.0f, 1.0f), 10);
		bf += kd * 1 * bi * clamp(dot(N, L), 0.0f, 1.0f) + ks * 1 * bi * pow(clamp(dot(V, R), 0.0f, 1.0f), 10);
	}

	r = (unsigned char)clamp(rf * 255, 0.0f, 255.0f);
	g = (unsigned char)clamp(gf * 255, 0.0f, 255.0f);
	b = (unsigned char)clamp(bf * 255, 0.0f, 255.0f);

	//return (255 << 16) | (255 << 8) | 255;
	//return 0;
	return (r << 16) | (g << 8) | b;
}