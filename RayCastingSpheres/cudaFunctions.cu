#include "cudaFunctions.cuh"

#include <stdio.h>
#include <iostream>
#include "functions.hpp"



__global__ void rayKernel(scene d_scene, unsigned char* texture)
{
	extern __shared__ float shared_circles[];
	int n = d_scene._circles.n;
	float* xs = shared_circles;
	float* ys = shared_circles + n;
	float* zs = ys + n;
	float* rs = zs + n;
	int j = 0, m_n = n;
	while (m_n > 0)
	{
		if (threadIdx.x < m_n)
		{
			xs[threadIdx.x + j * 1024] = d_scene._circles.xs[threadIdx.x + j * 1024];
			ys[threadIdx.x + j * 1024] = d_scene._circles.ys[threadIdx.x + j * 1024];
			zs[threadIdx.x + j * 1024] = d_scene._circles.zs[threadIdx.x + j * 1024];
			rs[threadIdx.x + j * 1024] = d_scene._circles.rs[threadIdx.x + j * 1024];
		}
		j++;
		m_n -= 1024;
	}
	__syncthreads();
	int x, y = blockIdx.x;
	x = blockIdx.y * 1024 + threadIdx.x;
	if (y * d_scene._camera.width + x >= d_scene._camera.width * d_scene._camera.height) return;
	unsigned char* pixel = texture + (y * (int)d_scene._camera.width * 3 + x * 3);
	ray light, out;
	light.origin = d_scene._camera.pos;
	light.direction = normalize(d_scene._camera.lowerLeft + x * d_scene._camera.horizontalStep + y * d_scene._camera.verticalStep - light.origin);
	//if (findHit(light, d_scene._circles, &out))
	if (sharedFindHit(light, xs, ys, zs, rs, n, &out))
	{
		unsigned int color = calculateColor(out, d_scene._lights, d_scene._camera.pos, 120 << 16);
		*pixel = (color >> 16) & 255;
		*(pixel + 1) = (color >> 8) & 255;
		*(pixel + 2) = color & 255;
	}
	else
	{
		*pixel = 0;
		*(pixel + 1) = 0;
		*(pixel + 2) = 0;
	}

}

void rayTrace(scene d_scene, unsigned char* texture)
{
	if (d_scene._camera.width > 1024)
	{
		dim3 dim(d_scene._camera.height, d_scene._camera.width / 1024 + 1);
		rayKernel << < dim, 1024, 4 * d_scene._circles.n * sizeof(float) >> > (d_scene, texture);
	}
	else
	{
		rayKernel << < d_scene._camera.height, d_scene._camera.width, 4 * d_scene._circles.n * sizeof(float) >> > (d_scene, texture);
	}
	cudaError_t cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
		printf(cudaGetErrorString(cudaGetLastError()));
	}
}

void rayTraceCPU(scene h_scene, unsigned char* texture)
{
	unsigned char* pixel;
	ray light, out;
	unsigned int color;
	for (int y = 0; y < 600; y++)
	{
		for (int x = 0; x < 800; x++)
		{
			pixel = texture + (y * (int)h_scene._camera.width * 3 + x * 3);
			light.origin = h_scene._camera.pos;
			light.direction = normalize(h_scene._camera.lowerLeft + x * h_scene._camera.horizontalStep + y * h_scene._camera.verticalStep - light.origin);
			if (findHit(light, h_scene._circles, &out))
			{
				color = calculateColor(out, h_scene._lights, h_scene._camera.pos, 120 << 16);
				*pixel = (color >> 16) & 255;
				*(pixel + 1) = (color >> 8) & 255;
				*(pixel + 2) = color & 255;
			}
			else
			{
				*pixel = 0;
				*(pixel + 1) = 0;
				*(pixel + 2) = 0;
			}
		}
	}
}

__device__ __host__ bool findHit(ray light, circles d_circles, ray* out)
{
	float closest = 0;
	bool hitSomething = false;
	float3 centre;
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

__device__ __host__ bool sharedFindHit(ray light, float* xs, float* ys, float* zs, float* rs, int n, ray* out)
{
	float closest = 0;
	bool hitSomething = false;
	float3 centre;
	float3 hit, point;
	float t;
	float d;
	float a;
	for (int i = 0; i < n; i++)
	{
		centre = make_float3(*(xs + i), *(ys + i), *(zs + i));
		t = dot(light.direction, (centre - light.origin));
		point = light.origin + light.direction * t;
		d = length(point - centre);
		if (d > *(rs + i)) continue;
		else
		{
			a = sqrt(*(rs + i) * *(rs + i) - d * d);
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

__device__ __host__ unsigned int calculateColor(ray point, lights d_lights, float3 pos, int color)
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
	float3 V = normalize(pos - point.origin);
	float3 R;
	int m = 10;
	for (int i = 0; i < d_lights.n; i++)
	{
		L = normalize(make_float3(*(d_lights.xs + i), *(d_lights.ys + i), *(d_lights.zs + i)) - point.origin);
		R = 2 * dot(N, L) * N - L;
		rf += kd * 1 * ri * clamp(dot(N, L), 0.0f, 1.0f) + ks * 1 * ri * pow(clamp(dot(V, R), 0.0f, 1.0f), 10);
		gf += kd * 1 * gi * clamp(dot(N, L), 0.0f, 1.0f) + ks * 1 * gi * pow(clamp(dot(V, R), 0.0f, 1.0f), 10);
		bf += kd * 1 * bi * clamp(dot(N, L), 0.0f, 1.0f) + ks * 1 * bi * pow(clamp(dot(V, R), 0.0f, 1.0f), 10);
	}

	r = (unsigned char)clamp(rf * 255, 0.0f, 255.0f);
	g = (unsigned char)clamp(gf * 255, 0.0f, 255.0f);
	b = (unsigned char)clamp(bf * 255, 0.0f, 255.0f);

	return (r << 16) | (g << 8) | b;
}