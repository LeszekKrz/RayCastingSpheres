#include "cudaFunctions.cuh"
#include "Vector.cuh"

#include <stdio.h>
#include <iostream>
#include "functions.hpp"


__global__ void rayKernel(scene d_scene, float3 start, int width, int height, unsigned char* texture)
{
	/*ray m_ray;
	ray hit;
	int x, y = blockIdx.x;
	if (width > 1024)
	{
		x = blockIdx.y * 1024 + threadIdx.x;
		if (y * width + x > width * height) return;
	}
	else
	{
		x = threadIdx.x;
	}
	unsigned char* pixel = texture + y * width + x;
	m_ray.origin = Vector(0, 0, 0);
	m_ray.direction = Vector(start) + Vector(x, y, 0) - Vector(0, 0, 0);
	m_ray.direction = m_ray.direction / m_ray.direction.length();

	if (findHit(m_ray, d_scene._circles, &hit))
	{
		*pixel = 0;
		*(pixel + 1) = 255;
		*(pixel + 2) = 0;
	}
	else
	{
		*pixel = 0;
		*(pixel + 1) = 0;
		*(pixel + 2) = 0;
	}*/
}

void rayTrace(scene d_scene, int width, int height, unsigned char* texture)
{
	float3 start = make_float3(-width / 2, -height / 2, 10);
	if (width > 1024)
	{
		dim3 dim(height, width / 1024 + 1);
		rayKernel << < dim, 1024 >> > (d_scene, start, width, height, texture);
	}
	else
	{
		rayKernel << < height, width >> > (d_scene, start, width, height, texture);
	}
}

__device__ __host__ bool findHit(ray light, circles d_circles, ray* out)
{
	float closest = 2000;
	Vector centre;
	//Vector hit, point;
	float t;
	float d;
	float a;
	/*for (int i = 0; i < d_circles.n; i++)
	{
		centre = Vector(*(d_circles.xs + i), *(d_circles.ys + i), *(d_circles.zs + i));
		t = light.direction.dot(centre - light.origin);
		point = light.origin + light.direction * t;
		d = (point - centre).length();
		if (d > *(d_circles.rs + i)) continue;
		else return true;
	}*/
	return false;
}