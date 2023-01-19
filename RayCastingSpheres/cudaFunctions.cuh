#pragma once

#include "functions.hpp"

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "helper_math.h"

typedef struct
{
	float3 origin;
	float3 direction;
} ray;


__global__ void rayKernel(scene d_scene, unsigned char* texture);

void rayTrace(scene d_scene, unsigned char* texture);
__device__ __host__ bool findHit(ray light, circles d_circles, ray* out);
__device__ __host__ unsigned int calculateColor(ray point, lights d_lights, float3 pov, int color);