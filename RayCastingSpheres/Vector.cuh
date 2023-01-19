#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cmath>

class Vector
{
public:
	float _x;
	float _y;
	float _z;

	__device__ __host__ Vector();
	__device__ __host__ Vector(float x, float y, float z);
	__device__ __host__ Vector(float3 source);
	__device__ __host__ Vector(const Vector& source);
	__device__ __host__ Vector operator + (Vector arg);
	__device__ __host__ Vector operator - (Vector arg);
	__device__ __host__ float length();
	__device__ __host__ Vector operator / (float arg);
	__device__ __host__ Vector operator * (float arg);
	__device__ __host__ float dot(Vector arg);
};