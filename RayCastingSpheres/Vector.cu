#include "Vector.cuh"

__device__ __host__ Vector::Vector()
{
	_x = 0;
	_y = 0;
	_z = 0;
}

__device__ __host__ Vector::Vector(float x, float y, float z)
{
	_x = x;
	_y = y;
	_z = z;
}

__device__ __host__ Vector::Vector(float3 source)
{
	_x = source.x;
	_y = source.y;
	_z = source.z;
}

__device__ __host__ Vector::Vector(const Vector& source)
{
	_x = source._x;
	_y = source._y;
	_z = source._z;
}

__device__ __host__ Vector Vector::operator+(Vector arg)
{
	return Vector(_x + arg._x, _y + arg._y, _z + arg._z);
}

__device__ __host__ Vector Vector::operator-(Vector arg)
{
	return Vector(_x - arg._x, _y - arg._y, _z - arg._z);
}

__device__ __host__ float Vector::length()
{
	return sqrt(_x * _x + _y * _y + _z * _z);
}

__device__ __host__ Vector Vector::operator/(float arg)
{
	return Vector(_x / arg, _y / arg, _z / arg);
}
__device__ __host__ Vector Vector::operator*(float arg)
{
	return Vector(_x * arg, _y * arg, _z * arg);
}

__device__ __host__ float Vector::dot(Vector arg)
{
	return _x * arg._x + _y * arg._y + _z * arg._z;
}