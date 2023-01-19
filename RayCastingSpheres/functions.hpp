#pragma once
#include "Functions.hpp"
#include <cstdlib>
#include <time.h>
#include <iostream>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "helper_math.h"

typedef struct
{
	float* xs;
	float* ys;
	float* zs;
	float* rs;
	int n;
} circles;

typedef struct
{
	float* xs;
	float* ys;
	float* zs;
	int n;
} lights;

typedef struct
{
	float3 pos;
	float width;
	float height;
	float fovV;
	float fovH;
	float3 lowerLeft;
	float3 horizontalStep;
	float3 verticalStep;
} camera;

typedef struct
{
	circles _circles;
	lights _lights;
	camera _camera;
} scene;



void CreateCircles(circles* h_circles);
void PrepareCircles(circles h_circles, circles* d_circles);
void DisplayCircles(circles h_circles);

void CreateLights(lights* h_lights);
void PrepareLights(lights h_lights, lights* d_lights);
void DisplayLights(lights h_lights);

void PrepareTexture(unsigned char** d_texture, int size);
void CopyTexture(unsigned char** h_texture, unsigned char** d_texture, int size, bool toDevice);

void PrepareCamera(camera* h_camera);