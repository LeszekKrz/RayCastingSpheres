#pragma once

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
	circles _circles;
	lights _lights;
} scene;


void CreateCircles(circles* h_circles);
void PrepareCircles(circles h_circles, circles* d_circles);
void DisplayCircles(circles h_circles);

void CreateLights(lights* h_lights);
void PrepareLights(lights h_lights, lights* d_lights);
void DisplayLights(lights h_lights);

void PrepareTexture(unsigned char** d_texture, int size);
void CopyTexture(unsigned char** h_texture, unsigned char** d_texture, int size, bool toDevice);