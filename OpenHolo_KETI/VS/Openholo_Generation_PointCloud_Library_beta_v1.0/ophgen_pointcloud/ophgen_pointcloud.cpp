#include "ophgen_pointcloud.h"

#define _USE_MATH_DEFINES
#include <math.h>
#include <time.h>

//Build Option : Multi Core Processing (OpenMP)
#ifdef _OPENMP
#include <omp.h>
#endif


int openholo::ophLoadPointCloud(char *inputData, OphPointCloud *&model) {
	FILE *file = fopen(inputData, "r");
	if (file == NULL) {
		fclose(file);
		return -1;
	}

	int n_points;
	fscanf(file, "%d\n", &n_points);

	// Memory Location
	model = (OphPointCloud*)malloc(sizeof(OphPointCloud)*n_points);

	// parse input point cloud file
	for (int i = 0; i < n_points; ++i) {
		int idx;
		float pX, pY, pZ, phase, amplitude;
		fscanf(file, "%d	%f %f %f %f %f\n", &idx, &pX, &pY, &pZ, &phase, &amplitude);

		if (idx == i) {
			(model + i)->x = pX;
			(model + i)->y = pY;
			(model + i)->z = pZ;
			(model + i)->amplitude = amplitude;
			(model + i)->phase = phase;
		}
		else {
			fclose(file);
			return -1;
		}
	}

	fclose(file);
	return n_points;
}

int openholo::ophLoadPointCloud(const std::string inputData, std::vector<OphPointCloud> &model) {
	FILE *file = fopen(inputData.c_str(), "r");
	if (file == NULL) {
		fclose(file);
		return -1;
	}

	int n_points;
	fscanf(file, "%d\n", &n_points);

	// parse input point cloud file
	for (int i = 0; i < n_points; ++i) {
		int idx;
		float pX, pY, pZ, phase, amplitude;
		fscanf(file, "%d	%f %f %f %f %f\n", &idx, &pX, &pY, &pZ, &phase, &amplitude);

		if (idx == i) {
			model.push_back(OphPointCloud(pX, pY, pZ, amplitude, phase));
		}
		else {
			fclose(file);
			return -1;
		}
	}

	fclose(file);
	return n_points;
}

int openholo::ophLoadPointCloud(const std::string inputData, std::vector<float> &vertexArray, std::vector<float> &phaseArray, std::vector<float> &amplitudeArray) {
	FILE *file = fopen(inputData.data(), "r");
	if (file == NULL) {
		fclose(file);
		return -1;
	}

	int n_points;
	fscanf(file, "%d\n", &n_points);

	// parse input point cloud file
	for (int i = 0; i < n_points; ++i) {
		int idx;
		float pX, pY, pZ, phase, amplitude;
		fscanf(file, "%d	%f %f %f %f %f\n", &idx, &pX, &pY, &pZ, &phase, &amplitude);

		if (idx == i) {
			vertexArray.push_back(pX);
			vertexArray.push_back(pY);
			vertexArray.push_back(pZ);
			phaseArray.push_back(phase);
			amplitudeArray.push_back(amplitude);
		}
		else {
			fclose(file);
			return -1;
		}
	}

	fclose(file);
	return n_points;
}


bool openholo::ophLoadSpecConfig(char *inputData, OphSpec *config) {
	FILE *file = fopen(inputData, "r");
	if (file == NULL) {
		std::cerr << "Failed Load Config Specification File!!" << std::endl;
		fclose(file);
		return false;
	}

	char *title[20];
	char *value[20];
	char line[128];

	int i = 0;
	while (fgets(line, 127, file)) {
		title[i] = (char*)malloc(sizeof(char) * 64);
		value[i] = (char*)malloc(sizeof(char) * 64);
		sscanf(line, "%s = %s", title[i], value[i]);
		++i;
	}

	if (i != 17) {
		std::cerr << "Failed Load Config Specification File!!" << std::endl;
		fclose(file);
		return false;
	}

	config->pointCloudScaleX = atof(value[0]);
	config->pointCloudScaleY = atof(value[1]);
	config->pointCloudScaleZ = atof(value[2]);
	config->offsetDepth = atof(value[3]);
	config->samplingPitchX = atof(value[4]);
	config->samplingPitchY = atof(value[5]);
	config->nx = atoi(value[6]);
	config->ny = atoi(value[7]);
	config->filterShapeFlag = value[8];
	config->filterXwidth = atof(value[9]);
	config->filterYwidth = atof(value[10]);
	config->focalLengthLensIn = atof(value[11]);
	config->focalLengthLensOut = atof(value[12]);
	config->focalLengthLenEyepiece = atof(value[13]);
	config->lambda = atof(value[14]);
	config->tiltAngleX = atof(value[15]);
	config->tiltAngleY = atof(value[16]);

	fclose(file);
	return true;
}

bool openholo::ophLoadSpecConfig(const std::string inputData, OphSpec &config) {
	return ophLoadSpecConfig((char*)inputData.c_str(), &config);
}


void openholo::ophSetScaleFactor(OphSpec &config, const float scaleX, const float scaleY, const float scaleZ) {
	config.pointCloudScaleX = scaleX;
	config.pointCloudScaleY = scaleY;
	config.pointCloudScaleZ = scaleZ;
}

void openholo::ophSetOffsetDepth(OphSpec &config, const float offsetDepth) {
	config.offsetDepth = offsetDepth;
}

void openholo::ophSetSamplingPitch(OphSpec &config, const float pitchX, const float pitchY) {
	config.samplingPitchX = pitchX;
	config.samplingPitchY = pitchY;
}

void openholo::ophSetImageSize(OphSpec &config, const int n_x, const int n_y) {
	config.nx = n_x;
	config.ny = n_y;
}

void openholo::ophSetWaveLength(OphSpec &config, const float lambda) {
	config.lambda = lambda;
}

void openholo::ophSetTiltAngle(OphSpec &config, const float tiltAngleX, const float tiltAngleY) {
	config.tiltAngleX = tiltAngleX;
	config.tiltAngleY = tiltAngleY;
}


double openholo::ophGenCghPointCloud(const OphPointCloud *model, const int n_points, const OphSpec &config, float *dst) {
	// Output Image Size
	int Nx = config.nx;
	int Ny = config.ny;

	// Tilt Angle
	float thetaX = config.tiltAngleX * (M_PI / 180.f); // Convert degree to radian Angle
	float thetaY = config.tiltAngleY * (M_PI / 180.f); // Convert degree to radian Angle

	// Wave Number
	float k = (2.f * M_PI) / config.lambda;

	// Pixel pitch at eyepiece lens plane (by simple magnification) ==> SLM pitch
	float pixel_x = config.samplingPitchX;
	float pixel_y = config.samplingPitchY;

	// Length (Width) of complex field at eyepiece plane (by simple magnification)
	float Length_x = pixel_x * Nx;
	float Length_y = pixel_y * Ny;

	clock_t time_start, time_finish;
	time_start = clock();
	{
		int n; // private variable for Multi Threading
#ifdef _OPENMP
		int num_threads = omp_get_num_threads(); // get number of Multi Threading
#pragma omp parallel for private(n)
#endif
		for (n = 0; n < n_points; ++n) {
			float x = (model + n)->x * (config.pointCloudScaleX);
			float y = (model + n)->y * (config.pointCloudScaleY);
			float z = (model + n)->z * (config.pointCloudScaleZ) + config.offsetDepth;
			float amplitude = (model + n)->amplitude;

			for (int row = 0; row < Ny; ++row) {
				// Y coordinate of the current pixel : Note that y index is reversed order
				float SLM_y = (Length_y / 2) - ((float)row + 0.5f) * pixel_y;

				for (int col = 0; col < Nx; ++col) {
					// X coordinate of the current pixel
					float SLM_x = ((float)col + 0.5f) * pixel_x - (Length_x / 2);

					float r = sqrtf((SLM_x - x)*(SLM_x - x) + (SLM_y - y)*(SLM_y - y) + z*z);
					float phi = k*r - k*SLM_x*sinf(thetaX) - k*SLM_y*sinf(thetaY); // Phase for printer
					float result = amplitude*cosf(phi);

					*(dst + col + row*Nx) += result; // R-S Integral
				}
			}
		}
	}
	time_finish = clock();

	return (double)(time_finish - time_start) / CLOCKS_PER_SEC;
}

double openholo::ophGenCghPointCloud(const std::vector<OphPointCloud> &model, const OphSpec &config, float *dst) {
	// Output Image Size
	int Nx = config.nx;
	int Ny = config.ny;

	// Tilt Angle
	float thetaX = config.tiltAngleX * (M_PI / 180.f); //Convert degree to radian Angle
	float thetaY = config.tiltAngleY * (M_PI / 180.f); //Convert degree to radian Angle

	// Wave Number
	float k = (2.f * M_PI) / config.lambda;

	// Pixel pitch at eyepiece lens plane (by simple magnification) ==> SLM pitch
	float pixel_x = config.samplingPitchX;
	float pixel_y = config.samplingPitchY;

	// Length (Width) of complex field at eyepiece plane (by simple magnification)
	float Length_x = pixel_x * Nx;
	float Length_y = pixel_y * Ny;

	int n_points = model.size();

	clock_t time_start, time_finish;
	time_start = clock();
	{
		int n; // private variable for Multi Threading
#ifdef _OPENMP
		int num_threads = omp_get_num_threads(); // get number of Multi Threading
#pragma omp parallel for private(n)
#endif
		for (n = 0; n < n_points; ++n) { //Create Fringe Pattern
			float x = model[n].x * config.pointCloudScaleX;
			float y = model[n].y * config.pointCloudScaleY;
			float z = model[n].z * config.pointCloudScaleZ + config.offsetDepth;
			float amplitude = model[n].amplitude;

			for (int row = 0; row < Ny; ++row) {
				// Y coordinate of the current pixel : Note that y index is reversed order
				float SLM_y = (Length_y / 2) - ((float)row + 0.5f) * pixel_y;

				for (int col = 0; col < Nx; ++col) {
					// X coordinate of the current pixel
					float SLM_x = ((float)col + 0.5f) * pixel_x - (Length_x / 2);

					float r = sqrtf((SLM_x - x)*(SLM_x - x) + (SLM_y - y)*(SLM_y - y) + z*z);
					float phi = k*r - k*SLM_x*sinf(thetaX) - k*SLM_y*sinf(thetaY); // Phase for printer
					float result = amplitude*cosf(phi);

					*(dst + col + row*Nx) += result; //R-S Integral
				}
			}
		}
	}
	time_finish = clock();

	return (double)(time_finish - time_start) / CLOCKS_PER_SEC;
}

double openholo::ophGenCghPointCloud(const std::vector<float> &vertexArray, const std::vector<float> &amplitudeArray, const OphSpec &config, float *dst) {
	// Output Image Size
	int Nx = config.nx;
	int Ny = config.ny;

	// Tilt Angle
	float thetaX = config.tiltAngleX * (M_PI / 180.f); //Convert degree to radian Angle
	float thetaY = config.tiltAngleY * (M_PI / 180.f); //Convert degree to radian Angle

	// Wave Number
	float k = (2.f * M_PI) / config.lambda;

	// Pixel pitch at eyepiece lens plane (by simple magnification) ==> SLM pitch
	float pixel_x = config.samplingPitchX;
	float pixel_y = config.samplingPitchY;

	// Length (Width) of complex field at eyepiece plane (by simple magnification)
	float Length_x = pixel_x * Nx;
	float Length_y = pixel_y * Ny;

	int n_points = vertexArray.size() / 3;

	clock_t time_start, time_finish;
	time_start = clock();
	{
		int n; // private variable for Multi Threading
#ifdef _OPENMP
		int num_threads = omp_get_num_threads(); // get number of Multi Threading
#pragma omp parallel for private(n)
#endif
		for (n = 0; n < n_points; ++n) { //Create Fringe Pattern
			float x = vertexArray[3 * n + 0] * config.pointCloudScaleX;
			float y = vertexArray[3 * n + 1] * config.pointCloudScaleY;
			float z = vertexArray[3 * n + 2] * config.pointCloudScaleZ + config.offsetDepth;
			float amplitude = amplitudeArray[n];

			for (int row = 0; row < Ny; ++row) {
				// Y coordinate of the current pixel : Note that y index is reversed order
				float SLM_y = (Length_y / 2) - ((float)row + 0.5f) * pixel_y;

				for (int col = 0; col < Nx; ++col) {
					// X coordinate of the current pixel
					float SLM_x = ((float)col + 0.5f) * pixel_x - (Length_x / 2);

					float r = sqrtf((SLM_x - x)*(SLM_x - x) + (SLM_y - y)*(SLM_y - y) + z*z);
					float phi = k*r - k*SLM_x*sinf(thetaX) - k*SLM_y*sinf(thetaY); // Phase for printer
					float result = amplitude*cosf(phi);

					*(dst + col + row*Nx) += result; //R-S Integral
				}
			}
		}
	}
	time_finish = clock();

	return (double)(time_finish - time_start) / CLOCKS_PER_SEC;
}