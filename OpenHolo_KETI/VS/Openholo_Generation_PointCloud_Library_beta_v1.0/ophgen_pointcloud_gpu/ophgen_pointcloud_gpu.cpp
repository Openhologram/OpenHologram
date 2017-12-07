#include "ophgen_pointcloud_gpu.h"

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


typedef struct OphKernelConst {
	int n_points;	///number of point cloud

	float scaleX;		/// Scaling factor of x coordinate of point cloud
	float scaleY;		/// Scaling factor of y coordinate of point cloud
	float scaleZ;		/// Scaling factor of z coordinate of point cloud

	float offsetDepth;	/// Offset value of point cloud in z direction

	int Nx;		/// Number of pixel of SLM in x direction
	int Ny;		/// Number of pixel of SLM in y direction

	float sin_thetaX; ///sin(tiltAngleX)
	float sin_thetaY; ///sin(tiltAngleY)
	float k;		  ///Wave Number = (2 * PI) / lambda;

	float pixel_x; /// Pixel pitch of SLM in x direction
	float pixel_y; /// Pixel pitch of SLM in y direction
	float halfLength_x; /// (pixel_x * nx) / 2
	float halfLength_y; /// (pixel_y * ny) / 2
} OphGpuConst;


__global__ void ophKernelCghPointCloud_cuda(float3 *pointCloud, float *amplitude, const OphGpuConst *config, float *dst) {
	int idxX = blockIdx.x * blockDim.x + threadIdx.x;
	int idxY = blockIdx.y * blockDim.y + threadIdx.y;

	if ((idxX < config->Nx) && (idxY < config->Ny)) {
		for (int j = 0; j < config->n_points; ++j) {
			//Convert to CUDA API Vector Data Type
			float3 scalePoint3D;
			scalePoint3D.x = pointCloud[j].x * config->scaleX;
			scalePoint3D.y = pointCloud[j].y * config->scaleY;
			scalePoint3D.z = pointCloud[j].z * config->scaleZ + config->offsetDepth;

			float3 planePoint = make_float3(0.f, 0.f, 0.f);
			planePoint.x = ((float)idxX + 0.5f) * config->pixel_x - config->halfLength_x;
			planePoint.y = config->halfLength_y - ((float)idxY + 0.5f) * config->pixel_y;

			float r = sqrtf((planePoint.x - scalePoint3D.x)*(planePoint.x - scalePoint3D.x) + (planePoint.y - scalePoint3D.y)*(planePoint.y - scalePoint3D.y) + scalePoint3D.z*scalePoint3D.z);
			float referenceWave = config->k*config->sin_thetaX*planePoint.x + config->k*config->sin_thetaY*planePoint.y;
			float result = amplitude[j] * cosf(config->k*r - referenceWave);

			*(dst + idxX + idxY * config->Nx) += result; //R-S Integral
		}
	}
	__syncthreads();
}


double openholo::gpu::ophGenCghPointCloud_cuda(const std::vector<float> &vertexArray, const std::vector<float> &amplitudeArray, const OphSpec &config, float *dst) {
	int _bx = config.nx / THREAD_X;
	int _by = config.ny / THREAD_Y;
	int block_x = 2;
	int block_y = 2;

	//blocks number
	while (1) {
		if ((block_x >= _bx) && (block_y >= _by)) break;
		if (block_x < _bx) block_x *= 2;
		if (block_y < _by) block_y *= 2;
	}

	//threads number
	const unsigned long long bufferSize = config.nx * config.ny * sizeof(float);

	//Host Memory Location
	float3 *hostPointCloud = (float3*)vertexArray.data();
	float *hostAmplitude = (float*)amplitudeArray.data();

	//Initializa Config for CUDA Kernel
	OphGpuConst hostConfig; {
		hostConfig.n_points = vertexArray.size() / 3;
		hostConfig.scaleX = config.pointCloudScaleX;
		hostConfig.scaleY = config.pointCloudScaleY;
		hostConfig.scaleZ = config.pointCloudScaleZ;
		hostConfig.offsetDepth = config.offsetDepth;

		// Output Image Size
		hostConfig.Nx = config.nx;
		hostConfig.Ny = config.ny;

		// Tilt Angle
		float thetaX = config.tiltAngleX * (CUDART_PI_F / 180.f); // Convert degree to radian Angle
		float thetaY = config.tiltAngleY * (CUDART_PI_F / 180.f); // Convert degree to radian Angle
		hostConfig.sin_thetaX = sinf(thetaX);
		hostConfig.sin_thetaY = sinf(thetaY);

		// Wave Number
		hostConfig.k = (2.f * CUDART_PI_F) / config.lambda;

		// Pixel pitch at eyepiece lens plane (by simple magnification) ==> SLM pitch
		hostConfig.pixel_x = config.samplingPitchX;
		hostConfig.pixel_y = config.samplingPitchY;

		// Length (Width) of complex field at eyepiece plane (by simple magnification)
		float Length_x = hostConfig.pixel_x * hostConfig.Nx;
		float Length_y = hostConfig.pixel_y * hostConfig.Ny;
		hostConfig.halfLength_x = Length_x / 2.f;
		hostConfig.halfLength_y = Length_y / 2.f;
	}

	//Device(GPU) Memory Location
	float3 *devicePointCloud;
	cudaMalloc((void**)&devicePointCloud, vertexArray.size() * sizeof(float));
	cudaMemcpy(devicePointCloud, hostPointCloud, vertexArray.size() * sizeof(float), cudaMemcpyHostToDevice);

	float *deviceAmplitude;
	cudaMalloc((void**)&deviceAmplitude, amplitudeArray.size() * sizeof(float));
	cudaMemcpy(deviceAmplitude, hostAmplitude, amplitudeArray.size() * sizeof(float), cudaMemcpyHostToDevice);

	OphGpuConst *deviceConfig;
	cudaMalloc((void**)&deviceConfig, sizeof(OphGpuConst));
	cudaMemcpy(deviceConfig, &hostConfig, sizeof(hostConfig), cudaMemcpyHostToDevice);

	float *deviceDst;
	cudaMalloc((void**)&deviceDst, bufferSize);

	clock_t time_start, time_finish;
	time_start = clock();
	{
		dim3 Dg(block_x, block_y, 1);  //grid : designed 2D blocks
		dim3 Db(THREAD_X, THREAD_Y, 1);  //block : designed 2D threads
		ophKernelCghPointCloud_cuda << < Dg, Db >> > (devicePointCloud, deviceAmplitude, deviceConfig, deviceDst);
		cudaMemcpy(dst, deviceDst, bufferSize, cudaMemcpyDeviceToHost);
	}
	time_finish = clock();

	//Device(GPU) Memory Delete
	cudaFree(devicePointCloud);
	cudaFree(deviceAmplitude);
	cudaFree(deviceDst);
	cudaFree(deviceConfig);

	return (double)(time_finish - time_start) / CLOCKS_PER_SEC;
}