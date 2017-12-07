/* OpenHolo Beta v1.0 - Generation Module - PointCloud
Point Cloud based Fringe Pattern Generation with CUDA GPGPU Example
*/

#include <iostream>
#include <vector>
using namespace std;

// Point Cloud Based CGH using CUDA GPGPU Library
#include "../ophgen_pointcloud_gpu/ophgen_pointcloud_gpu.h"

// Normalize and Save BMP File Library
#include "../oph_common/oph_common.h"


// Define I/O Data File
#define INPUT_MODEL	"TestPointCloud.dat" // 3D Point Cloud Data Base
#define INPUT_SPEC	"TestSpec.config" // Parameters Config Specification
#define OUTPUT_BMP	"Result_FringePattern" // Fringe Pattern Image Output Bitmap File Name


int main(int argc, char **argv)
{
	vector<float> vertexArray;
	vector<float> amplitudeArray;
	vector<float> phaseArray;
	openholo::OphSpec config;

	// Load Input Data File
	bool ok = openholo::ophLoadSpecConfig(INPUT_SPEC, config);
	if (!ok) {
		cerr << "Error : Failed to Load Specification Config File" << endl;
		exit(1);
	}
	int n_x = config.nx;
	int n_y = config.ny;
	int n_points = openholo::ophLoadPointCloud(INPUT_MODEL, vertexArray, phaseArray, amplitudeArray);
	if (n_points == -1) {
		cerr << "Error : Failed to Load Point Cloud File" << endl;
		exit(1);
	}
	else {
		cout << "Success to Load All " << n_points << " Points!" << endl;
	}

	// Memory Location for Result Image
	float *fringe = (float*)calloc(1, sizeof(float)*n_x*n_y);
	unsigned char *result = (unsigned char*)calloc(1, sizeof(unsigned char)*n_x*n_y);

	// Create CGH Fringe Pattern by 3D Point Cloud
	double time = openholo::gpu::ophGenCghPointCloud_cuda(vertexArray, amplitudeArray, config, fringe);
	cout << "Implement Time : " << time << " sec" << endl;

	// Normalization & Save *.bmp File
	openholo::ophNormalize(fringe, result, n_x, n_y);
	openholo::ophSaveFileBmp(result, n_x, n_y, OUTPUT_BMP);
	cout << "Success!!" << endl;

	// Delete Memory
	free(fringe);
	free(result);

	return 0;
}