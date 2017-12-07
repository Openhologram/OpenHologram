/**
* @mainpage Openholo Generation Point Cloud : GPGPU Accelation using CUDA
* @brief
*/

#ifndef OPH_GEN_POINTCLOUD_GPU_LIB_H
#define OPH_GEN_POINTCLOUD_GPU_LIB_H

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <memory.h>

#include <iostream>
#include <vector>
#include <string>

/* CUDA Library Include */
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <device_functions.h>
#include <device_launch_parameters.h>

#define __CUDA_INTERNAL_COMPILATION__ //for CUDA Math Module
#include <math_constants.h>
#include <math_functions.h> //Single Precision Floating
//#include <math_functions_dbl_ptx3.h> //Double Precision Floating
#include <vector_functions.h> //Vector Processing Function
#undef __CUDA_INTERNAL_COMPILATION__


#define THREAD_X 32
#define THREAD_Y 16


namespace openholo {

	/**
	* @brief 3D Point Cloud Data
	*/
	typedef struct OphPointCloud {
		/// constructor
		OphPointCloud(const float x, const float y, const float z) {
			this->x = x; this->y = y; this->z = z;
		}

		OphPointCloud(const float x, const float y, const float z, const float amp, const float phase) {
			this->x = x; this->y = y; this->z = z;
			this->amplitude = amp; this->phase = phase;
		}

		float x;
		float y;
		float z;
		float amplitude;
		float phase;
	} OphPC;

	/**
	* @brief Config Information Specification : All 17 Elements
	*/
	typedef struct OphConfig {
		float pointCloudScaleX;	/// Scaling factor of x coordinate of point cloud
		float pointCloudScaleY;	/// Scaling factor of y coordinate of point cloud
		float pointCloudScaleZ;	/// Scaling factor of z coordinate of point cloud

		float offsetDepth;		/// Offset value of point cloud in z direction

		float samplingPitchX;	/// Pixel pitch of SLM in x direction
		float samplingPitchY;	/// Pixel pitch of SLM in y direction

		int nx;	/// Number of pixel of SLM in x direction
		int ny;	/// Number of pixel of SLM in y direction

		char *filterShapeFlag;	/// Shape of spatial bandpass filter ("Circle" or "Rect" for now)
		float filterXwidth;		/// Width of spatial bandpass filter in x direction (For "Circle," only this is used)
		float filterYwidth;		/// Width of spatial bandpass filter in y direction

		float focalLengthLensIn;		/// Focal length of input lens of Telecentric
		float focalLengthLensOut;		/// Focal length of output lens of Telecentric
		float focalLengthLenEyepiece;	/// Focal length of eyepiece lens				

		float lambda;		/// Wavelength of laser

		float tiltAngleX;	/// Tilt angle in x direction for spatial filtering
		float tiltAngleY;	/// Tilt angle in y direction for spatial filtering
	} OphSpec;


	/**
	\defgroup PointCloud_Load
	* @{
	* @brief Import Point Cloud Data Base File : *.dat file.
	* This Function is included memory location of Input Point Clouds.
	*/
	/**
	* @param inputData PointCloud(*.dat) input file path
	* @param model	PointCloud Output Data Array
	* @return number of Pointcloud (if it failed loading, it returned -1)
	*/
	int ophLoadPointCloud(char *inputData, OphPointCloud *&model);

	/**
	* @overload
	* @param inputData PointCloud(*.dat) input file path
	* @param model PointCloud Output Data Array
	* @return number of Pointcloud (if it failed loading, it returned -1)
	*/
	int ophLoadPointCloud(const std::string inputData, std::vector<OphPointCloud> &model);

	/**
	* @overload
	* @param inputData PointCloud(*.dat) input file path
	* @param vertexArray PointCloud coordinate Array
	* @param phaseArray  PointCloud Phase Array
	* @param amplitudeArray PointCloud Amplitude Array
	* @return number of Pointcloud (if it failed loading, it returned -1)
	*/
	int ophLoadPointCloud(const std::string inputData, std::vector<float> &vertexArray, std::vector<float> &phaseArray, std::vector<float> &amplitudeArray);
	/** @} */


	/**
	\defgroup Import_Configfile
	* @{
	* @brief Import Specification Config File(*.config) file
	*/
	/**
	* @param InputData Specification Config(*.config) file path
	* @param Config output config OphSepc struct
	*/
	bool ophLoadSpecConfig(char *inputData, OphSpec *config);

	/**
	* @overload
	* @param InputData Specification Config(*.config) file path
	* @param config output config OphSepc struct
	*/
	bool ophLoadSpecConfig(const std::string inputData, OphSpec &config);
	/** @} */


	/**
	\defgroup Set_Data
	* @{
	* @brief Directly Set Config Specification Information without *.config file
	*/
	/**
	* @param config Output config OphSepc struct
	* @param scaleX Scaling factor of x coordinate of point cloud
	* @param scaleY Scaling factor of y coordinate of point cloud
	* @param scaleZ Scaling factor of z coordinate of point cloud
	*/
	void ophSetScaleFactor(OphSpec &config, const float scaleX, const float scaleY, const float scaleZ);

	/**
	* @param config Output config OphSepc struct
	* @param offsetDepth Offset value of point cloud in z direction
	*/
	void ophSetOffsetDepth(OphSpec &config, const float offsetDepth);

	/**
	* @param config Output config OphSepc struct
	* @param pitchX Pixel pitch of SLM in x direction
	* @param pitchY Pixel pitch of SLM in y direction
	*/
	void ophSetSamplingPitch(OphSpec &config, const float pitchX, const float pitchY);

	/**
	* @param config Output config OphSepc struct
	* @param n_x Number of pixel of SLM in x direction
	* @param n_y Number of pixel of SLM in y direction
	*/
	void ophSetImageSize(OphSpec &config, const int n_x, const int n_y);

	/**
	* @param config Output config OphSepc struct
	* @param lambda Wavelength of laser
	*/
	void ophSetWaveLength(OphSpec &config, const float lambda);

	/**
	* @param config Output config OphSepc struct
	* @param tiltAngleX Tilt angle in x direction for spatial filtering
	* @param tiltAngleY Tilt angle in y direction for spatial filtering
	*/
	void ophSetTiltAngle(OphSpec &config, const float tiltAngleX, const float tiltAngleY);
	/** @}	*/


	/**
	\defgroup PointCloud_Generation
	* @{
	* @brief Calculate Integral Fringe Pattern of 3D Point Cloud based Computer Generated Holography
	*/
	/**
	* @param model Input 3D PointCloud Model Data
	* @param n_points Number of model
	* @param config Input config OphSepc struct
	* @param dst Output Fringe Pattern
	* @return implement time (sec)
	*/
	double ophGenCghPointCloud(const OphPointCloud *model, const int n_points, const OphSpec &config, float *dst);

	/**
	* @overload
	* @param model Input 3D PointCloud Model Data
	* @param config Input config OphSepc struct
	* @param dst Output Fringe Pattern
	* @return implement time (sec)
	*/
	double ophGenCghPointCloud(const std::vector<OphPointCloud> &model, const OphSpec &config, float *dst);

	/**
	* @overload
	* @param vertexArray Input 3D PointCloud Model Coordinate Array Data
	* @param amplitudeArray Input 3D PointCloud Model Amplitude Array Data
	* @param config Input config OphSepc struct
	* @param dst Output Fringe Pattern
	* @return implement time (sec)
	*/
	double ophGenCghPointCloud(const std::vector<float> &vertexArray, const std::vector<float> &amplitudeArray, const OphSpec &config, float *dst);
	/** @}	*/


	namespace gpu {

		/**
		\defgroup PointCloud_Generation
		* @{
		* @brief GPGPU Accelation of ophGenCghPointCloud() using nVidia CUDA
		*/
		/**
		* @param vertexArray Input 3D PointCloud Model Coordinate Array Data
		* @param amplitudeArray Input 3D PointCloud Model Amplitude Array Data
		* @param config Input config OphSepc struct
		* @param dst Output Fringe Pattern
		* @return implement time (sec)
		*/
		double ophGenCghPointCloud_cuda(const std::vector<float> &vertexArray, const std::vector<float> &amplitudeArray, const OphSpec &config, float *dst);
		/** @}	*/
	}
}

#endif