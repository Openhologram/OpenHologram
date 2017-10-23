/** @mainpage
@tableofcontents
@section intro Introduction
This library implements the hologram generation method using depth map data. <br>
It is implemented on the CPU and the GPU to improve the performance of the hologram generation method.
Thus, user can compare the performance between the CPU and GPU implementation. 
<br>
@image html doc_intro.png
@image latex doc_intro.png

@section algo Algorithm Reference
The original algorithm is modified in the way that can be easily implemented in parallel. <br>
Back propagate each depth plane to the hologram plane and accumulate the results of each propagation.
<br>
@image html doc_algo.png "Depth Map Hologram Generation Algorithm"
@image latex doc_algo.png "Depth Map Hologram Generation Algorithm"

@section swcom Software Components
The library consists a main hologram generation module(Hologram folder) and its sample program(HologramDepthmap folder).
<br>The following shows the list of files with the directory structure.
<br>
@image html doc_swfolders.png
@image latex doc_swfolders.png

@section proc Main Procedure
The main function of the library is a  \c \b GenerateHologram() of \c HologramGenerator class.
The following is the procedure of it and functions called form it..
<br><br>
@image html doc_proc.png "GenerateHologram Function Procedure"
@image latex doc_proc.png "GenerateHologram Function Procedure"

@section env Environment
 - Microsoft Visual Studio 2015 C++
 - Qt 5.6.2
 - CUDA 8.0
 - FFTW 3.3.5

@section build How to Build 
Before building an execution file, you need to install MS Visual Studio 2015 C++ and Qt, also CUDA for the GPU execution. 
 1. Download the source code from <a href="https://github.com/Openhologram/OpenHologram/tree/master/OpenHolo_DepthMap">here</a>.
 2. Go to the directory 'HologramDepthmap'.
 3. Open the Visual Studio soulution file, 'HologramDepthmap.sln'. 
 4. Check the configuation of the Qt & CUDA to work with the Visual Studio.
 5. To use FFTW, copy 'libfftw3-3.dll' into the 'bin' directory and copy 'libfftw3-3.lib' into the 'lib' directory.
 6. Visual Studio Build Menu -> Configuration Menu, set "Release" for the Active solution configuration, "x64" for the Active solution platform.
 7. Set 'HologramDepthmap' as a StartUp Project.
 8. Build a Solution.
 9. After building, you can find the execution file, 'HologramDepthmap.exe' under the 'bin' directory.
 10. Execute 'HologramDepthmap.exe', then you can see the following GUI of the sample program. <br><br>
  @image html doc_exe.png "the Sample Program & its Execution"
  @image latex doc_exe.png "the Sample Program & its Execution"
 */

 /**
 * \defgroup init_module Initialize
 * \defgroup load_module Loading Data
 * \defgroup depth_module Computing Depth Value
 * \defgroup trans_module Transform 
 * \defgroup gen_module Generation Hologram
 * \defgroup encode_modulel Encoding
 * \defgroup write_module Writing Image
 * \defgroup recon_module Reconstruction
 */


#ifndef __Hologram_Generator_h
#define __Hologram_Generator_h

#include <graphics/vec.h>
#include <graphics/complex.h>
#include <QtCore/QDir>
#include <QtCore/QFile>
#include <QtGui/QImage>
#include <QtWidgets/qmessagebox.h>
#include <vector>
#include <cufft.h>

using namespace graphics;

/**
* @brief Structure variable for hologram paramemters
* @details This structure has all parameters for generating a hologram, which is read from a config file.
*/
struct HologramParams{
	
	double				field_lens;					///< FIELD_LENS at config file  
	double				lambda;						///< WAVELENGTH  at config file
	double				k;							///< 2 * PI / lambda
	ivec2				pn;							///< SLM_PIXEL_NUMBER_X & SLM_PIXEL_NUMBER_Y
	vec2				pp;							///< SLM_PIXEL_PITCH_X & SLM_PIXEL_PITCH_Y
	vec2				ss;							///< pn * pp

	double				near_depthmap;				///< NEAR_OF_DEPTH_MAP at config file
	double				far_depthmap;				///< FAR_OF_DEPTH_MAP at config file
	
	uint				num_of_depth;				///< the number of depth level.
													/**< <pre>
													   if FLAG_CHANGE_DEPTH_QUANTIZATION == 0  
													      num_of_depth = DEFAULT_DEPTH_QUANTIZATION 
													   else  
												          num_of_depth = NUMBER_OF_DEPTH_QUANTIZATION  </pre> */

	std::vector<int>	render_depth;				///< Used when only few specific depth levels are rendered, usually for test purpose
};

/** 
* @brief Main class for generating a hologram using depth map data.
* @details This is a main class for generating a digital hologram using depth map data. It is implemented on the CPU and GPU.
*  1. Read Config file. - to set all parameters needed for generating a hologram.
*  2. Initialize all variables. - memory allocation on the CPU and GPU.
*  3. Generate a digital hologram using depth map data.
*  4. For the testing purpose, reconstruct a image from the generated hologram.
*/
class HologramGenerator {

public:

	HologramGenerator();
	~HologramGenerator();
	
	void setMode(bool isCPU);

	/** \ingroup init_module */
	bool readConfig();

	/** \ingroup init_module */
	void initialize();
	
	/** \ingroup gen_module */
	void GenerateHologram();

	/** \ingroup recon_module */
	void ReconstructImage();

	//void writeMatFileComplex(const char* fileName, Complex* val);							
	//void writeMatFileDouble(const char* fileName, double * val);
	//bool readMatFileDouble(const char* fileName, double * val);

private:

	/** \ingroup init_module
	* @{ */
	void init_CPU();   
	void init_GPU();
	/** @} */

	/** \ingroup load_module
	* @{ */
	bool ReadImageDepth(int ftr);
	bool prepare_inputdata_CPU(uchar* img, uchar* dimg);
	bool prepare_inputdata_GPU(uchar* img, uchar* dimg);
	/** @} */

	/** \ingroup depth_module
	* @{ */
	void GetDepthValues();
	void change_depth_quan_CPU();
	void change_depth_quan_GPU();
	/** @} */

	/** \ingroup trans_module
	* @{ */
	void TransformViewingWindow();
	/** @} */

	/** \ingroup gen_module 
	* @{ */
	void Calc_Holo_by_Depth(int frame);
	void Calc_Holo_CPU(int frame);
	void Calc_Holo_GPU(int frame);
	void Propagation_AngularSpectrum_CPU(char domain, Complex* input_u, double propagation_dist);
	void Propagation_AngularSpectrum_GPU(char domain, cufftDoubleComplex* input_u, double propagation_dist);
	/** @} */

	/** \ingroup encode_modulel
	* @{ */
	void Encoding_Symmetrization(ivec2 sig_location);
	void encoding_CPU(int cropx1, int cropx2, int cropy1, int cropy2, ivec2 sig_location);
	void encoding_GPU(int cropx1, int cropx2, int cropy1, int cropy2, ivec2 sig_location);
	/** @} */

	/** \ingroup write_module
	* @{ */
	void Write_Result_image(int ftr);
	/** @} */

	void get_rand_phase_value(Complex& rand_phase_val);
	void get_shift_phase_value(Complex& shift_phase_val, int idx, ivec2 sig_location);

	void fftwShift(Complex* in, Complex* out, int nx, int ny, int type, bool bNomalized = false);
	void exponent_complex(Complex* val);
	void fftShift(int nx, int ny, Complex* input, Complex* output);

	//void writeIntensity_gray8_bmp(const char* fileName, int nx, int ny, double* intensity);
	//void writeIntensity_gray8_bmp(const char* fileName, int nx, int ny, Complex* complexvalue);
	//void writeIntensity_gray8_real_bmp(const char* fileName, int nx, int ny, Complex* complexvalue);
	//void writeImage_fromGPU(QString imgname, int pnx, int pny, cufftDoubleComplex* gpu_data);

	/** \ingroup recon_module
	* @{ */
	void Reconstruction(Complex* hh_e);
	void Test_Propagation_to_Eye_Pupil(Complex* hh_e);
	void Write_Simulation_image(int num, double val);
	void circshift(Complex* in, Complex* out, int shift_x, int shift_y, int nx, int ny);
	/** @} */

private:

	bool					isCPU_;						///< if true, it is implemeted on the CPU, otherwise on the GPU.

	unsigned char*			img_src_gpu_;				///< GPU variable - image source data
	unsigned char*			dimg_src_gpu_;				///< GPU variable - depth map data
	double*					depth_index_gpu_;			///< GPU variable - quantized depth map data
	
	double*					img_src_;					///< CPU variable - image source data
	double*					dmap_src_;					///< CPU variable - depth map data
	double*					depth_index_;				///< CPU variable - quantized depth map data
	int*					alpha_map_;					///< CPU variable - calculated alpha map data
	double*					dmap_;						///< CPU variable - physical distances of depth map
	
	double					dstep_;						///< the physical increment of each depth map layer.
	std::vector<double>		dlevel_;					///< the physical value of all depth map layer.
	std::vector<double>		dlevel_transform_;			///< transfomed dlevel_ variable
	
	Complex*				U_complex_;					///< CPU variable - the generated hologram before encoding.
	double*					u255_fringe_;				///< the final hologram, used for writing the result image.

	HologramParams			params_;					///< structure variable for hologram parameters

	std::string				SOURCE_FOLDER;				///< input source folder - config file.
	std::string				IMAGE_PREFIX;				///< the prefix of the input image file - config file.
	std::string				DEPTH_PREFIX;				///< the prefix of the deptmap file - config file	
	std::string				RESULT_FOLDER;				///< the name of the result folder - config file
	std::string				RESULT_PREFIX;				///< the prefix of the result file - config file
	bool					FLAG_STATIC_IMAGE;			///< if true, the input image is static.
	uint					START_OF_FRAME_NUMBERING;	///< the start frame number.
	uint					NUMBER_OF_FRAME;			///< the total number of the frame.	
	uint					NUMBER_OF_DIGIT_OF_FRAME_NUMBERING; ///< the number of digit of frame number.

	int						Transform_Method_;			///< transform method 
	int						Propagation_Method_;		///< propagation method - currently AngularSpectrum
	int						Encoding_Method_;			///< encoding method - currently Symmetrization

	double					WAVELENGTH;					///< wave length

	bool					FLAG_CHANGE_DEPTH_QUANTIZATION;	///< if true, change the depth quantization from the default value.
	uint					DEFAULT_DEPTH_QUANTIZATION;		///< default value of the depth quantization - 256
	uint					NUMBER_OF_DEPTH_QUANTIZATION;   ///< depth level of input depthmap.
	bool					RANDOM_PHASE;					///< If true, random phase is imposed on each depth layer.

	// for Simulation (reconstruction)
	//===================================================
	std::string				Simulation_Result_File_Prefix_;	///< reconstruction variable for testing
	int						test_pixel_number_scale_;		///< reconstruction variable for testing
	vec2					Pixel_pitch_xy_;				///< reconstruction variable for testing
	ivec2					SLM_pixel_number_xy_;			///< reconstruction variable for testing
	double					f_field_;						///< reconstruction variable for testing
	double					eye_length_;					///< reconstruction variable for testing
	double					eye_pupil_diameter_;			///< reconstruction variable for testing
	vec2					eye_center_xy_;					///< reconstruction variable for testing
	double					focus_distance_;				///< reconstruction variable for testing
	int						sim_type_;						///< reconstruction variable for testing
	double					sim_from_;						///< reconstruction variable for testing
	double					sim_to_;						///< reconstruction variable for testing
	int						sim_step_num_;					///< reconstruction variable for testing
	double*					sim_final_;						///< reconstruction variable for testing

};


#endif