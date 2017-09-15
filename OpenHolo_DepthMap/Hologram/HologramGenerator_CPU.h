#ifndef __Hologram_Generator_h
#define __Hologram_Generator_h

#include <graphics/vec.h>
#include <graphics/complex.h>
#include <QtCore/QDir>
#include <QtCore/QFile>
#include <QtGui/QImage>
#include <QtWidgets/qmessagebox.h>
#include <vector>

using namespace graphics;

struct HologramParams{
	
	double				field_lens;
	double				lambda;
	double				k;
	ivec2				pn;
	vec2				pp;
	vec2				ss;

	double				near_depthmap;
	double				far_depthmap;
	
	uint				num_of_depth;

	std::vector<int>	render_depth;

	uint				pixel_pitch_depthmap_num;
	double*				pixel_pitch_depthmap_x;
	double*				pixel_pitch_depthmap_y;

};

class HologramGenerator {

public:

	HologramGenerator();
	~HologramGenerator();
	
	// 1. read config
	bool readConfig();

	// 2. initialize
	void initialize();
	
	// 3. Generate Hologram
	void GenerateHologram();

	void writeMatFileComplex(const char* fileName, Complex* val);							
	void writeMatFileDouble(const char* fileName, double * val);
	bool readMatFileDouble(const char* fileName, double * val);

	void ReconstructImage();
		
private:

	bool ReadImageDepth(int ftr);
	void GetDepthValues();
	void TransformViewingWindow();
	void Calc_Holo_by_Depth(int frame);
	void Propagation_AngularSpectrum(char domain, Complex* input_u, double propagation_dist);
	void Encoding_Symmetrization(ivec2 sig_location);
	void Write_Result_image(int ftr);

	void get_rand_phase_value(Complex& rand_phase_val);
	void get_shift_phase_value(Complex& shift_phase_val, int idx, ivec2 sig_location);

	void fftwShift(Complex* in, Complex* out, int nx, int ny, int type, bool bNomalized = false, char domain='S');
	void exponent_complex(Complex* val);
	void fftShift(int nx, int ny, Complex* input, Complex* output);
	void circshift(Complex* in, Complex* out, int shift_x, int shift_y, int nx, int ny);
	void writeIntensity_gray8_bmp(const char* fileName, int nx, int ny, double* intensity);
	void writeIntensity_gray8_bmp(const char* fileName, int nx, int ny, Complex* complexvalue);
	void writeIntensity_gray8_real_bmp(const char* fileName, int nx, int ny, Complex* complexvalue);

	void Test_Propagation_to_Eye_Pupil(Complex* hh_e);
	void Reconstruction(Complex* hh_e);
	void Write_Simulation_image(int num, double val);

private:

	double*					img_src_;
	int*					alpha_map_;
	double*					depth_index_;
	double*					dmap_src_;
	double*					dmap_;				// physical distances of depth map

	double					dstep_;
	std::vector<double>		dlevel_;
	std::vector<double>		dlevel_transform_;
	
	Complex*				U_complex_;			// frequency domain
	double*					u255_fringe_;
	double*					sim_final_;

	HologramParams			params_;

	std::string				SOURCE_FOLDER;
	std::string				IMAGE_PREFIX;
	std::string				DEPTH_PREFIX;
	std::string				RESULT_FOLDER;
	std::string				RESULT_PREFIX;
	bool					FLAG_STATIC_IMAGE;
	uint					START_OF_FRAME_NUMBERING;		
	uint					NUMBER_OF_FRAME;				
	uint					NUMBER_OF_DIGIT_OF_FRAME_NUMBERING;

	int						Transform_Method_;
	int						Propagation_Method_;
	int						Encoding_Method_;

	double					WAVELENGTH;

	bool					FLAG_CHANGE_DEPTH_QUANTIZATION;
	uint					DEFAULT_DEPTH_QUANTIZATION;
	uint					NUMBER_OF_DEPTH_QUANTIZATION;
	bool					RANDOM_PHASE;
	bool					FLAG_SIMULATION;

	// for Simulation (reconstruction)
	//===================================================
	std::string				Simulation_Result_File_Prefix_;
	int						test_pixel_number_scale_;
	vec2					Pixel_pitch_xy_;
	ivec2					SLM_pixel_number_xy_;
	double					f_field_;
	double					eye_length_;
	double					eye_pupil_diameter_;
	vec2					eye_center_xy_;
	double					focus_distance_;
	int						sim_type_;
	double					sim_from_;
	double					sim_to_;
	int						sim_step_num_;


};


#endif