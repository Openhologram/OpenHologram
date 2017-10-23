
#include	"Hologram/HologramGenerator.h"
#include	"Hologram/fftw3.h"
#include    "graphics/sys.h"

/**
* @brief Initialize variables for the CPU implementation.
* @details Memory allocation for the CPU variables.
* @see initialize
*/
void HologramGenerator::init_CPU()
{
	if (img_src_)	delete img_src_;
	img_src_ = new double[params_.pn[0] * params_.pn[1]];

	if (dmap_src_) delete dmap_src_;
	dmap_src_ = new double[params_.pn[0] * params_.pn[1]];

	if (alpha_map_) delete alpha_map_;
	alpha_map_ = new int[params_.pn[0] * params_.pn[1]];

	if (depth_index_) delete depth_index_;
	depth_index_ = new double[params_.pn[0] * params_.pn[1]];

	if (dmap_) delete dmap_;
	dmap_ = new double[params_.pn[0] * params_.pn[1]];

	if (U_complex_)	delete U_complex_;
	U_complex_ = new Complex[params_.pn[0] * params_.pn[1]];

}

/**
* @brief Preprocess input image & depth map data for the CPU implementation.
* @details Prepare variables, img_src_, dmap_src_, alpha_map_, depth_index_.
* @param imgptr : input image data pointer
* @param dimgptr : input depth map data pointer
* @return true if input data are sucessfully prepared, flase otherwise.
* @see ReadImageDepth
*/
bool HologramGenerator::prepare_inputdata_CPU(uchar* imgptr, uchar* dimgptr)
{
	int pnx = params_.pn[0];
	int pny = params_.pn[1];

	memset(img_src_, 0, sizeof(double)*pnx * pny);
	memset(dmap_src_, 0, sizeof(double)*pnx * pny);
	memset(alpha_map_, 0, sizeof(int)*pnx * pny);
	memset(depth_index_, 0, sizeof(double)*pnx * pny);
	memset(dmap_, 0, sizeof(double)*pnx * pny);

	int k = 0;
#pragma omp parallel for private(k)
	for (k = 0; k < pnx*pny; k++)
	{
		img_src_[k] = double(imgptr[k]) / 255.0;
		dmap_src_[k] = double(dimgptr[k]) / 255.0;
		alpha_map_[k] = (imgptr[k] > 0 ? 1 : 0);
		dmap_[k] = (1 - dmap_src_[k])*(params_.far_depthmap - params_.near_depthmap) + params_.near_depthmap;

		if (FLAG_CHANGE_DEPTH_QUANTIZATION == 0)
			depth_index_[k] = DEFAULT_DEPTH_QUANTIZATION - double(dimgptr[k]);
	}
}

/**
* @brief Quantize depth map on the CPU, when the number of depth quantization is not the default value (i.e. FLAG_CHANGE_DEPTH_QUANTIZATION == 1 ).
* @details Calculate the value of 'depth_index_'.
* @see GetDepthValues
*/
void HologramGenerator::change_depth_quan_CPU()
{
	int pnx = params_.pn[0];
	int pny = params_.pn[1];

	double temp_depth, d1, d2;
	int tdepth;

	for (int dtr = 0; dtr < params_.num_of_depth; dtr++)
	{
		temp_depth = dlevel_[dtr];
		d1 = temp_depth - dstep_ / 2.0;
		d2 = temp_depth + dstep_ / 2.0;

		int p;
#pragma omp parallel for private(p)
		for (p = 0; p < pnx * pny; p++)
		{
			int tdepth;
			if (dtr < params_.num_of_depth - 1)
				tdepth = (dmap_[p] >= d1 ? 1 : 0) * (dmap_[p] < d2 ? 1 : 0);
			else
				tdepth = (dmap_[p] >= d1 ? 1 : 0) * (dmap_[p] <= d2 ? 1 : 0);

			depth_index_[p] += tdepth*(dtr + 1);
		}
	}

	//writeIntensity_gray8_bmp("test.bmp", pnx, pny, depth_index_);
}

/**
* @brief Main method for generating a hologram on the CPU.
* @details For each depth level, 
*   1. find each depth plane of the input image.
*   2. apply carrier phase delay.
*   3. propagate it to the hologram plan.
*   4. accumulate the result of each propagation.
* .
* The final result is accumulated in the variable 'U_complex_'.
* @param frame : the frame number of the image.
* @see Calc_Holo_by_Depth, Propagation_AngularSpectrum_CPU
*/
void HologramGenerator::Calc_Holo_CPU(int frame)
{
	int pnx = params_.pn[0];
	int pny = params_.pn[1];

	memset(U_complex_, 0.0, sizeof(Complex)*pnx*pny);

	int depth_sz = params_.render_depth.size();

	int p = 0;
#pragma omp parallel for private(p)
	for (p = 0; p < depth_sz; ++p)
	{
		int dtr = params_.render_depth[p];
		double temp_depth = dlevel_transform_[dtr - 1];

		Complex* u_o = new Complex[pnx*pny];
		memset(u_o, 0.0, sizeof(Complex)*pnx*pny);

		double sum = 0.0;
		int i = 0;
#pragma omp parallel for private(i)
		for (i = 0; i < pnx * pny; i++)
		{
			u_o[i].a = img_src_[i] * alpha_map_[i] * (depth_index_[i] == dtr ? 1.0 : 0.0);
			sum += u_o[i].a;
		}

		if (sum > 0.0)
		{
			LOG("Frame#: %d, Depth: %d of %d, z = %f mm\n", frame, dtr, params_.num_of_depth, -temp_depth * 1000);

			Complex rand_phase_val;
			get_rand_phase_value(rand_phase_val);

			Complex carrier_phase_delay(0, params_.k* temp_depth);
			exponent_complex(&carrier_phase_delay);

			int i = 0;
#pragma omp parallel for private(i)
			for (i = 0; i < pnx * pny; i++)
				u_o[i] = u_o[i] * rand_phase_val * carrier_phase_delay;

			if (Propagation_Method_ == 0) 
				Propagation_AngularSpectrum_CPU('S', u_o, -temp_depth);
			
			
		}
		else
			LOG("Frame#: %d, Depth: %d of %d : Nothing here\n", frame, dtr, params_.num_of_depth);

		delete u_o;
	}

	//writeIntensity_gray8_real_bmp("final_fr", pnx, pny, U_complex_);

}

/**
* @brief Angular spectrum propagation method for CPU implementation.
* @details The propagation results of all depth levels are accumulated in the variable 'U_complex_'.
* @param domain : Spatial domain -> 'S', Frequency domain -> 'F'
*  If the input data is in the spatial domain, this function converts it to the frequency domain.
* @param input_u : each depth plane data.
* @param propagation_dist : the distance from the object to the hologram plane.
* @see Calc_Holo_by_Depth, Calc_Holo_CPU, fftwShift
*/
void HologramGenerator::Propagation_AngularSpectrum_CPU(char domain, Complex* input_u, double propagation_dist)
{
	int pnx = params_.pn[0];
	int pny = params_.pn[1];
	double ppx = params_.pp[0];
	double ppy = params_.pp[1];
	double ssx = params_.ss[0];
	double ssy = params_.ss[1];
	double lambda = params_.lambda;

	if (domain == 'S')
		fftwShift(input_u, input_u, pnx, pny, 1, false);

	int i = 0;
#pragma omp parallel for private(i)
	for (i = 0; i < pnx * pny; i++)
	{
		double x = i % pnx;
		double y = i / pnx;

		double fxx = (-1.0 / (2.0*ppx)) + (1.0 / ssx) * x;
		double fyy = (1.0 / (2.0*ppy)) - (1.0 / ssy) - (1.0 / ssy) * y;

		double sval = sqrt(1 - (lambda*fxx)*(lambda*fxx) - (lambda*fyy)*(lambda*fyy));
		sval *= params_.k * propagation_dist;
		Complex kernel(0, sval);
		exponent_complex(&kernel);

		int prop_mask = ((fxx * fxx + fyy * fyy) < (params_.k *params_.k)) ? 1 : 0;

		Complex u_frequency;
		if (prop_mask == 1)
			u_frequency = kernel * input_u[i];

		U_complex_[i] = U_complex_[i] + u_frequency;
	}

	//fftwShift(u_frequency, u_spatial, pnx, pny, -1, true);
}

/**
* @brief Encode the CGH according to a signal location parameter on the CPU.
* @details The CPU variable, u255_fringe_ on CPU has the final result.
* @param cropx1 : the start x-coordinate to crop.
* @param cropx2 : the end x-coordinate to crop.
* @param cropy1 : the start y-coordinate to crop.
* @param cropy2 : the end y-coordinate to crop.
* @param sig_location : ivec2 type,
*  sig_location[0]: upper or lower half, sig_location[1]:left or right half.
* @see Encoding_Symmetrization, fftwShift
*/
void HologramGenerator::encoding_CPU(int cropx1, int cropx2, int cropy1, int cropy2, ivec2 sig_location)
{
	int pnx = params_.pn[0];
	int pny = params_.pn[1];

	Complex* h_crop = new Complex[pnx*pny];
	memset(h_crop, 0.0, sizeof(Complex)*pnx*pny);

	int p = 0;
#pragma omp parallel for private(p)	
	for (p = 0; p < pnx*pny; p++)
	{
		int x = p % pnx;
		int y = p / pnx;
		if (x >= cropx1 && x <= cropx2 && y >= cropy1 && y <= cropy2)
			h_crop[p] = U_complex_[p];
	}

	fftwShift(h_crop, h_crop, pnx, pny, -1, true);

	memset(u255_fringe_, 0.0, sizeof(double)*pnx*pny);
	int i = 0;
#pragma omp parallel for private(i)	
	for (i = 0; i < pnx*pny; i++) {

		Complex shift_phase(1, 0);
		get_shift_phase_value(shift_phase, i, sig_location);

		u255_fringe_[i] = (h_crop[i] * shift_phase).a;
	}

	//writeIntensity_gray8_bmp("fringe_255", pnx, pny, u255_fringe_);

	delete h_crop;

}

/**
* @brief Convert data from the spatial domain to the frequency domain using 2D FFT on CPU.
* @details It is equivalent to Matlab code, dst = ifftshift(fft2(fftshift(src))).
* @param src : input data variable
* @param dst : output data variable
* @param nx : the number of column of the input data
* @param ny : the number of row of the input data
* @param type : If type == 1, forward FFT, if type == -1, backward FFT.
* @param bNomarlized : If bNomarlized == true, normalize the result after FFT.
* @see Propagation_AngularSpectrum_CPU, encoding_CPU
*/
void HologramGenerator::fftwShift(Complex* src, Complex* dst, int nx, int ny, int type, bool bNomarlized)
{
	Complex* tmp = new Complex[nx*ny];
	memset(tmp, 0.0, sizeof(Complex)*nx*ny);
	fftShift(nx, ny, src, tmp);

	fftw_complex *in, *out;
	in = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * nx * ny);
	out = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * nx * ny);
	
	int i = 0;
	#pragma omp parallel for private(i)
	for (i = 0; i < nx*ny; i++)
	{
		in[i][0] = tmp[i].a;
		in[i][1] = tmp[i].b;
	}
	fftw_plan fft_plan;
	if (type == 1)
		fft_plan = fftw_plan_dft_2d(ny, nx, in, out, FFTW_FORWARD, FFTW_ESTIMATE);
	else
		fft_plan = fftw_plan_dft_2d(ny, nx, in, out, FFTW_BACKWARD, FFTW_ESTIMATE);
	fftw_execute(fft_plan);

	int normalF = 1;
	if (bNomarlized) normalF = nx * ny;
	memset(tmp, 0, sizeof(Complex)*nx*ny);

	int k = 0;
	#pragma omp parallel for private(k)
	for (k = 0; k < nx*ny; k++) {
		tmp[k].a = out[k][0] / normalF;
		tmp[k].b = out[k][1] / normalF;
	}

	fftw_destroy_plan(fft_plan);
	fftw_free(in); fftw_free(out);

	fftShift(nx, ny, tmp, dst);
	
	delete tmp;
	
}

/**
* @brief Swap the top-left quadrant of data with the bottom-right , and the top-right quadrant with the bottom-left.
* @param nx : the number of column of the input data
* @param ny : the number of row of the input data
* @param input : input data variable
* @param output : output data variable
* @see fftwShift
*/
void HologramGenerator::fftShift(int nx, int ny, Complex* input, Complex* output)
{
	int i = 0, j = 0;
#pragma omp parallel for private(i, j)
	for (i = 0; i < nx; i++)
	{
		for (j = 0; j < ny; j++)
		{
			int ti = i - nx / 2; if (ti < 0) ti += nx;
			int tj = j - ny / 2; if (tj < 0) tj += ny;

			output[ti + tj * nx] = input[i + j * nx];
		}
	}
}

/**
* @brief Calculate the exponential of the complex number.
* @param val : input & ouput value
* @see Propagation_AngularSpectrum_CPU, Calc_Holo_CPU
*/
void HologramGenerator::exponent_complex(Complex* val)
{
	double realv = val->a;
	double imgv = val->b;
	val->a = exp(realv)*cos(imgv);
	val->b = exp(realv)*sin(imgv);

}

/**
* @brief Calculate the shift phase value.
* @param shift_phase_val : output variable.
* @param idx : the current pixel position.
* @param sig_location :  signal location.
* @see encoding_CPU
*/
void HologramGenerator::get_shift_phase_value(Complex& shift_phase_val, int idx, ivec2 sig_location)
{
	int pnx = params_.pn[0];
	int pny = params_.pn[1];
	double ppx = params_.pp[0];
	double ppy = params_.pp[1];
	double ssx = params_.ss[0];
	double ssy = params_.ss[1];

	if (sig_location[1] != 0)
	{
		int r = idx / pnx;
		int c = idx % pnx;
		double yy = (ssy / 2.0) - (ppy)*r - ppy;

		Complex val;
		if (sig_location[1] == 1)
			val.b = 2 * PI * (yy / (4 * ppy));
		else
			val.b = 2 * PI * (-yy / (4 * ppy));

		exponent_complex(&val);
		shift_phase_val *= val;
	}

	if (sig_location[0] != 0)
	{
		int r = idx / pnx;
		int c = idx % pnx;
		double xx = (-ssx / 2.0) - (ppx)*c - ppx;

		Complex val;
		if (sig_location[0] == -1)
			val.b = 2 * PI * (-xx / (4 * ppx));
		else
			val.b = 2 * PI * (xx / (4 * ppx));

		exponent_complex(&val);
		shift_phase_val *= val;
	}

}