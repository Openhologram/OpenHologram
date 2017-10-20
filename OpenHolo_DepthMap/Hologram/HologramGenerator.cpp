
#include	"Hologram/HologramGenerator.h"
#include	<fstream>
#include	<sstream>
#include	<random>
#include    "graphics/sys.h"
#include	"matlab/mat.h"
#include	"Hologram/fftw3.h"

#include	<QtScript/QScriptEngine>
#include	<QtCore/QRegularExpression>

HologramGenerator::HologramGenerator()
{
	img_src_ = 0;
	dmap_src_ = 0;
	alpha_map_ = 0;
	depth_index_ = 0;
	dmap_ = 0;
	dstep_ = 0;
	dlevel_.clear();
	u_complex_ = 0;
	U_complex_ = 0;
	u255_fringe_ = 0;

}

HologramGenerator::~HologramGenerator()
{

}

bool HologramGenerator::readConfig()
{
	std::string inputFileName_ = "config_openholo.txt";

	LOG("Reading....%s\n", inputFileName_.c_str());

	std::ifstream inFile(inputFileName_.c_str());

	if (!inFile.is_open()){
		LOG("file not found.\n");
		return false;
	}

	// skip 7 lines
	std::string temp;
	getline(inFile, temp, '\n');	getline(inFile, temp, '\n');	getline(inFile, temp, '\n');
	getline(inFile, temp, '\n');	getline(inFile, temp, '\n');	getline(inFile, temp, '\n');
	getline(inFile, temp, '\n');

	inFile >> SOURCE_FOLDER;						getline(inFile, temp, '\n');
	inFile >> IMAGE_PREFIX;							getline(inFile, temp, '\n');
	inFile >> DEPTH_PREFIX;							getline(inFile, temp, '\n');
	inFile >> RESULT_FOLDER;						getline(inFile, temp, '\n');
	inFile >> RESULT_PREFIX;						getline(inFile, temp, '\n');
	inFile >> FLAG_STATIC_IMAGE;					getline(inFile, temp, '\n');
	inFile >> START_OF_FRAME_NUMBERING;				getline(inFile, temp, '\n');
	inFile >> NUMBER_OF_FRAME;						getline(inFile, temp, '\n');
	inFile >> NUMBER_OF_DIGIT_OF_FRAME_NUMBERING;	getline(inFile, temp, '\n');

	// skip 3 lines
	getline(inFile, temp, '\n');	getline(inFile, temp, '\n');	getline(inFile, temp, '\n');

	inFile >> Transform_Method_;					getline(inFile, temp, '\n');
	inFile >> Propagation_Method_;					getline(inFile, temp, '\n');
	inFile >> Encoding_Method_;						getline(inFile, temp, '\n');
	
	// skip 3 lines
	getline(inFile, temp, '\n');	getline(inFile, temp, '\n');	getline(inFile, temp, '\n');

	inFile >> params_.field_lens;					getline(inFile, temp, '\n');
	inFile >> WAVELENGTH;							getline(inFile, temp, '\n');
	params_.lambda = WAVELENGTH;
	params_.k = 2 * PI / WAVELENGTH;
	
	inFile >> temp;									
	QScriptEngine en;
	QScriptValue result = en.evaluate(QString().fromStdString(temp));
	if (en.hasUncaughtException()) {
		LOG("Error: SLM_PIXEL_NUMBER_X \n");
		return false;
	}		

	params_.pn[0] = result.toNumber();
	getline(inFile, temp, '\n');
	
	inFile >> temp;				
	result = en.evaluate(QString().fromStdString(temp));
	if (en.hasUncaughtException()) {
		LOG("Error: SLM_PIXEL_NUMBER_Y \n");
		return false;
	}
	
	params_.pn[1] = result.toNumber();
	getline(inFile, temp, '\n');
	
	inFile >> temp;		
	result = en.evaluate(QString().fromStdString(temp));
	if (en.hasUncaughtException()) {
		LOG("Error: SLM_PIXEL_PITCH_X \n");
		return false;
	}
	params_.pp[0] = result.toNumber();
	getline(inFile, temp, '\n');

	inFile >> temp;		
	result = en.evaluate(QString().fromStdString(temp));
	if (en.hasUncaughtException()) {
		LOG("Error: SLM_PIXEL_PITCH_Y \n");
		return false;
	}
	params_.pp[1] = result.toNumber();
	getline(inFile, temp, '\n');

	params_.ss[0] = params_.pp[0] * params_.pn[0];
	params_.ss[1] = params_.pp[1] * params_.pn[1];

	// skip 3 lines
	getline(inFile, temp, '\n');	getline(inFile, temp, '\n');	getline(inFile, temp, '\n');

	double NEAR_OF_DEPTH_MAP, FAR_OF_DEPTH_MAP;
	inFile >> NEAR_OF_DEPTH_MAP;					getline(inFile, temp, '\n');
	inFile >> FAR_OF_DEPTH_MAP;						getline(inFile, temp, '\n');

	params_.near_depthmap = min(NEAR_OF_DEPTH_MAP, FAR_OF_DEPTH_MAP);
	params_.far_depthmap = max(NEAR_OF_DEPTH_MAP, FAR_OF_DEPTH_MAP);

	inFile >> FLAG_CHANGE_DEPTH_QUANTIZATION;		getline(inFile, temp, '\n');
	inFile >> DEFAULT_DEPTH_QUANTIZATION;			getline(inFile, temp, '\n');
	inFile >> NUMBER_OF_DEPTH_QUANTIZATION;			getline(inFile, temp, '\n');
		
	if (FLAG_CHANGE_DEPTH_QUANTIZATION == 0)
		params_.num_of_depth = DEFAULT_DEPTH_QUANTIZATION;
	else
		params_.num_of_depth = NUMBER_OF_DEPTH_QUANTIZATION;
	
	getline(inFile, temp, '\n');
	QString src = QString().fromStdString(temp);
	src = src.left(src.indexOf("//")).trimmed();
	
	QRegularExpressionMatch match;
	if (src.contains(":"))
	{
		QRegularExpression re1("(\\d+):(\\d+)");
		match = re1.match(src);
		if (match.hasMatch()) {
			int start = match.captured(1).toInt();
			int end = match.captured(2).toInt();
			params_.render_depth.clear();
			for (int k = start; k <= end; k++)
				params_.render_depth.push_back(k);
		}
	}else {
		
		QRegularExpression re2("(\\d+)");
		QRegularExpressionMatchIterator i = re2.globalMatch(src);
		params_.render_depth.clear();
		while (i.hasNext()) {
			int num = i.next().captured(1).toInt();
			params_.render_depth.push_back(num);
		}
	}
	if (params_.render_depth.empty()){
		LOG("Error: RENDER_DEPTH \n");
		return false;
	}
		
	inFile >> RANDOM_PHASE;							getline(inFile, temp, '\n');
	
	//==Simulation parameters ======================================================================
	getline(inFile, temp, '\n');	getline(inFile, temp, '\n');	getline(inFile, temp, '\n');

	inFile >> Simulation_Result_File_Prefix_;			getline(inFile, temp, '\n');
	inFile >> test_pixel_number_scale_;					getline(inFile, temp, '\n');
	inFile >> eye_length_;								getline(inFile, temp, '\n');
	inFile >> eye_pupil_diameter_;						getline(inFile, temp, '\n');
	inFile >> eye_center_xy_[0];						getline(inFile, temp, '\n');
	inFile >> eye_center_xy_[1];						getline(inFile, temp, '\n');
	inFile >> focus_distance_;							getline(inFile, temp, '\n');

	getline(inFile, temp, '\n');	getline(inFile, temp, '\n');	getline(inFile, temp, '\n');
	getline(inFile, temp, '\n');	getline(inFile, temp, '\n');	
	
	inFile >> sim_type_;								getline(inFile, temp, '\n');
	inFile >> sim_from_;								getline(inFile, temp, '\n');
	inFile >> sim_to_;									getline(inFile, temp, '\n');
	inFile >> sim_step_num_;							getline(inFile, temp, '\n');
	
	//=====================================================================================
	inFile.close();

	LOG("done\n");

	return true;

}

void HologramGenerator::initialize()
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
	
	dstep_ = 0;
	dlevel_.clear();

	if (u_complex_)	delete u_complex_;
	u_complex_ = new Complex[params_.pn[0] * params_.pn[1]];

	if (U_complex_)	delete U_complex_;
	U_complex_ = new Complex[params_.pn[0] * params_.pn[1]];

	if (u255_fringe_)		delete u255_fringe_;
	u255_fringe_ = new double[params_.pn[0] * params_.pn[1]];
	
}

void HologramGenerator::GenerateHologram()
{
	int num_of_frame;
	if (FLAG_STATIC_IMAGE == 0)
		num_of_frame = NUMBER_OF_FRAME;
	else
		num_of_frame = 1;

	for (int ftr = 0; ftr <= num_of_frame - 1; ftr++)
	{
		LOG("Calculating hologram of frame %d.\n", ftr);

		if (!ReadImageDepth()) {
			LOG("Error: Reading image of frame %d.\n", ftr);
			continue;
		}

		GetDepthValues();

		if (Transform_Method_ == 0)
			TransformViewingWindow();

		Calc_Holo_by_Depth(ftr);
		
		if (Encoding_Method_ == 0)
			Encoding_Symmetrization(ivec2(0,1));

		Write_Result_image();

		//writeMatFileDouble("u255_fringe", u255_fringe_);
		//writeMatFileComplex("u_complex", u_complex_, 0);
		//writeMatFileComplex("U_complex", U_complex_, 1);
	}

}

bool HologramGenerator::ReadImageDepth()
{
	// not implemented!
	if (FLAG_STATIC_IMAGE == 0)
		return false;

	QString sdir = "./" + QString().fromStdString(SOURCE_FOLDER);
	QDir dir(sdir);
	if (!dir.exists())
	{
		LOG("Error: Source folder does not exist: %s.\n", sdir.toStdString().c_str());
		return false;
	}

	QStringList nlist;
	nlist << QString().fromStdString(IMAGE_PREFIX) + "*.*";
	dir.setNameFilters(nlist);
	dir.setFilter(QDir::Files | QDir::NoDotAndDotDot | QDir::NoSymLinks);
	QStringList names = dir.entryList();
	if (names.empty()) {
		LOG("Error: Source image does not exist.\n");
		return false;
	}

	QString imgfullname = sdir + "/" + names[0];
	QImage img;
	img.load(imgfullname);
	img = img.convertToFormat(QImage::Format_Grayscale8);

	nlist.clear();
	names.clear();
	nlist << QString().fromStdString(DEPTH_PREFIX) + "*.*";
	dir.setNameFilters(nlist);
	dir.setFilter(QDir::Files | QDir::NoDotAndDotDot | QDir::NoSymLinks);
	names = dir.entryList();
	if (names.empty()) {
		LOG("Error: Source depthmap does not exist.\n");
		return false;
	}

	QString dimgfullname = sdir + "/" + names[0];
	QImage dimg;
	dimg.load(dimgfullname);
	dimg = dimg.convertToFormat(QImage::Format_Grayscale8);

	//resize image
	int pnx = params_.pn[0];
	int pny = params_.pn[1];

	QSize imgsize = img.size();
	if (imgsize.width() != pnx || imgsize.height() != pny)
		img = img.scaled(pnx, pny);

	QSize dimgsize = dimg.size();
	if (dimgsize.width() != pnx || dimgsize.height() != pny)
		dimg = dimg.scaled(pnx, pny);

	// normalize
	memset(img_src_, 0, sizeof(double)*pnx * pny);
	memset(dmap_src_, 0, sizeof(double)*pnx * pny);
	memset(alpha_map_, 0, sizeof(int)*pnx * pny);
	memset(depth_index_, 0, sizeof(double)*pnx * pny);
	memset(dmap_, 0, sizeof(double)*pnx * pny);

	uchar* imgptr = img.bits();
	uchar* dimgptr = dimg.bits();
	for (int k = 0; k < pnx*pny; k++)
	{
		img_src_[k] = double(imgptr[k]) / 255.0;
		dmap_src_[k] = double(dimgptr[k]) / 255.0;
		alpha_map_[k] = (imgptr[k] > 0 ? 1 : 0);
		dmap_[k] = (1 - dmap_src_[k])*(params_.far_depthmap - params_.near_depthmap) + params_.near_depthmap;

		if (FLAG_CHANGE_DEPTH_QUANTIZATION == 0)
			depth_index_[k] = DEFAULT_DEPTH_QUANTIZATION - double(dimgptr[k]);
	}
	
	//writeIntensity_gray8_bmp("test.bmp", pnx, pny, img_src_);
	//dimg.save("test_dmap.bmp");
	//img.save("test_img.bmp");
	return true;

}

// Calculate the physical distances of depth map layers
void HologramGenerator::GetDepthValues()
{
	if (params_.num_of_depth > 1)
	{
		dstep_ = (params_.far_depthmap - params_.near_depthmap) / (params_.num_of_depth - 1);
		double val = params_.near_depthmap;
		while (val <= params_.far_depthmap)
		{
			dlevel_.push_back(val);
			val += dstep_;
		}

	} else {

		dstep_ = (params_.far_depthmap + params_.near_depthmap) / 2;
		dlevel_.push_back(params_.far_depthmap - params_.near_depthmap);

	}
	
	if (FLAG_CHANGE_DEPTH_QUANTIZATION == 1)
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

			for (int p = 0; p < pnx * pny; p++)
			{
				if (dtr < params_.num_of_depth - 1)
					tdepth = (dmap_[p] >= d1 ? 1 : 0) * (dmap_[p] < d2 ? 1 : 0);
				else
					tdepth = (dmap_[p] >= d1 ? 1 : 0) * (dmap_[p] <= d2 ? 1 : 0);

				depth_index_[p] += tdepth*(dtr+1);
			}
		}
		
		//writeIntensity_gray8_bmp("test.bmp", pnx, pny, depth_index_);
	}
}

void HologramGenerator::TransformViewingWindow()
{
	int pnx = params_.pn[0];
	int pny = params_.pn[1];

	double val;
	dlevel_transform_.clear();
	for (int p = 0; p < dlevel_.size(); p++)
	{
		val = -params_.field_lens * dlevel_[p] / (dlevel_[p] - params_.field_lens);
		dlevel_transform_.push_back(val);
	}
}

//Prepare: dmap_src_(normalized), img_src_, alpha_map_
//Output : u_complex
void HologramGenerator::Calc_Holo_by_Depth(int frame)
{
	int pnx = params_.pn[0];
	int pny = params_.pn[1];

	bool flag_first_plane = true;

	std::default_random_engine generator;
	std::uniform_real_distribution<double> distribution(0.0, 1.0);
	
	int num_rdepth = params_.render_depth.size();
	int dtr;
	double temp_depth;
	double ran_val, rand_phase_val, carrier_phase_delay;
	double* u_o = new double[pnx*pny];
	Complex* prev_u = new Complex[pnx*pny];
	double prev_z;
	int prev_num;

	for (auto i = params_.render_depth.begin();i!= params_.render_depth.end(); ++i)
	{
		dtr = *i;
		temp_depth = dlevel_transform_[dtr-1];
		memset(u_o, 0.0, sizeof(double)*pnx*pny);

		double sum = 0.0;
		for (int p = 0; p < pnx * pny; p++)
		{
			u_o[p] = img_src_[p] * alpha_map_[p] * (depth_index_[p] == dtr ? 1.0 : 0.0);
			sum += u_o[p];
		}
		if (sum > 0.0)
		{			
			if (RANDOM_PHASE > 0)
			{
				for (int p = 0; p < pnx * pny; p++)
				{
					ran_val = distribution(generator);
					Complex cvalue(0, 2 * PI * ran_val);
					exponent_complex(&cvalue);
					rand_phase_val = cvalue.a;
				
					u_o[p] *= rand_phase_val;
				}
			}

			//writeIntensity_gray8_bmp("u_o.bmp", pnx, pny, u_o);

			if (flag_first_plane == true)
			{
				flag_first_plane = false;

				Complex carrier_phase_delay(0, params_.k* temp_depth);
				exponent_complex(&carrier_phase_delay);

				memset(prev_u, 0.0, sizeof(Complex)*pnx*pny);
				for (int p = 0; p < pnx * pny; p++)
				{
					Complex tvalue = u_o[p] * carrier_phase_delay;
					prev_u[p] = tvalue;
				}
				//writeIntensity_gray8_bmp("prev_u.bmp", pnx, pny, prev_u);

				prev_z = temp_depth;
				prev_num = dtr;

			} else {

				LOG("Frame#: %d, Depth: %d of %d, z = %f mm\n", frame, prev_num, params_.num_of_depth, dlevel_[prev_num-1]*1000);
				
				if (Propagation_Method_ == 0)
					Propagation_AngularSpectrum('S', prev_u, temp_depth - prev_z);

				Complex carrier_phase_delay(0, params_.k* temp_depth);
				exponent_complex(&carrier_phase_delay);

				memset(prev_u, 0.0, sizeof(Complex)*pnx*pny);
				for (int p = 0; p < pnx * pny; p++)
				{
					Complex tvalue = u_o[p] * carrier_phase_delay;
					prev_u[p] = u_complex_[p]*(u_o[p]==0.0?1.0:0.0) + tvalue;
				}
				prev_z = temp_depth;
				prev_num = dtr;
			}

		} else 
			LOG("Frame#: %d, Depth: %d of %d : Nothing here\n", frame, dtr, params_.num_of_depth);
	}

	//Back propagation to field lens plane
	Complex csum;
	for (int p = 0; p < pnx * pny; p++)
		csum += prev_u[p];

	if (csum.a != 0.0 || csum.b != 0.0)
	{ 
		LOG("Frame#: %d, Depth: %d of %d\n", frame, dtr, params_.num_of_depth);

		if (Propagation_Method_ == 0)
			Propagation_AngularSpectrum('S', prev_u, -prev_z);
	}

	//writeIntensity_gray8_real_bmp("final_sp", pnx, pny, u_complex_);
	//writeIntensity_gray8_real_bmp("final_fr", pnx, pny, U_complex_);

	delete u_o, prev_u;

}

void HologramGenerator::Propagation_AngularSpectrum(char domain, Complex* input_u, double propagation_dist)
{
	int pnx = params_.pn[0];
	int pny = params_.pn[1];
	double ppx = params_.pp[0];
	double ppy = params_.pp[1];
	double ssx = params_.ss[0];
	double ssy = params_.ss[1];
	double lambda = params_.lambda;

	Complex* U = new Complex[pnx*pny];
	memset(u_complex_, 0.0, sizeof(Complex)*pnx*pny);
	memset(U_complex_, 0.0, sizeof(Complex)*pnx*pny);
	memset(U, 0.0, sizeof(Complex)*pnx*pny);

	if (domain == 'S') 	
		fftwShift(input_u, U, pnx, pny, 1, false);

	else if (domain == 'F') 
		memcpy(U, input_u, sizeof(Complex)*pnx*pny);
		
	double fxx, fyy, x, y;
	int prop_mask;
	for (int p = 0; p < pnx * pny; p++)
	{
		x = p % pnx;
		y = p / pnx;

		fxx = (-1.0 / (2.0*ppx)) + (1.0 / ssx) * x;
		fyy = ( 1.0 / (2.0*ppy)) - (1.0 / ssy) - (1.0 / ssy) * y;

		double sval = sqrt(1 - (lambda*fxx)*(lambda*fxx) - (lambda*fyy)*(lambda*fyy));
		sval *= params_.k * propagation_dist;
		Complex kernel(0,sval);
		exponent_complex(&kernel);

		prop_mask = ((fxx * fxx + fyy * fyy) < (params_.k *params_.k))?1:0;

		if (prop_mask == 1)
			U_complex_[p] = kernel * U[p];
		else
			U_complex_[p] = Complex(0, 0);

	}

	fftwShift(U_complex_, u_complex_, pnx, pny, -1, true);

	delete U;
}

void HologramGenerator::Encoding_Symmetrization(ivec2 sig_location)
{
	int pnx = params_.pn[0];
	int pny = params_.pn[1];
	double ppx = params_.pp[0];
	double ppy = params_.pp[1];
	double ssx = params_.ss[0];
	double ssy = params_.ss[1];

	Complex* h_crop = new Complex[pnx*pny];
	Complex* shift_phase = new Complex[pnx*pny];
	for (int k = 0; k < pnx*pny; k++)
	{
		h_crop[k].a = 0.0;		
		h_crop[k].b = 0.0;
		shift_phase[k].a = 1.0; 
		shift_phase[k].b = 0.0;
	}

	int cropx1, cropx2, cropx, cropy1, cropy2, cropy;
	if (sig_location[1] == 0) //Left or right half
	{
		cropy1 = 1;
		cropy2 = pny;

	} else {

		cropy = floor(pny / 2);
		cropy1 = cropy - floor(cropy / 2);
		cropy2 = cropy1 + cropy - 1;

		double yy;
		for (int k = 0; k < pnx*pny; k++)
		{
			int r = k / pnx;
			int c = k % pnx;
			yy = (ssy / 2.0) - (ppy)*r - ppy;

			Complex val;
			if (sig_location[1] == 1)
				val.b = 2 * PI * (yy / (4 * ppy));
			else
				val.b = 2 * PI * (-yy / (4 * ppy));
						
			exponent_complex(&val);
			shift_phase[k] *= val;
		}
	}

	if (sig_location[0] == 0) // Upper or lower half
	{
		cropx1 = 1;
		cropx2 = pnx;

	} else {
		cropx = floor(pnx / 2);
		cropx1 = cropx - floor(cropx / 2);
		cropx2 = cropx1 + cropx - 1;

		double xx;
		for (int k = 0; k < pnx*pny; k++)
		{
			int r = k / pnx;
			int c = k % pnx;
			xx = (-ssx / 2.0) - (ppx)*c - ppx;

			Complex val;
			if (sig_location[0] == -1)
				val.b = 2 * PI * (-xx / (4 * ppx));
			else
				val.b = 2 * PI * ( xx / (4 * ppx));

			exponent_complex(&val);
			shift_phase[k] *= val;
		}
	}

	cropx1 -= 1;
	cropx2 -= 1;
	cropy1 -= 1;
	cropy2 -= 1;

	for (int k = 0; k < pnx*pny; k++)
	{
		int x = k % pnx;
		int y = k / pnx;
		if (x >= cropx1 && x <= cropx2 && y >= cropy1 && y <= cropy2)
			h_crop[k] = U_complex_[k];
	}

	fftwShift(h_crop, h_crop, pnx, pny, -1, true);

	double* u = new double[pnx*pny];
	memset(u, 0.0, sizeof(double)*pnx*pny);
	for (int i = 0; i < pnx*pny; i++) {
		h_crop[i] = h_crop[i] * shift_phase[i];
		u[i] = h_crop[i].a;
	}
	
	double min_val, max_val;
	min_val = u[0];
	max_val = u[0];
	for (int i = 0; i < pnx*pny; ++i)
	{
		if (min_val > u[i])
			min_val = u[i];
		else if (max_val < u[i])
			max_val = u[i];
	}

	memset(u255_fringe_, 0.0, sizeof(double)*pnx*pny);
	for (int i = 0; i < pnx*pny; ++i)
		u255_fringe_[i] = (u[i] - min_val) / (max_val - min_val) * 255;
	
	delete h_crop, shift_phase, u;

}

void HologramGenerator::Write_Result_image()
{
	QDir dir("./");
	if (!dir.exists(QString().fromStdString(RESULT_FOLDER)))
		dir.mkdir(QString().fromStdString(RESULT_FOLDER));

	QString fname = "./" + QString().fromStdString(RESULT_FOLDER) + "/"
		+ QString().fromStdString(RESULT_PREFIX) + ".bmp";

	int pnx = params_.pn[0];
	int pny = params_.pn[1];
	int px = pnx / 3;
	int py = pny;

	uchar* data = new uchar[pnx*pny];
	memset(data, 0, sizeof(uchar)*pnx*pny);
	for (int k = 0; k < pnx*pny; k++)
		data[k] = (uint)u255_fringe_[k];
	
	QImage img(data, px, py, QImage::Format::Format_RGB888);
	img.save(QString(fname));

}

void HologramGenerator::fftShift(int nx, int ny, Complex* input, Complex* output)
{
	int ti, tj;
	for (int i = 0; i < nx; i++)
	{
		for (int j = 0; j < ny; j++)
		{
			ti = i - nx / 2; if (ti < 0) ti += nx;
			tj = j - ny / 2; if (tj < 0) tj += ny;

			output[ti + tj * nx] = input[i + j * nx];
		}
	}
}

void HologramGenerator::exponent_complex(Complex* val)
{
	double realv = val->a;
	double imgv = val->b;
	val->a = exp(realv)*cos(imgv);
	val->b = exp(realv)*sin(imgv);

}

void HologramGenerator::writeIntensity_gray8_bmp(const char* fileName, int nx, int ny, double* intensity)
{
	const int n = nx*ny;

	double min_val, max_val;
	min_val = intensity[0];
	max_val = intensity[0];

	for (int i = 0; i < n; ++i)
	{
		if (min_val > intensity[i])
			min_val = intensity[i];
		else if (max_val < intensity[i])
			max_val = intensity[i];
	}

	char fname[100];
	strcpy(fname, fileName);
	strcat(fname, ".bmp");

	//LOG("minval %f, max val %f\n", min_val, max_val);
	unsigned char* cgh = new unsigned char[n];

	for (int i = 0; i < n; ++i){
		double val = 255 * ((intensity[i] - min_val) / (max_val - min_val));
		cgh[i] = val;
	}

	QImage img(cgh, nx, ny, QImage::Format::Format_Grayscale8);
	img.save(QString(fname));

	delete cgh;
}

void HologramGenerator::writeIntensity_gray8_bmp(const char* fileName, int nx, int ny, Complex* complexvalue)
{
	const int n = nx*ny;

	double* intensity = new double[n];
	for (int i = 0; i < n; i++)
		intensity[i] = complexvalue[i].mag2();

	double min_val, max_val;
	min_val = intensity[0];
	max_val = intensity[0];

	for (int i = 0; i < n; ++i)
	{
		if (min_val > intensity[i])
			min_val = intensity[i];
		else if (max_val < intensity[i])
			max_val = intensity[i];
	}

	char fname[100];
	strcpy(fname, fileName);
	strcat(fname, ".bmp");

	//LOG("minval %e, max val %e\n", min_val, max_val);

	unsigned char* cgh;

	cgh = new unsigned char[n];

	for (int i = 0; i < n; ++i) {
		double val = (intensity[i] - min_val) / (max_val - min_val);
		//val = pow(val, 1.0 / 1.5);
		val = val * 255.0;
		unsigned char v = (uchar)val;

		cgh[i] = v;
	}

	QImage img(cgh, nx, ny, QImage::Format::Format_Grayscale8);
	img.save(QString(fname));


	delete intensity, cgh;
}

void HologramGenerator::writeIntensity_gray8_real_bmp(const char* fileName, int nx, int ny, Complex* complexvalue)
{
	const int n = nx*ny;

	double* intensity = new double[n];
	for (int i = 0; i < n; i++)
		intensity[i] = complexvalue[i].a;

	double min_val, max_val;
	min_val = intensity[0];
	max_val = intensity[0];

	for (int i = 0; i < n; ++i)
	{
		if (min_val > intensity[i])
			min_val = intensity[i];
		else if (max_val < intensity[i])
			max_val = intensity[i];
	}

	char fname[100];
	strcpy(fname, fileName);
	strcat(fname, ".bmp");

	//LOG("minval %e, max val %e\n", min_val, max_val);

	unsigned char* cgh;

	cgh = new unsigned char[n];

	for (int i = 0; i < n; ++i) {
		double val = (intensity[i] - min_val) / (max_val - min_val);
		//val = pow(val, 1.0 / 1.5);
		val = val * 255.0;
		unsigned char v = (uchar)val;

		cgh[i] = v;
	}

	QImage img(cgh, nx, ny, QImage::Format::Format_Grayscale8);
	img.save(QString(fname));


	delete intensity, cgh;
}

bool HologramGenerator::readMatFileDouble(const char* fileName, double * val)
{
	MATFile *pmat;
	mxArray *parray;

	char fname[100];
	strcpy(fname, fileName);

	pmat = matOpen(fname, "r");
	if (pmat == NULL) {
		LOG("Error creating file %s\n", fname);
		return false;
	}

	//===============================================================
	parray = matGetVariableInfo(pmat, "inputmat");

	if (parray == NULL) {
		printf("Error reading existing matrix \n");
		return false;
	}

	int m = mxGetM(parray);
	int n = mxGetN(parray);

	if (params_.pn[0] * params_.pn[1] != m*n)
	{
		printf("Error : different matrix dimension \n");
		return false;
	}

	double* dst_r;
	parray = matGetVariable(pmat, "inputmat");
	dst_r = val;

	double* CompRealPtr = mxGetPr(parray);

	for (int col = 0; col < n; col++)
	{
		for (int row = 0; row < m; row++)
		{
			dst_r[n*row + col] = *CompRealPtr++;
		}
	}

	// clean up 
	mxDestroyArray(parray);

	if (matClose(pmat) != 0) {
		LOG("Error closing file %s\n", fname);
		return false;
	}

	LOG("Read Mat file %s\n", fname);
	return true;
}

void HologramGenerator::writeMatFileComplex(const char* fileName, Complex* val, int type)
{
	MATFile *pmat;
	mxArray *pData;

	char fname[100];
	strcpy(fname, fileName);
	if (type == 0)
		strcat(fname, "_spatial.mat");
	else
		strcat(fname, "_frequency.mat");

	pmat = matOpen(fname, "w");
	if (pmat == NULL) {
		LOG("Error creating file %s\n", fname);
		return;
	}

	ivec2 pn = params_.pn;
	int w = pn[0];
	int h = pn[1];
	const int n = w * h;

	pData = mxCreateDoubleMatrix(h, w, mxCOMPLEX);
	if (pData == NULL) {
		LOG("Unable to create mxArray.\n");
		return;
	}

	double* CompRealPtr = mxGetPr(pData);
	double* CompImgPtr = mxGetPi(pData);

	for (int col = 0; col < w; col++)
	{
		//for (int row = h-1; row >= 0; row--)
		for (int row = 0; row < h; row++)
		{
			*CompRealPtr++ = val[w*row + col].a;
			*CompImgPtr++ = val[w*row + col].b;
		}
	}

	int status;
	if (type == 0)
		status = matPutVariable(pmat, "Spatial", pData);
	else
		status = matPutVariable(pmat, "Frequency", pData);

	if (status != 0) {
		LOG("%s :  Error using matPutVariable on line %d\n", __FILE__, __LINE__);
		return;
	}

	///* clean up 
	mxDestroyArray(pData);

	if (matClose(pmat) != 0) {
		LOG("Error closing file %s\n", fname);
		return;
	}

	LOG("Write Mat file %s\n", fname);


}
void HologramGenerator::writeMatFileDouble(const char* fileName, double * val)
{
	MATFile *pmat;
	mxArray *pData;

	char fname[100];
	strcpy(fname, fileName);
	strcat(fname, ".mat");

	pmat = matOpen(fname, "w");
	if (pmat == NULL) {
		LOG("Error creating file %s\n", fname);
		return;
	}

	ivec2 pn = params_.pn;
	int w = pn[0];
	int h = pn[1];
	const int n = w * h;

	pData = mxCreateDoubleMatrix(h, w, mxREAL);
	if (pData == NULL) {
		LOG("Unable to create mxArray.\n");
		return;
	}

	double* CompRealPtr = mxGetPr(pData);
	for (int col = 0; col < w; col++)
	{
		//for (int row = h-1; row >= 0; row--)
		for (int row = 0; row < h; row++)
			*CompRealPtr++ = val[w*row + col];
	}

	int status;
	status = matPutVariable(pmat, "inputmat", pData);

	if (status != 0) {
		LOG("%s :  Error using matPutVariable on line %d\n", __FILE__, __LINE__);
		return;
	}

	///* clean up 
	mxDestroyArray(pData);

	if (matClose(pmat) != 0) {
		LOG("Error closing file %s\n", fname);
		return;
	}

	LOG("Write Mat file %s\n", fname);
}

void HologramGenerator::ReconstructImage()
{
	if (!u255_fringe_) {
		u255_fringe_ = new double[params_.pn[0] * params_.pn[1]];
		if (!readMatFileDouble("u255_fringe.mat", u255_fringe_))
			return;
	}

	Pixel_pitch_xy_[0] = params_.pp[0] / test_pixel_number_scale_;
	Pixel_pitch_xy_[1] = params_.pp[1] / test_pixel_number_scale_;

	SLM_pixel_number_xy_[0] = params_.pn[0] / test_pixel_number_scale_;
	SLM_pixel_number_xy_[1] = params_.pn[1] / test_pixel_number_scale_;

	f_field_ = params_.field_lens;

	sim_final_ = new double[SLM_pixel_number_xy_[0] * SLM_pixel_number_xy_[1]];
	memset(sim_final_, 0.0, sizeof(double)*SLM_pixel_number_xy_[0] * SLM_pixel_number_xy_[1]);

	double vmax, vmin, vstep, vval;
	if (sim_step_num_ > 1)
	{
		vmax = max(sim_to_, sim_from_);
		vmin = min(sim_to_, sim_from_);
		vstep = (sim_to_ - sim_from_) / (sim_step_num_ - 1);

	} else if (sim_step_num_ == 1) {
		vval = (sim_to_ + sim_from_) / 2.0;
	}

	Complex* hh_e = new Complex[SLM_pixel_number_xy_[0] * SLM_pixel_number_xy_[1]];
	Test_Propagation_to_Eye_Pupil(hh_e);

	if (sim_step_num_ > 0)
	{
		for (int vtr = 1; vtr <= sim_step_num_; vtr++)
		{
			LOG("Calculating Frame %d of %d \n", vtr, sim_step_num_);
			if (sim_step_num_ > 1)
				vval = vmin + (vtr - 1)*vstep;
			if (sim_type_ == 0)
				focus_distance_ = vval;
			else
				eye_center_xy_[1] = vval;

			Reconstruction(hh_e);
			Write_Simulation_image(vtr, vval);
		}

	} else {
		
		Reconstruction(hh_e);
		Write_Simulation_image(0,0);

	}
	
	delete hh_e;

}

void HologramGenerator::Test_Propagation_to_Eye_Pupil(Complex* hh_e)
{
	int pnx = SLM_pixel_number_xy_[0];
	int pny = SLM_pixel_number_xy_[1];
	double ppx = Pixel_pitch_xy_[0];
	double ppy = Pixel_pitch_xy_[1];
	double F_size_x = pnx*ppx;
	double F_size_y = pny*ppy;
	double lambda = params_.lambda;

	Complex* hh = new Complex[pnx*pny];
	Complex* hh_e_0 = new Complex[pnx*pny];
	for (int k = 0; k < pnx*pny; k++)
	{
		hh[k].a = u255_fringe_[k];
		hh[k].b = 0.0;

		hh_e_0[k].a = 0.0;		
		hh_e_0[k].b = 0.0;
	}

	fftwShift(hh, hh_e_0, pnx, pny, 1, false);
		
	double pp_ex = lambda * f_field_ / F_size_x;
	double pp_ey = lambda * f_field_ / F_size_y;
	double E_size_x = pp_ex*pnx;
	double E_size_y = pp_ey*pny;
	
	double xe, ye, x, y;
	memset(hh, 0.0, sizeof(Complex)*pnx*pny);
	for (int p = 0; p < pnx * pny; p++)
	{
		x = p % pnx;
		y = p / pnx;

		xe = (-E_size_x / 2.0) + (pp_ex * x);
		ye = (E_size_y / 2.0 - pp_ey) - (pp_ey * y);

		double sval = PI / lambda / f_field_ * (xe*xe + ye*ye);
		Complex kernel(0, sval);
		exponent_complex(&kernel);

		hh_e[p] = hh_e_0[p] * kernel;

	}
	
	delete hh, hh_e_0;

}

void HologramGenerator::Reconstruction(Complex* hh_e)
{
	int pnx = SLM_pixel_number_xy_[0];
	int pny = SLM_pixel_number_xy_[1];
	double ppx = Pixel_pitch_xy_[0];
	double ppy = Pixel_pitch_xy_[1];
	double F_size_x = pnx*ppx;
	double F_size_y = pny*ppy;
	double lambda = params_.lambda;
	double pp_ex = lambda * f_field_ / F_size_x;
	double pp_ey = lambda * f_field_ / F_size_y;
	double E_size_x = pp_ex*pnx;
	double E_size_y = pp_ey*pny;

	Complex* hh_e_shift = new Complex[pnx*pny];
	Complex* hh_e_ = new Complex[pnx*pny];

	int eye_shift_by_pnx = round(eye_center_xy_[0] / pp_ex);
	int eye_shift_by_pny = round(eye_center_xy_[1] / pp_ey);
	circshift(hh_e, hh_e_shift, -eye_shift_by_pnx, eye_shift_by_pny, pnx, pny);
		
	double f_eye = eye_length_*(f_field_ - focus_distance_) / (eye_length_ + (f_field_ - focus_distance_));
	double effective_f = f_eye*eye_length_ / (f_eye - eye_length_);
	double xe, ye, x, y;
	int eye_lens_anti_aliasing_mask, eye_pupil_mask;
	for (int p = 0; p < pnx * pny; p++)
	{
		x = p % pnx;
		y = p / pnx;

		xe = (-E_size_x / 2.0) + (pp_ex * x);
		ye = (E_size_y / 2.0 - pp_ey) - (pp_ey * y);

		Complex eye_propagation_kernel(0, PI / lambda / effective_f * (xe*xe + ye*ye));
		exponent_complex(&eye_propagation_kernel);
		eye_lens_anti_aliasing_mask = ( sqrt(xe*xe+ye*ye) < abs(lambda*effective_f / (2.0 * max(pp_ex, pp_ey))) )?1:0;
		eye_pupil_mask = (sqrt(xe*xe+ye*ye) < (eye_pupil_diameter_/2.0))?1:0;

		hh_e_[p] = hh_e_shift[p] * eye_propagation_kernel * eye_lens_anti_aliasing_mask * eye_pupil_mask;

	}

	Complex* hh_retina_0 = new Complex[pnx*pny];
	memset(hh_retina_0, 0.0, sizeof(Complex)*pnx*pny);
	fftwShift(hh_e_, hh_retina_0, pnx, pny, 1, false);

	double pp_ret_x = lambda*eye_length_/ E_size_x;
	double pp_ret_y = lambda*eye_length_ / E_size_y;
	double Ret_size_x = pp_ret_x*pnx;
	double Ret_size_y = pp_ret_y*pny;

	double xr, yr;
	for (int p = 0; p < pnx * pny; p++)
	{
		x = p % pnx;
		y = p / pnx;

		xr = (-Ret_size_x / 2.0) + (pp_ret_x * x);
		yr = (Ret_size_y / 2.0 - pp_ret_y) - (pp_ret_y * y);

		double sval = PI/lambda/ eye_length_*(xr*xr + yr*yr);
		Complex kernel(0, sval);
		exponent_complex(&kernel);

		sim_final_[p] = (hh_retina_0[p] * kernel).mag();

	}

	delete hh_e_shift, hh_retina_0, hh_e_;

}
void HologramGenerator::Write_Simulation_image(int num, double val)
{
	QDir dir("./");
	if (!dir.exists(QString().fromStdString(RESULT_FOLDER)))
		dir.mkdir(QString().fromStdString(RESULT_FOLDER));

	QString fname = "./" + QString().fromStdString(RESULT_FOLDER) + "/"
		+ QString().fromStdString(Simulation_Result_File_Prefix_) + "_"
		+ QString().fromStdString(RESULT_PREFIX)
		+ QString().setNum(num) 
		+ QString("_") + (sim_type_==0?"FOCUS_":"EYE_Y_") + QString().setNum(round(val*1000))
		+ ".bmp";

	int pnx = params_.pn[0];
	int pny = params_.pn[1];
	int px = pnx / 3;
	int py = pny;

	double min_val, max_val;
	min_val = sim_final_[0];
	max_val = sim_final_[0];
	for (int i = 0; i < pnx*pny; ++i)
	{
		if (min_val > sim_final_[i])
			min_val = sim_final_[i];
		else if (max_val < sim_final_[i])
			max_val = sim_final_[i];
	}

	uchar* data = new uchar[pnx*pny];
	memset(data, 0, sizeof(uchar)*pnx*pny);
	for (int k = 0; k < pnx*pny; k++)
		data[k] = (uint)((sim_final_[k] - min_val) / (max_val - min_val) * 255);

	QImage img(data, px, py, QImage::Format::Format_RGB888);
	img.save(QString(fname));

}
void HologramGenerator::circshift(Complex* in, Complex* out, int shift_x, int shift_y, int nx, int ny)
{
	int ti, tj;
	for (int i = 0; i < nx; i++)
	{
		for (int j = 0; j < ny; j++)
		{
			ti = (i + shift_x) % nx;
			if (ti < 0)
				ti = ti + nx;
			tj = (j + shift_y) % ny;
			if (tj < 0)
				tj = tj + ny;

			out[ti + tj * nx] = in[i + j * nx];
		}
	}
}

//U = ifftshift(fft2(fftshift(u))); or 	//u = ifftshift(ifft2(fftshift(U)));
// type == 1 FFTW_FORWARD (default), type == -1 FFTW_BACKWARD
void HologramGenerator::fftwShift(Complex* src, Complex* dst, int nx, int ny, int type, bool bNomalized)
{
	Complex* tmp = new Complex[nx*ny];
	memset(tmp, 0.0, sizeof(Complex)*nx*ny);

	fftw_complex *in, *out;
	in = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * nx * ny);
	out = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * nx * ny);

	//writeIntensity_gray8_bmp("src", nx, ny, src);
	fftShift(nx, ny, src, tmp);
	//writeIntensity_gray8_bmp("fftshift", nx, ny, tmp);

	for (int i = 0; i < nx*ny; i++)
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
	if (bNomalized) normalF = nx * ny;
	memset(tmp, 0, sizeof(Complex)*nx*ny);
	for (int k = 0; k < nx*ny; k++) {
		tmp[k].a = out[k][0] / normalF;
		tmp[k].b = out[k][1] / normalF;
	}

	//writeIntensity_gray8_bmp("fft2", nx, ny, tmp);
	//writeIntensity_gray8_real_bmp("fft2_real", nx, ny, tmp);

	fftShift(nx, ny, tmp, dst);

	//writeIntensity_gray8_bmp("ifftshift", nx, ny, dst);

	fftw_destroy_plan(fft_plan);
	fftw_free(in); fftw_free(out);

	delete tmp;
}
