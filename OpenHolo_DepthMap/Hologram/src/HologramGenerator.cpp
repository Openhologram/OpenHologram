
#include	"Hologram/HologramGenerator.h"
#include	<fstream>
#include	<sstream>
#include	<random>
#include    "graphics/sys.h"
#include	<QtScript/QScriptEngine>
#include	<QtCore/QRegularExpression>
//#include	"matlab/mat.h"

/** 
* @brief Constructor
* @details Initialize variables.
*/
HologramGenerator::HologramGenerator()
{
	isCPU_ = true;

	// GPU Variables
	img_src_gpu_ = 0;
	dimg_src_gpu_ = 0;
	depth_index_gpu_ = 0;

	// CPU Variables
	img_src_ = 0;
	dmap_src_ = 0;
	alpha_map_ = 0;
	depth_index_ = 0;
	dmap_ = 0;
	dstep_ = 0;
	dlevel_.clear();
	U_complex_ = 0;
	u255_fringe_ = 0;

}

/**
* @brief Destructor 
*/
HologramGenerator::~HologramGenerator()
{
}

/**
* @brief Set the value of a variable isCPU_(true or false)
* @details <pre>
    if isCPU_ == true
	   CPU implementation
	else
	   GPU implementation </pre>
* @param isCPU : the value for specifying whether the hologram generation method is implemented on the CPU or GPU
*/
void HologramGenerator::setMode(bool isCPU) 
{ 
	isCPU_ = isCPU; 
}

/**
* @brief Read parameters from a config file.
* @return true if config infomation are sucessfully read, flase otherwise.
*/
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

/**
* @brief Initialize variables for CPU and GPU implementation.
* @see init_CPU, init_GPU
*/
void HologramGenerator::initialize()
{
	dstep_ = 0;
	dlevel_.clear();

	if (u255_fringe_)		delete u255_fringe_;
	u255_fringe_ = new double[params_.pn[0] * params_.pn[1]];


	if (isCPU_)
		init_CPU();
	else
		init_GPU();	
}

/**
* @brief Generate a hologram, main funtion.
* @details For each frame, 
*    1. Read image depth data.
*    2. Compute the physical distance of depth map.
*    3. Transform target object to reflect the system configuration of holographic display.
*    4. Generate a hologram.
*    5. Encode the generated hologram.
*    6. Write the hologram to a image.
* .
* @see ReadImageDepth, GetDepthValues, TransformViewingWindow, Calc_Holo_by_Depth, Encoding_Symmetrization, Write_Result_image
*/
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

		if (!ReadImageDepth(ftr)) {
			LOG("Error: Reading image of frame %d.\n", ftr);
			continue;
		}

		GetDepthValues();

		if (Transform_Method_ == 0)
			TransformViewingWindow();

		Calc_Holo_by_Depth(ftr);
		
		if (Encoding_Method_ == 0)
			Encoding_Symmetrization(ivec2(0,1));

		Write_Result_image(ftr);

		//writeMatFileDouble("u255_fringe", u255_fringe_);
		//writeMatFileComplex("U_complex", U_complex_);
	}

}

/**
* @brief Read image and depth map.
* @details Read input files and load image & depth map data.
*  If the input image size is different with the dislay resolution, resize the image size.
* @param ftr : the frame number of the image.
* @return true if image data are sucessfully read, flase otherwise.
* @see prepare_inputdata_CPU, prepare_inputdata_GPU
*/
bool HologramGenerator::ReadImageDepth(int ftr)
{
	QString src_folder;
	if (FLAG_STATIC_IMAGE == 0)
	{
		if (NUMBER_OF_DIGIT_OF_FRAME_NUMBERING > 0)
			src_folder = QString().fromStdString(SOURCE_FOLDER) + "/" + QString("%1").arg((uint)(ftr + START_OF_FRAME_NUMBERING), (int)NUMBER_OF_DIGIT_OF_FRAME_NUMBERING, 10, (QChar)'0');
		else
			src_folder = QString().fromStdString(SOURCE_FOLDER) + "/" + QString().setNum(ftr + START_OF_FRAME_NUMBERING);

	}else 
		src_folder = QString().fromStdString(SOURCE_FOLDER);

	
	QString sdir = "./" + src_folder;
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
	LOG("Image Load: %s\n", imgfullname.toStdString().c_str());

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

	int ret;
	if (isCPU_)
		ret = prepare_inputdata_CPU(img.bits(), dimg.bits());
	else
		ret = prepare_inputdata_GPU(img.bits(), dimg.bits());

	//writeIntensity_gray8_bmp("test.bmp", pnx, pny, dmap_src_);
	//writeIntensity_gray8_bmp("test2.bmp", pnx, pny, dmap_);
	//dimg.save("test_dmap.bmp");
	//img.save("test_img.bmp");

	return ret;

}

/**
* @brief Calculate the physical distances of depth map layers
* @details Initialize 'dstep_' & 'dlevel_' variables.
*  If FLAG_CHANGE_DEPTH_QUANTIZATION == 1, recalculate  'depth_index_' variable.
* @see change_depth_quan_CPU, change_depth_quan_GPU
*/
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
		if (isCPU_)
			change_depth_quan_CPU();
		else
			change_depth_quan_GPU();
	}
}

/**
* @brief Transform target object to reflect the system configuration of holographic display.
* @details Calculate 'dlevel_transform_' variable by using 'field_lens' & 'dlevel_'.
*/
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

/**
* @brief Generate a hologram.
* @param frame : the frame number of the image.
* @see Calc_Holo_CPU, Calc_Holo_GPU
*/
void HologramGenerator::Calc_Holo_by_Depth(int frame)
{
	if (isCPU_)
		Calc_Holo_CPU(frame);
	else
		Calc_Holo_GPU(frame);
	
}

/**
* @brief Assign random phase value if RANDOM_PHASE == 1
* @details If RANDOM_PHASE == 1, calculate a random phase value using random generator;
*  otherwise, random phase value is 1.
* @param rand_phase_val : Input & Ouput value.
*/
void HologramGenerator::get_rand_phase_value(Complex& rand_phase_val)
{
	if (RANDOM_PHASE > 0)
	{
		std::default_random_engine generator;
		std::uniform_real_distribution<double> distribution(0.0, 1.0);

		rand_phase_val.a = 0.0;
		rand_phase_val.b = 2 * PI * distribution(generator);
		exponent_complex(&rand_phase_val);

	} else {
		rand_phase_val.a = 1.0;
		rand_phase_val.b = 0.0;
	}

}

/**
* @brief Encode the CGH according to a signal location parameter.
* @param sig_location : ivec2 type, 
*  sig_location[0]: upper or lower half, sig_location[1]:left or right half.
* @see encoding_CPU, encoding_GPU
*/
void HologramGenerator::Encoding_Symmetrization(ivec2 sig_location)
{
	int pnx = params_.pn[0];
	int pny = params_.pn[1];

	int cropx1, cropx2, cropx, cropy1, cropy2, cropy;
	if (sig_location[1] == 0) //Left or right half
	{
		cropy1 = 1;
		cropy2 = pny;

	} else {

		cropy = floor(pny / 2);
		cropy1 = cropy - floor(cropy / 2);
		cropy2 = cropy1 + cropy - 1;
	}

	if (sig_location[0] == 0) // Upper or lower half
	{
		cropx1 = 1;
		cropx2 = pnx;

	} else {

		cropx = floor(pnx / 2);
		cropx1 = cropx - floor(cropx / 2);
		cropx2 = cropx1 + cropx - 1;
	}

	cropx1 -= 1;
	cropx2 -= 1;
	cropy1 -= 1;
	cropy2 -= 1;

	if (isCPU_)
		encoding_CPU(cropx1, cropx2, cropy1, cropy2, sig_location);
	else
		encoding_GPU(cropx1, cropx2, cropy1, cropy2, sig_location);


}

/**
* @brief Write the result image.
* @param ftr : the frame number of the image.
*/
void HologramGenerator::Write_Result_image(int ftr)
{
	QDir dir("./");
	if (!dir.exists(QString().fromStdString(RESULT_FOLDER)))
		dir.mkdir(QString().fromStdString(RESULT_FOLDER));

	QString fname = "./" + QString().fromStdString(RESULT_FOLDER) + "/"
		+ QString().fromStdString(RESULT_PREFIX) + QString().setNum(ftr) + ".bmp";

	int pnx = params_.pn[0];
	int pny = params_.pn[1];
	int px = pnx / 3;
	int py = pny;

	double min_val, max_val;
	min_val = u255_fringe_[0];
	max_val = u255_fringe_[0];
	for (int i = 0; i < pnx*pny; ++i)
	{
		if (min_val > u255_fringe_[i])
			min_val = u255_fringe_[i];
		else if (max_val < u255_fringe_[i])
			max_val = u255_fringe_[i];
	}

	uchar* data = new uchar[pnx*pny];
	memset(data, 0, sizeof(uchar)*pnx*pny);

	int x = 0;
#pragma omp parallel for private(x)	
	for (x = 0; x < pnx*pny; ++x)
		data[x] = (uint)((u255_fringe_[x] - min_val) / (max_val - min_val) * 255);
	
	QImage img(data, px, py, QImage::Format::Format_RGB888);
	img.save(QString(fname));

}

/*
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
		intensity[i] = complexvalue[i].a;
		//intensity[i] = complexvalue[i].mag2();

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
*/

//============================================================================================
/**
* @brief It is a testing function used for the reconstruction.
*/
void HologramGenerator::ReconstructImage()
{
	if (!u255_fringe_) {
		//u255_fringe_ = new double[params_.pn[0] * params_.pn[1]];
		//if (!readMatFileDouble("u255_fringe.mat", u255_fringe_))
		LOG("Error: No Hologram Data\n");
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

/**
* @brief It is a testing function used for the reconstruction.
*/
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
	
	memset(hh, 0.0, sizeof(Complex)*pnx*pny);

	int p;
#pragma omp parallel for private(p)
	for (p = 0; p < pnx * pny; p++)
	{
		double x = p % pnx;
		double y = p / pnx;

		double xe = (-E_size_x / 2.0) + (pp_ex * x);
		double ye = (E_size_y / 2.0 - pp_ey) - (pp_ey * y);

		double sval = PI / lambda / f_field_ * (xe*xe + ye*ye);
		Complex kernel(0, sval);
		exponent_complex(&kernel);

		hh_e[p] = hh_e_0[p] * kernel;

	}
	
	delete hh, hh_e_0;

}

/**
* @brief It is a testing function used for the reconstruction.
*/
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

	int p;
#pragma omp parallel for private(p)
	for (p = 0; p < pnx * pny; p++)
	{
		double x = p % pnx;
		double y = p / pnx;

		double xe = (-E_size_x / 2.0) + (pp_ex * x);
		double ye = (E_size_y / 2.0 - pp_ey) - (pp_ey * y);

		Complex eye_propagation_kernel(0, PI / lambda / effective_f * (xe*xe + ye*ye));
		exponent_complex(&eye_propagation_kernel);
		int eye_lens_anti_aliasing_mask = ( sqrt(xe*xe+ye*ye) < abs(lambda*effective_f / (2.0 * max(pp_ex, pp_ey))) )?1:0;
		int eye_pupil_mask = (sqrt(xe*xe+ye*ye) < (eye_pupil_diameter_/2.0))?1:0;

		hh_e_[p] = hh_e_shift[p] * eye_propagation_kernel * eye_lens_anti_aliasing_mask * eye_pupil_mask;

	}

	Complex* hh_retina_0 = new Complex[pnx*pny];
	memset(hh_retina_0, 0.0, sizeof(Complex)*pnx*pny);
	fftwShift(hh_e_, hh_retina_0, pnx, pny, 1, false);

	double pp_ret_x = lambda*eye_length_/ E_size_x;
	double pp_ret_y = lambda*eye_length_ / E_size_y;
	double Ret_size_x = pp_ret_x*pnx;
	double Ret_size_y = pp_ret_y*pny;

#pragma omp parallel for private(p)
	for (p = 0; p < pnx * pny; p++)
	{
		double x = p % pnx;
		double y = p / pnx;

		double xr = (-Ret_size_x / 2.0) + (pp_ret_x * x);
		double yr = (Ret_size_y / 2.0 - pp_ret_y) - (pp_ret_y * y);

		double sval = PI/lambda/ eye_length_*(xr*xr + yr*yr);
		Complex kernel(0, sval);
		exponent_complex(&kernel);

		sim_final_[p] = (hh_retina_0[p] * kernel).mag();

	}

	delete hh_e_shift, hh_retina_0, hh_e_;

}

/**
* @brief It is a testing function used for the reconstruction.
*/
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

/**
* @brief It is a testing function used for the reconstruction.
*/
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


/*
bool HologramGenerator::readMatFileDouble(const char* fileName, double * val)
{
	MATFile *pmat;
	mxArray *parray;

	char fname[100];
	strcpy(fname, fileName);

	pmat = matOpen(fname, "r");
	if (pmat == NULL) {
		OG("Error creating file %s\n", fname);
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

void HologramGenerator::writeMatFileComplex(const char* fileName, Complex* val)
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
	status = matPutVariable(pmat, "data", pData);

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
*/
