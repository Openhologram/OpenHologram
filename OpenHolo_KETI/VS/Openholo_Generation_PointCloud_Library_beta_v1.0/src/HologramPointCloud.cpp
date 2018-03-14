#define Hologram_EXPORTS

#include "HologramPointCloud.h"


openholo::PointCloud::PointCloud(const float x, const float y, const float z) {
	this->x = x;
	this->y = y;
	this->z = z;
	this->amplitude = 0.f;
	this->phase = 0.f;
}


openholo::PointCloud::PointCloud(const float x, const float y, const float z, const float amp, const float phase) {
	this->x = x;
	this->y = y;
	this->z = z;
	this->amplitude = amp;
	this->phase = phase;
}


openholo::ConfigParams::ConfigParams(const std::string InputConfigFile) {
	std::ifstream File(InputConfigFile, std::ios::in);
	if (!File.is_open()) {
		std::cerr << "OpenHolo Error : Failed to load Config Specification Data File(*.config)" << std::endl;
		File.close();
		return;
	}

	std::vector<std::string> Title;
	std::vector<std::string> Value;
	std::string Line;
	std::stringstream LineStream;

	int i = 0;
	while (std::getline(File, Line)) {
		std::string _Title;
		std::string _Value;
		std::string _Equal; // " = "
		LineStream << Line;
		LineStream >> _Title >> _Equal >> _Value;
		LineStream.clear();

		Title.push_back(_Title);
		Value.push_back(_Value);
		++i;
	}

	if (i != 17) {
		std::cerr << "OpenHolo Error : Failed to load Config Specification Data File(*.config)" << std::endl;
		File.close();
		return;
	}

	this->pointCloudScaleX = atof(Value[0].c_str());
	this->pointCloudScaleY = atof(Value[1].c_str());
	this->pointCloudScaleZ = atof(Value[2].c_str());
	this->offsetDepth = atof(Value[3].c_str());
	this->samplingPitchX = atof(Value[4].c_str());
	this->samplingPitchY = atof(Value[5].c_str());
	this->nx = atoi(Value[6].c_str());
	this->ny = atoi(Value[7].c_str());
	this->filterShapeFlag = (char*)Value[8].c_str();
	this->filterXwidth = atof(Value[9].c_str());
	this->filterYwidth = atof(Value[10].c_str());
	this->focalLengthLensIn = atof(Value[11].c_str());
	this->focalLengthLensOut = atof(Value[12].c_str());
	this->focalLengthLenEyepiece = atof(Value[13].c_str());
	this->lambda = atof(Value[14].c_str());
	this->tiltAngleX = atof(Value[15].c_str());
	this->tiltAngleY = atof(Value[16].c_str());
	File.close();
}


openholo::HologramPointCloud::HologramPointCloud() {
	setMode(false);
	this->InputSrcFile = "";
	this->InputConfigFile = "";
	this->n_points = -1;
	this->data_hologram = nullptr;
}


openholo::HologramPointCloud::HologramPointCloud(std::string InputModelFile, std::string InputConfigFile) {
	setMode(false);
	this->InputSrcFile = InputModelFile;
	this->n_points = loadPointCloud(InputModelFile);
	if (n_points == -1) std::cerr << "OpenHolo Error : Failed to load Point Cloud Data File(*.dat)" << std::endl;

	this->InputConfigFile = InputConfigFile;
	bool ok = readConfig(InputConfigFile);
	if (!ok) std::cerr << "OpenHolo Error : Failed to load Config Specification Data File(*.config)" << std::endl;

	this->data_hologram = nullptr;
}


openholo::HologramPointCloud::~HologramPointCloud() {
	this->clear();
}


void openholo::HologramPointCloud::setMode(bool isCPU) {
	this->bIsCPU = isCPU;
}

bool openholo::HologramPointCloud::getMode() {
	return this->bIsCPU;
}


void openholo::HologramPointCloud::clear() {
	this->InputSrcFile.clear();
	this->InputConfigFile.clear();

	//this->ModelData.clear();
	this->VertexArray.clear();
	this->AmplitudeArray.clear();
	this->PhaseArray.clear();

	if (this->data_hologram != nullptr) free(data_hologram);
}


int openholo::HologramPointCloud::loadPointCloud(const std::string InputModelFile) {
	std::ifstream File(InputModelFile, std::ios::in);
	if (!File.is_open()) {
		File.close();
		return -1;
	}

	std::string Line;
	std::getline(File, Line);
	int n_pts = atoi(Line.c_str());
	this->n_points = n_pts;

	// parse input point cloud file
	for (int i = 0; i < n_pts; ++i) {
		int idx;
		float pX, pY, pZ, phase, amplitude;
		std::getline(File, Line);
		sscanf(Line.c_str(), "%d %f %f %f %f %f\n", &idx, &pX, &pY, &pZ, &phase, &amplitude);

		if (idx == i) {
			this->VertexArray.push_back(pX);
			this->VertexArray.push_back(pY);
			this->VertexArray.push_back(pZ);
			this->PhaseArray.push_back(phase);
			this->AmplitudeArray.push_back(amplitude);
			//this->ModelData.push_back(PointCloud(pX, pY, pZ, amplitude, phase));
		}
		else {
			File.close();
			return -1;
		}
	}
	File.close();
	this->InputSrcFile = InputModelFile;
	return n_pts;
}


bool openholo::HologramPointCloud::readConfig(const std::string InputConfigFile) {
	std::ifstream File(InputConfigFile, std::ios::in);
	if (!File.is_open()) {
		File.close();
		return false;
	}

	std::vector<std::string> Title;
	std::vector<std::string> Value;
	std::string Line;
	std::stringstream LineStream;

	int i = 0;
	while (std::getline(File, Line)) {
		std::string _Title;
		std::string _Value;
		std::string _Equal; // " = "
		LineStream << Line;
		LineStream >> _Title >> _Equal >> _Value;
		LineStream.clear();

		Title.push_back(_Title);
		Value.push_back(_Value);
		++i;
	}

	if (i != 17) {
		File.close();
		return false;
	}

	this->ConfigParams.pointCloudScaleX = atof(Value[0].c_str());
	this->ConfigParams.pointCloudScaleY = atof(Value[1].c_str());
	this->ConfigParams.pointCloudScaleZ = atof(Value[2].c_str());
	this->ConfigParams.offsetDepth = atof(Value[3].c_str());
	this->ConfigParams.samplingPitchX = atof(Value[4].c_str());
	this->ConfigParams.samplingPitchY = atof(Value[5].c_str());
	this->ConfigParams.nx = atoi(Value[6].c_str());
	this->ConfigParams.ny = atoi(Value[7].c_str());
	this->ConfigParams.filterShapeFlag = (char*)Value[8].c_str();
	this->ConfigParams.filterXwidth = atof(Value[9].c_str());
	this->ConfigParams.filterYwidth = atof(Value[10].c_str());
	this->ConfigParams.focalLengthLensIn = atof(Value[11].c_str());
	this->ConfigParams.focalLengthLensOut = atof(Value[12].c_str());
	this->ConfigParams.focalLengthLenEyepiece = atof(Value[13].c_str());
	this->ConfigParams.lambda = atof(Value[14].c_str());
	this->ConfigParams.tiltAngleX = atof(Value[15].c_str());
	this->ConfigParams.tiltAngleY = atof(Value[16].c_str());
	File.close();
	this->InputConfigFile = InputConfigFile;
	return true;
}


void openholo::HologramPointCloud::setPointCloudModel(const std::vector<float> &VertexArray, const std::vector<float> &AmplitudeArray, const std::vector<float> &PhaseArray) {
	this->VertexArray = VertexArray;
	this->AmplitudeArray = AmplitudeArray;
	this->PhaseArray = PhaseArray;
}

void openholo::HologramPointCloud::getPointCloudModel(std::vector<float> &VertexArray, std::vector<float> &AmplitudeArray, std::vector<float> &PhaseArray) {
	getModelVertexArray(VertexArray);
	getModelAmplitudeArray(AmplitudeArray);
	getModelPhaseArray(PhaseArray);
}

/*
void openholo::HologramPointCloud::setPointCloudModel(const std::vector<PointCloud> &Model) {
this->ModelData = Model;
}

void openholo::HologramPointCloud::getPointCloudModel(std::vector<PointCloud> &Model) {
Model = this->ModelData;
}
*/

void openholo::HologramPointCloud::getModelVertexArray(std::vector<float> &VertexArray) {
	VertexArray = this->VertexArray;
}

void openholo::HologramPointCloud::getModelAmplitudeArray(std::vector<float> &AmplitudeArray) {
	AmplitudeArray = this->AmplitudeArray;
}

void openholo::HologramPointCloud::getModelPhaseArray(std::vector<float> &PhaseArray) {
	PhaseArray = this->PhaseArray;
}

int openholo::HologramPointCloud::getNumberOfPoints() {
	return this->n_points;
}

uchar* openholo::HologramPointCloud::getHologramBufferData() {
	return this->data_hologram;
}

void openholo::HologramPointCloud::setConfigParams(const SpecParams &InputConfig) {
	this->ConfigParams = InputConfig;
}

openholo::SpecParams openholo::HologramPointCloud::getConfigParams() {
	return this->ConfigParams;
}

void openholo::HologramPointCloud::setScaleFactor(const float scaleX, const float scaleY, const float scaleZ) {
	this->ConfigParams.pointCloudScaleX = scaleX;
	this->ConfigParams.pointCloudScaleY = scaleY;
	this->ConfigParams.pointCloudScaleZ = scaleZ;
}

void openholo::HologramPointCloud::getScaleFactor(float &scaleX, float &scaleY, float &scaleZ) {
	scaleX = this->ConfigParams.pointCloudScaleX;
	scaleY = this->ConfigParams.pointCloudScaleY;
	scaleZ = this->ConfigParams.pointCloudScaleZ;
}

void openholo::HologramPointCloud::setOffsetDepth(const float offsetDepth) {
	this->ConfigParams.offsetDepth = offsetDepth;
}

float openholo::HologramPointCloud::getOffsetDepth() {
	return this->ConfigParams.offsetDepth;
}

void openholo::HologramPointCloud::setSamplingPitch(const float pitchX, const float pitchY) {
	this->ConfigParams.samplingPitchX = pitchX;
	this->ConfigParams.samplingPitchY = pitchY;
}

void openholo::HologramPointCloud::getSamplingPitch(float &pitchX, float &pitchY) {
	pitchX = this->ConfigParams.samplingPitchX;
	pitchY = this->ConfigParams.samplingPitchY;
}

void openholo::HologramPointCloud::setImageSize(const int n_x, const int n_y) {
	this->ConfigParams.nx = n_x;
	this->ConfigParams.ny = n_y;
}

void openholo::HologramPointCloud::getImageSize(int &n_x, int &n_y) {
	n_x = this->ConfigParams.nx;
	n_y = this->ConfigParams.ny;
}

void openholo::HologramPointCloud::setWaveLength(const float lambda) {
	this->ConfigParams.lambda = lambda;
}

float openholo::HologramPointCloud::getWaveLength() {
	return this->ConfigParams.lambda;
}

void openholo::HologramPointCloud::setTiltAngle(const float tiltAngleX, const float tiltAngleY) {
	this->ConfigParams.tiltAngleX = tiltAngleX;
	this->ConfigParams.tiltAngleY = tiltAngleY;
}

void openholo::HologramPointCloud::getTiltAngle(float &tiltAngleX, float &tiltAngleY) {
	tiltAngleX = this->ConfigParams.tiltAngleX;
	tiltAngleY = this->ConfigParams.tiltAngleY;
}


double openholo::HologramPointCloud::generateHologram() {
	// Output Image Size
	int n_x = ConfigParams.nx;
	int n_y = ConfigParams.ny;

	// Memory Location for Result Image
	if (this->data_hologram != nullptr) free(data_hologram);
	this->data_hologram = (uchar*)calloc(1, sizeof(uchar)*n_x*n_y);
	float *data_fringe = (float*)calloc(1, sizeof(float)*n_x*n_y);

	// Create CGH Fringe Pattern by 3D Point Cloud
	double time = 0.0;
	if (this->bIsCPU == true) { //Run CPU
#ifdef _OPENMP
		std::cout << "Generate Hologram with Multi Core CPU" << std::endl;
#else
		std::cout << "Generate Hologram with Single Core CPU" << std::endl;
#endif
		time = genCghPointCloud(this->VertexArray, this->AmplitudeArray, data_fringe);
	}
	else { //Run GPU
		std::cout << "Generate Hologram with GPU" << std::endl;

		time = genCghPointCloud_cuda(this->VertexArray, this->AmplitudeArray, data_fringe);
		std::cout << ">>> CUDA GPGPU" << std::endl;
	}

	// Normalization data_fringe to data_hologram
	normalize(data_fringe, this->data_hologram, n_x, n_y);
	free(data_fringe);
	return time;
}


/*
double openholo::HologramPointCloud::genCghPointCloud(const std::vector<PointCloud> &Model, float *dst) {
// Output Image Size
int n_x = ConfigParams.nx;
int n_y = ConfigParams.ny;

// Tilt Angle
float thetaX = RADIAN(ConfigParams.tiltAngleX);
float thetaY = RADIAN(ConfigParams.tiltAngleY);

// Wave Number
float k = (2.f * M_PI) / ConfigParams.lambda;

// Pixel pitch at eyepiece lens plane (by simple magnification) ==> SLM pitch
float pixel_x = ConfigParams.samplingPitchX;
float pixel_y = ConfigParams.samplingPitchY;

// Length (Width) of complex field at eyepiece plane (by simple magnification)
float Length_x = pixel_x * n_x;
float Length_y = pixel_y * n_y;

std::chrono::system_clock::time_point time_start = std::chrono::system_clock::now();
int j; // private variable for Multi Threading
#ifdef _OPENMP
int num_threads = 0;
#pragma omp parallel
{
num_threads = omp_get_num_threads(); // get number of Multi Threading
#pragma omp for private(j)
#endif
for (j = 0; j < this->n_points; ++j) { //Create Fringe Pattern
float x = Model[j].x * ConfigParams.pointCloudScaleX;
float y = Model[j].y * ConfigParams.pointCloudScaleY;
float z = Model[j].z * ConfigParams.pointCloudScaleZ + ConfigParams.offsetDepth;
float amplitude = Model[j].amplitude;

for (int row = 0; row < n_y; ++row) {
// Y coordinate of the current pixel : Note that y index is reversed order
float SLM_y = (Length_y / 2) - ((float)row + 0.5f) * pixel_y;

for (int col = 0; col < n_x; ++col) {
// X coordinate of the current pixel
float SLM_x = ((float)col + 0.5f) * pixel_x - (Length_x / 2);

float r = sqrtf((SLM_x - x)*(SLM_x - x) + (SLM_y - y)*(SLM_y - y) + z * z);
float phi = k * r - k * SLM_x*sinf(thetaX) - k * SLM_y*sinf(thetaY); // Phase for printer
float result = amplitude * cosf(phi);

*(dst + col + row * n_x) += result; //R-S Integral
}
}
}
#ifdef _OPENMP
}
std::cout << ">>> All " << num_threads << " threads" << std::endl;
#endif
std::chrono::system_clock::time_point time_finish = std::chrono::system_clock::now();
return ((std::chrono::duration<double>)(time_finish - time_start)).count();
}
*/


double openholo::HologramPointCloud::genCghPointCloud(const std::vector<float> &VertexArray, const std::vector<float> &AmplitudeArray, float *dst) {
	// Output Image Size
	int n_x = ConfigParams.nx;
	int n_y = ConfigParams.ny;

	// Tilt Angle
	float thetaX = RADIAN(ConfigParams.tiltAngleX);
	float thetaY = RADIAN(ConfigParams.tiltAngleY);

	// Wave Number
	float k = (2.f * M_PI) / ConfigParams.lambda;

	// Pixel pitch at eyepiece lens plane (by simple magnification) ==> SLM pitch
	float pixel_x = ConfigParams.samplingPitchX;
	float pixel_y = ConfigParams.samplingPitchY;

	// Length (Width) of complex field at eyepiece plane (by simple magnification)
	float Length_x = pixel_x * n_x;
	float Length_y = pixel_y * n_y;

	std::chrono::system_clock::time_point time_start = std::chrono::system_clock::now();
	int j; // private variable for Multi Threading
#ifdef _OPENMP
	int num_threads = 0;
#pragma omp parallel
	{
		num_threads = omp_get_num_threads(); // get number of Multi Threading
#pragma omp for private(j)
#endif
		for (j = 0; j < this->n_points; ++j) { //Create Fringe Pattern
			float x = VertexArray[3 * j + 0] * ConfigParams.pointCloudScaleX;
			float y = VertexArray[3 * j + 1] * ConfigParams.pointCloudScaleY;
			float z = VertexArray[3 * j + 2] * ConfigParams.pointCloudScaleZ + ConfigParams.offsetDepth;
			float amplitude = AmplitudeArray[j];

			for (int row = 0; row < n_y; ++row) {
				// Y coordinate of the current pixel : Note that y index is reversed order
				float SLM_y = (Length_y / 2) - ((float)row + 0.5f) * pixel_y;

				for (int col = 0; col < n_x; ++col) {
					// X coordinate of the current pixel
					float SLM_x = ((float)col + 0.5f) * pixel_x - (Length_x / 2);

					float r = sqrtf((SLM_x - x)*(SLM_x - x) + (SLM_y - y)*(SLM_y - y) + z * z);
					float phi = k * r - k * SLM_x*sinf(thetaX) - k * SLM_y*sinf(thetaY); // Phase for printer
					float result = amplitude * cosf(phi);

					*(dst + col + row * n_x) += result; //R-S Integral
				}
			}
		}
#ifdef _OPENMP
	}
	std::cout << ">>> All " << num_threads << " threads" << std::endl;
#endif
	std::chrono::system_clock::time_point time_finish = std::chrono::system_clock::now();
	return ((std::chrono::duration<double>)(time_finish - time_start)).count();
}


//Convert Angle
#define	CUDA_RADIAN(theta) (theta*CUDART_PI_F)/180.0 //convert degree to radian angle for CUDA
#define	CUDA_DEGREE(theta) (theta*180.0)/CUDART_PI_F //convert radian to degree angle for CUDA


// for PointCloud
typedef struct KernelConst {
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
} GpuConst;


extern "C"
{
	void cudaPointCloudKernel(const int block_x, const int block_y, const int thread_x, const int thread_y, float3 *PointCloud, float *amplitude, const GpuConst *Config, float *dst);
}


double openholo::HologramPointCloud::genCghPointCloud_cuda(const std::vector<float> &VertexArray, const std::vector<float> &AmplitudeArray, float *dst) {
	int _bx = ConfigParams.nx / THREAD_X;
	int _by = ConfigParams.ny / THREAD_Y;
	int block_x = 2;
	int block_y = 2;

	//blocks number
	while (1) {
		if ((block_x >= _bx) && (block_y >= _by)) break;
		if (block_x < _bx) block_x *= 2;
		if (block_y < _by) block_y *= 2;
	}

	//threads number
	const ulonglong bufferSize = ConfigParams.nx * ConfigParams.ny * sizeof(float);

	//Host Memory Location
	float3 *HostPointCloud = (float3*)VertexArray.data();
	float *hostAmplitude = (float*)AmplitudeArray.data();

	//Initializa Config for CUDA Kernel
	GpuConst HostConfig; {
		HostConfig.n_points = this->n_points;
		HostConfig.scaleX = ConfigParams.pointCloudScaleX;
		HostConfig.scaleY = ConfigParams.pointCloudScaleY;
		HostConfig.scaleZ = ConfigParams.pointCloudScaleZ;
		HostConfig.offsetDepth = ConfigParams.offsetDepth;

		// Output Image Size
		HostConfig.Nx = ConfigParams.nx;
		HostConfig.Ny = ConfigParams.ny;

		// Tilt Angle
		float thetaX = CUDA_RADIAN(ConfigParams.tiltAngleX);
		float thetaY = CUDA_RADIAN(ConfigParams.tiltAngleY);
		HostConfig.sin_thetaX = sinf(thetaX);
		HostConfig.sin_thetaY = sinf(thetaY);

		// Wave Number
		HostConfig.k = (2.f * CUDART_PI_F) / ConfigParams.lambda;

		// Pixel pitch at eyepiece lens plane (by simple magnification) ==> SLM pitch
		HostConfig.pixel_x = ConfigParams.samplingPitchX;
		HostConfig.pixel_y = ConfigParams.samplingPitchY;

		// Length (Width) of complex field at eyepiece plane (by simple magnification)
		float Length_x = HostConfig.pixel_x * HostConfig.Nx;
		float Length_y = HostConfig.pixel_y * HostConfig.Ny;
		HostConfig.halfLength_x = Length_x / 2.f;
		HostConfig.halfLength_y = Length_y / 2.f;
	}

	//Device(GPU) Memory Location
	float3 *DevicePointCloud;
	cudaMalloc((void**)&DevicePointCloud, VertexArray.size() * sizeof(float));
	cudaMemcpy(DevicePointCloud, HostPointCloud, VertexArray.size() * sizeof(float), cudaMemcpyHostToDevice);

	float *deviceAmplitude;
	cudaMalloc((void**)&deviceAmplitude, AmplitudeArray.size() * sizeof(float));
	cudaMemcpy(deviceAmplitude, hostAmplitude, AmplitudeArray.size() * sizeof(float), cudaMemcpyHostToDevice);

	GpuConst *DeviceConfig;
	cudaMalloc((void**)&DeviceConfig, sizeof(GpuConst));
	cudaMemcpy(DeviceConfig, &HostConfig, sizeof(HostConfig), cudaMemcpyHostToDevice);

	float *deviceDst;
	cudaMalloc((void**)&deviceDst, bufferSize);

	std::chrono::system_clock::time_point time_start = std::chrono::system_clock::now();
	{
		cudaPointCloudKernel(block_x, block_y, THREAD_X, THREAD_Y, DevicePointCloud, deviceAmplitude, DeviceConfig, deviceDst);
		cudaMemcpy(dst, deviceDst, bufferSize, cudaMemcpyDeviceToHost);
	}
	std::chrono::system_clock::time_point time_finish = std::chrono::system_clock::now();

	//Device(GPU) Memory Delete
	cudaFree(DevicePointCloud);
	cudaFree(deviceAmplitude);
	cudaFree(deviceDst);
	cudaFree(DeviceConfig);
	return ((std::chrono::duration<double>)(time_finish - time_start)).count();
}


/* Bitmap File */
#define OPH_Bitsperpixel 8 //24 // 3byte=24 
#define OPH_Planes 1
#define OPH_Compression 0
#define OPH_Xpixelpermeter 0x130B //2835 , 72 DPI
#define OPH_Ypixelpermeter 0x130B //2835 , 72 DPI
#define OPH_Pixel 0xFF


#pragma pack(push,1)
typedef struct fileheader {
	uint8_t signature[2];
	uint32_t filesize;
	uint32_t reserved;
	uint32_t fileoffset_to_pixelarray;
} fileheader;

typedef struct bitmapinfoheader {
	uint32_t dibheadersize;
	uint32_t width;
	uint32_t height;
	uint16_t planes;
	uint16_t bitsperpixel;
	uint32_t compression;
	uint32_t imagesize;
	uint32_t ypixelpermeter;
	uint32_t xpixelpermeter;
	uint32_t numcolorspallette;
	uint32_t mostimpcolor;
} bitmapinfoheader;

typedef struct rgbquad {
	uint8_t rgbBlue;
	uint8_t rgbGreen;
	uint8_t rgbRed;
	uint8_t rgbReserved;
} rgbquad;

typedef struct bitmap {
	fileheader fileheader;
	bitmapinfoheader bitmapinfoheader;
	rgbquad rgbquad[256]; // 8 bit 256 Color(Grayscale)
} bitmap;
#pragma pack(pop)


void openholo::HologramPointCloud::saveFileBmp(std::string OutputFileName) {
	int _height = this->ConfigParams.ny;
	int _width = this->ConfigParams.nx;
	int _pixelbytesize = _height * _width * OPH_Bitsperpixel / 8;
	int _filesize = _pixelbytesize + sizeof(bitmap);

	OutputFileName.append(".bmp");
	FILE *fp = fopen(OutputFileName.c_str(), "wb");
	bitmap *pbitmap = (bitmap*)calloc(1, sizeof(bitmap));
	memset(pbitmap, 0x00, sizeof(bitmap));

	// File Header
	pbitmap->fileheader.signature[0] = 'B';
	pbitmap->fileheader.signature[1] = 'M';
	pbitmap->fileheader.filesize = _filesize;
	pbitmap->fileheader.fileoffset_to_pixelarray = sizeof(bitmap);

	// Initialize pallets : to Grayscale
	for (int i = 0; i < 256; i++) {
		pbitmap->rgbquad[i].rgbBlue = i;
		pbitmap->rgbquad[i].rgbGreen = i;
		pbitmap->rgbquad[i].rgbRed = i;
	}

	// Image Header
	pbitmap->bitmapinfoheader.dibheadersize = sizeof(bitmapinfoheader);
	pbitmap->bitmapinfoheader.width = _width;
	pbitmap->bitmapinfoheader.height = _height;
	pbitmap->bitmapinfoheader.planes = OPH_Planes;
	pbitmap->bitmapinfoheader.bitsperpixel = OPH_Bitsperpixel;
	pbitmap->bitmapinfoheader.compression = OPH_Compression;
	pbitmap->bitmapinfoheader.imagesize = _pixelbytesize;
	pbitmap->bitmapinfoheader.ypixelpermeter = OPH_Ypixelpermeter;
	pbitmap->bitmapinfoheader.xpixelpermeter = OPH_Xpixelpermeter;
	pbitmap->bitmapinfoheader.numcolorspallette = 256;
	fwrite(pbitmap, 1, sizeof(bitmap), fp);
	fwrite(this->data_hologram, 1, _pixelbytesize, fp);
	fclose(fp);
	free(pbitmap);
}


void openholo::HologramPointCloud::normalize(float *src, uchar *dst, const int nx, const int ny) {
	float minVal, maxVal;
	for (int ydx = 0; ydx < ny; ydx++) {
		for (int xdx = 0; xdx < nx; xdx++) {
			float *temp_pos = src + xdx + ydx * nx;
			if ((xdx == 0) && (ydx == 0)) {
				minVal = *(temp_pos);
				maxVal = *(temp_pos);
			}
			else {
				if (*(temp_pos) < minVal) minVal = *(temp_pos);
				if (*(temp_pos) > maxVal) maxVal = *(temp_pos);
			}
		}
	}

	for (int ydx = 0; ydx < ny; ydx++) {
		for (int xdx = 0; xdx < nx; xdx++) {
			float *src_pos = src + xdx + ydx * nx;
			uchar *res_pos = dst + xdx + (ny - ydx - 1)*nx;	// Flip image vertically to consider flipping by Fourier transform and projection geometry

			*(res_pos) = (uchar)(((*(src_pos)-minVal) / (maxVal - minVal)) * 255 + 0.5);
		}
	}
}