#pragma once

#define N  1024			// ÇØ»óµµ
// rLamda => wavelength
#define R_LAMDA		0.000000633 

// fringe pattern resolution
#define CGH_WIDTH		1024
#define CGH_HEIGHT		1024
#define DEFAULT_DEPTH	0.5
#define CGH_SCALE		0.0104

// point cloud data file 
#define INFILE		"./in.dat"
#define OUTFILE_RE		"./point9_twoPlane_fringe_re.bmp"
#define OUTFILE_IM		"./point9_twoPlane_fringe_im.bmp"


typedef struct _pointCloud
{
	int index;			// index
	float x;				// x coordinate
	float y;				// y coordinate
	float z;				// z coordinate
	float amp;			// amplitude
	float phase;			// phase
}PCLOUD;

typedef struct _specifications
{
	float pcScaleX;			// Scaling factor of x coordinate of point cloud
	float pcScaleY;			// Scaling factor of y coordinate of point cloud
	float pcScaleZ;			// Scaling factor of z coordinate of point cloud
	float offsetDepth;		// Offset value of point cloud in z direction
	float sPitchX;			// Pixel pitch of SLM in x direction
	float sPitchY;			// Pixel pitch of SLM in y direction
	int sNumX;				// Number of pixel of SLM in x direction
	int sNumY;				// Number of pixel of SLM in y direction
	char* filterShape;		// Shape of spatial bandpass filter ("Circle" or "Rect" for now)
	float wFilterX;			// Width of spatial bandpass filter in x direction (For "Circle," only this is used)
	float wFilterY;			// Width of spatial bandpass filter in y direction
	float fIn;			// Focal length of input lens of Telecentric
	float fOut;			// Focal length of output lens of Telecentric
	float fEye;			// Focal length of eyepiece lens
	float lambda;		// Wavelength of laser
	float tiltAngleX;	// Tilt angle in x direction for spatial filtering
	float tiltAngleY;	// Tilt angle in y direction for spatial filtering
}SPEC;

int creatBitmapFile(unsigned char* pixelbuffer, int pic_width, int pic_height, char* file_name);

// UTILITY FUCTION
void LOG(char* logmsg);
char* getTimeString();
void get_yymmdd_y2(char buf[]);
void get_hhmmss(char buf[]);