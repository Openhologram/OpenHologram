#ifndef _CWOPLS_H
#define _CWOPLS_H
#endif

#include <cwo.h>
#include "math.h"
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <fstream>
#include <iostream>
#include "stdlib.h""
#include "stdio.h"
#include <conio.h>
#include "time.h"

using namespace cv;
using namespace std;

std::vector<std::string> split(const std::string& str_arg, const char token);

class OHWRP: public CWO
{
private:
	CWO c_tex;
	CWO c_dep;

public:
	void LoadTex(char *fname, int c = CWO_GREY);
	//void LoadDep(char *fname, int c = CWO_GREY);
	//void SavetoMat(CWO *holoimg, Mat);

	void SingleWRP(float z);
	void SingleWRP(float z, float delta_z, cwoInt2 range);
	//float* readtxt3d(string filename,const char token, int dim);

};

