
#include "..\cwo\src\cwo.h"
#include "math.h"
#include <fstream>
#include <iostream>
#include <sstream>
#include <vector>
#include <string>
#include "stdlib.h"
#include "stdio.h"
#include <conio.h>
#include "time.h"

using namespace std;
/**
* @brief OPHWRP is inheritance of CWO
* @detail this library need build in x64 environment and setting with CWO library
*/
class OPHWRP : public CWO
{
private:
	CWO c_tex;
	CWO c_dep;

public:
	cwoObjPoint* obj;
	int obj_num;
	vector<float> vec_xyz;


public:

	/**
	*@brief Generate WRP plane using Single WRP method
	*@param z: WRP location distance
	*/
	void SingleWRP(float z);
	/**
	*@brief Read point cloud (x,y,z) with txt file
	*@param filename: filename to read
	*/
	int readxyz(string filename);   //read point cloud

	/**
	*@brief Setting parameter for hologram generation
	*@param m_wavelength: wavelength of light
	*@param m_pixelpitch: pixel pitch of hologram and wrp plane
	*@param m_resulution: resolution of hologram and wrp plane
	*/
	int Setparameter(float m_wavelength, float m_pixelpitch, float m_resolution); //set wrp parameter
	/**
	*@brief Save Buffer to bitmap
	*@param filename: filename to save
	*/
	int SavetoImage(char* filename);  //save WRP angular image
	/**
	*@brief Fresnel convolution propagation
	*@param z: fresnel propagation distance
	*/
	int Fresnelpropagation(float z);  //fresnel convolution method
};


