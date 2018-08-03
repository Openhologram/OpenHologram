/** @mainpage
@brief Diffraction calculations, such as the angular spectrum method and Fresnel diffractions, 
are used for calculating scalar light propagation. 
The calculations are used in wide-ranging optics fields: for example, Computer Generated Holograms (CGHs), digital holography, diffractive optical elements, microscopy,
image encryption and decryption, three-dimensional analysis for optical devices and so on.@n
We develop a C++ class library for diffraction and CGH calculations, which is referred to as
a CWO++ library, running on a CPU and GPU.@n
Increasing demands made by large-scale diffraction calculations have rendered 
the computational power of recent computers insufficient.@n
@n

@code
//The following sample code is to calculate diffraction using the angular spectrum method.

CWO a;
a.Load("sample.bmp"); //Load image file 
a.Diffract(0.1, CWO_ANGULAR); //Calculate diffraction from the image using the angular spectrum method
a.Intensity(); //Calculate the light intensity from the diffracted result
a.Scale(255); // Convert the intensity to 255-steps data
a.Save("diffract.bmp"); //Save the intensity as bitmap file
@endcode

The following image (right) is the diffracted image.
@image html lena512x512_diffract.jpg

For more sample codes, please see example page.

@section Environment
- Windows XP or higher
- Cent OS 6.3
- If using NVIDIA GPU, you need to install CUDA 5.0

@section Installation
Dear users,
I corrected some bugs in the library.
And, this sample program includes simple diffraction, 4 steps phase shifting holography 
and GS algorithm, which are run on NVIDIA GPU and CUDA 4.0 (32 bit version).
If you want to run CWO++ library on a NVIDIA GPU, before running this program, you need to install CUDA 4.0 toolkit and SDK, which are 32 bit version. 
In order to run this program correctly, you also need to set the path for CUDA library and include files.

If you have problems, for example no finding "cudart32_40_17.dll" etc, you need to set the following 
library path to Windows's system variable "PATH" : 

C:\ProgramData\NVIDIA Corporation\NVIDIA GPU Computing SDK 4.0\C\common\bin
(This is default path to CUDA4.0 SDK's DLL files)


@section Reference
- T. Shimobaba, J. Weng, T. Sakurai, N. Okada, T. Nishitsuji, N. Takada, A. Shiraki, 
N. Masuda and T. Ito, "Computational wave optics library for C++: CWO++ library", 
Computer Physics Communications, 183, 1124-1138 (2012) @n

@section Copyright
@author Tomoyoshi Shimobaba
*/


#ifndef _CWO_H
#define _CWO_H


//###########################################
/** @defgroup fld_type Field-types
*/
//@{
/**
* @def CWO_FLD_COMPLEX
* This field-type indicates complex amplitude.
*/
#define CWO_FLD_COMPLEX		(0)
/**
* @def CWO_FLD_INTENSITY
* This field-type indicates the light intensity of complex amplitude.
*/
#define CWO_FLD_INTENSITY	(1)
/**
* @def CWO_FLD_PHASE
* This field-type indicates the argument of complex amplitude.
*/
#define CWO_FLD_PHASE		(2)
/**
* @def CWO_FLD_CHAR
* This field-type is now testing. Do not use it. 
*/
#define CWO_FLD_CHAR		(3)
//@}

//###########################################
/** @defgroup color_type Color-types
*/
//@{

/**
* @def CWO_RED
* This macro indicates red color. 
*/
#define CWO_RED		(0)
/**
* @def CWO_GREEN
* This macro indicates green color. 
*/
#define CWO_GREEN	(1)
/**
* @def CWO_BLUE
* This macro indicates blue color. 
*/
#define CWO_BLUE	(2)
/**
* @def CWO_GREY
* This macro indicates glay scale. 
*/
#define CWO_GREY	(3)
//@}


//###########################################
/** @defgroup inter Interpolation methods
*/
//@{
/**
* @def CWO_INTER_NEAREST
* This macro indicates nearrest neighbor interpolation. 
*/
#define CWO_INTER_NEAREST	(0)
/**
* @def CWO_INTER_LINEAR
* This macro indicates bi-linear interpolation. 
*/
#define CWO_INTER_LINEAR	(1)
/**
* @def CWO_INTER_CUBIC
* This macro indicates bi-cubic interpolation.
*/
#define CWO_INTER_CUBIC		(2)
//@}

//###########################################
/** @defgroup ed Error diffusion methods
*/
//@{

/**
* @def CWO_ED_FLOYD
* This macro indicates Floyd & Steinberg's error diffusion method
*/
#define CWO_ED_FLOYD	(0)

/**
* @def CWO_ED_JARVIS
* This macro indicates Jarvis, Judice & Ninke's error diffusion method
*/
#define CWO_ED_JARVIS	(1)
//@}

#define CWO_SAVE_AS_INTENSITY	(1)
#define CWO_SAVE_AS_PHASE		(2)
#define CWO_SAVE_AS_AMP			(3)
#define CWO_SAVE_AS_RE			(4)
#define CWO_SAVE_AS_IM			(5)
#define CWO_SAVE_AS_ARG			(6)

#define CWO_PROP_CENTER			(0)
#define CWO_PROP_FFT_SHIFT		(1)
#define CWO_PROP_MUL_CENTER		(2)
#define CWO_PROP_MUL_FFT_SHIFT	(3)

#define CWO_FILL_INSIDE		(0)
#define CWO_FILL_OUTSIDE	(1)



#define CWO_BUFFER_FIELD				(0)
#define CWO_BUFFER_DIFF_A				(1)
#define CWO_BUFFER_DIFF_B				(2)
#define CWO_BUFFER_D1					(3)
#define CWO_BUFFER_D2					(4)
#define CWO_BUFFER_X1					(5)
#define CWO_BUFFER_X2					(6)
#define CWO_BUFFER_PLS					(7)

#define CWO_SUCCESS	(0)
#define CWO_FAIL	(-1)


#include "cwo_lib.h"
#include "math.h"

//!  Top class. 
/*!
This class is top class of CWO++ library
*/
class CWO {
public:
	cwoCtx ctx;//context
	cwoCtx prev_ctx;

private:


public:
	//****************
	//For Diffraction
	//****************
	void *p_field;//complex amplitude field
	cwoComplex *p_diff_a;//temporary buffer for diffraction (convolution type)
	cwoComplex *p_diff_b;//temporary buffer for diffraction (convolution type)

	float *p_d_x1; //displacement from source plane
	float *p_d_x2; //displacement from destination plane
	cwoFloat2 *p_x1;
	cwoFloat2 *p_x2;

	float *fre_s_tbl;
	float *fre_c_tbl;
	
	//****************
	//For PLS
	//****************
	//int Npnt; // number of object point
	cwoObjPoint *p_pnt; //buffer for object point
	//unsigned char *p_tex; //buffer for texture
	//unsigned char *p_dep; //buffer for depth

	//****************
	//other
	//****************
	int field_type; //field type
	void InitParams(); /// Intialize parameters 
	void InitBuffers(); /// Initialze buffers
	void FreeBuffers(); /// Free buffers


public:
	virtual void* __Malloc(size_t size); 
	virtual void __Free(void **a);
	virtual void __Memcpy(void *dst, void *src, size_t size);
	virtual void __Memset(void *p, int c, size_t size);
	//virtual void __Memset(void *p, cwoComplex c, unsigned int size);

	virtual void __Expand(
		void *src, int sx, int sy, int srcNx, int srcNy, 
		void *dst, int dx, int dy, int dstNx, int dstNy,
		int type);
	

	virtual void __ShiftedFresnelAperture(cwoComplex *a);
	virtual void __ShiftedFresnelProp(cwoComplex *a);
	virtual void __ShiftedFresnelCoeff(cwoComplex *a);

	virtual void __ARSSFresnelAperture(cwoComplex *a);
	virtual void __ARSSFresnelProp(cwoComplex *a);
	virtual void __ARSSFresnelCoeff(cwoComplex *a);

	virtual void __FresnelConvProp(cwoComplex *a);
	virtual void __FresnelConvCoeff(cwoComplex *a, float const_val=1.0f);

	virtual void __AngularProp(cwoComplex *a, int flag);
	void __AngularLim(float *fx_c, float *fx_w, float *fy_c, float *fy_w);
//	virtual void __ShiftedAngularProp(cwoComplex *a);

	virtual void __HuygensProp(cwoComplex *a);

	virtual void __FresnelFourierProp(cwoComplex *a);
	virtual void __FresnelFourierCoeff(cwoComplex *a);

	virtual void __FresnelDblAperture(cwoComplex *a, float z1);
	virtual void __FresnelDblFourierDomain(cwoComplex *a, float z1, float z2, cwoInt4 *zp);
	virtual void __FresnelDblCoeff(cwoComplex *a, float z1, float z2);

	virtual void __FFT(void *src, void *dst, int type);
	virtual void __IFFT(void *src, void *dst);
	virtual void __FFTShift(void *src);

	void __Add(cwoComplex *a, float b, cwoComplex *c);//c=a+b
	virtual void __Add( cwoComplex *a,  cwoComplex b, cwoComplex *c);//c=a+b
	virtual void __Add( cwoComplex *a,  cwoComplex *b, cwoComplex *c);//c=a+b

	void __Sub( cwoComplex *a,  float b, cwoComplex *c);//c=a-b
	virtual void __Sub( cwoComplex *a,  cwoComplex b, cwoComplex *c);//c=a-b
	virtual void __Sub( cwoComplex *a,  cwoComplex *b, cwoComplex *c);//c=a-b

	void __Mul( cwoComplex *a,  float b, cwoComplex *c);//c=a*b
	virtual void __Mul( cwoComplex *a,  cwoComplex b, cwoComplex *c);//c=a*b
	virtual void __Mul( cwoComplex *a,  cwoComplex *b, cwoComplex *c);//c=a*b
	
	void __Div( cwoComplex *a,  float b, cwoComplex *c);//c=a/b
	virtual void __Div( cwoComplex *a,  cwoComplex b, cwoComplex *c);//c=a/b
	virtual void __Div( cwoComplex *a,  cwoComplex *b, cwoComplex *c);//c=a/b

	
	virtual void __Re(cwoComplex *a , cwoComplex *b);
	virtual void __Im(cwoComplex *a , cwoComplex *b);
	virtual void __Conj(cwoComplex *a);

	
	virtual void __Intensity(cwoComplex *a, cwoComplex *b); //OK
	virtual void __Amp(cwoComplex *a , cwoComplex *b);//OK
	virtual void __Phase(cwoComplex *a , cwoComplex *b, float offset);//OK
	virtual void __Arg(cwoComplex *a , cwoComplex *b, float offset);//OK
	virtual void __Real2Complex(float *src, cwoComplex *dst);
	virtual void __Phase2Complex(float *src, cwoComplex *dst);
	virtual void __Polar(float *amp, float *ph, cwoComplex *c);
	virtual void __ReIm(cwoComplex *re, cwoComplex *im, cwoComplex *c);
	virtual void __RectFillInside(cwoComplex *p, int x, int y, int Sx, int Sy, cwoComplex a);
	virtual void __RectFillOutside(cwoComplex *p, int x, int y, int Sx, int Sy, cwoComplex a);

	//virtual void __CopyFloat(float *src, int x1, int y1, int sNx, int sNy, float *dst, int x2, int y2, int dNx, int dNy, int Sx, int Sy);
	//virtual void __CopyComplex(cwoComplex *src, int x1, int y1, int sNx, int sNy, cwoComplex *dst, int x2, int y2, int dNx, int dNy, int Sx, int Sy);
	
	virtual void __FloatToChar(char *dst, float *src, int N);
	virtual void __CharToFloat(float *dst, char *src, int N);



	//virtual void __Gamma(float *src, float gamma);



	void SetFieldType(int type); 
	void SetSize(int Nx, int Ny, int Nz=1);
	void SetPropDist(float z);
	float GetPropDist();

public:
	
	//###########################################
	/** @defgroup const Constructors & Destructors
	*/
	//@{
	CWO();//!< constructor

	//! Constructor with size
    /*!
      \sa CWO() 
      \param Nx x size
      \param Ny y size
    */
	CWO(int Nx, int Ny, int Nz=1);//!< constructor
	virtual ~CWO();//!< destructor
	CWO(CWO &tmp);//!< copy constructor
	//@}

	//###########################################
	/** @defgroup operator Operators
	*/
	//@{
	CWO& operator=(CWO &tmp);	
	
	CWO operator+(CWO &a);
	CWO operator+(float a);
	CWO operator+(cwoComplex a);
	CWO operator-(CWO &a);
	CWO operator-(float a);
	CWO operator-(cwoComplex a);
	CWO operator*(CWO &a);
	CWO operator*(float a);
	CWO operator*(cwoComplex a);
	CWO operator/(CWO &a);
	CWO operator/(float a);
	CWO operator/(cwoComplex a);
	
	CWO& operator+=(CWO &a);	
	CWO& operator+=(float a);
	CWO& operator+=(cwoComplex a);
	CWO& operator-=(CWO &a);
	CWO& operator-=(float a);
	CWO& operator-=(cwoComplex a);
	CWO& operator*=(CWO &a);
	CWO& operator*=(float a);
	CWO& operator*=(cwoComplex a);
	CWO& operator/=(CWO &a);
	CWO& operator/=(float a);
	CWO& operator/=(cwoComplex a);

	//CWO operator+(CWO &a);
	//CWO operator+(float a);
	//CWO operator-(CWO &a);
	//CWO operator-(float a);
	//CWO operator*(CWO &a);
	//CWO operator*(float a);
	//CWO operator/(CWO &a);
	//CWO operator/(float a);
	
	//@}

	//###########################################
	/** @defgroup commun Communication
	*/
	//@{

	virtual void Send(CWO &a);
	virtual void Recv(CWO &a);
	
	//@}

	//!  Create complex amplitude with Nx x Ny size (in pixel unit)
    /*!
	@param Nx : x size (in pixel unit)
	@param Ny : y size (in pixel unit)
    */
	int Create(int Nx, int Ny, int Nz=1); 
	virtual void Destroy();
//	void ChangeSize(int Nx, int Ny);

	void Attach(CWO &a);

	//###########################################
	/** @defgroup time Time mesurement
	*/
	//@{

	//! Start time mesurement
    /*!
    */
	void SetTimer(); 

	//! End time mesurement
    /*!
	@return Elapsed time (in millisecond) between SetTimer() and EndTimer()
    */
	double EndTimer(); 

	//@}



	//###########################################
	/** @defgroup params Parameters
	*/
	//@{

	//! Set diffraction type. Do not use it.
    /*!
      \param type diffraction type (e.g. CWO_ANGULAR)
    */
	void SetCalcType(int type);
	
	//! Set the sampling rate on source and destination planes
    /*!
      \param px horizon sampling rate (in meter) 
	  \param py vertical sampling rate (in meter)
    */
	void SetPitch(float px, float py, float pz=10.0e-6f); 
	
	void SetPitch(float p); 
	
	//! Set the sampling rate on source plane only
    /*!
      \param px horizon sampling rate (in meter) 
	  \param py vertical sampling rate (in meter)
    */
	void SetSrcPitch(float px, float py, float pz=10.0e-6f); //!< Set sampling rate on source plane
	
	void SetSrcPitch(float p); 

	//! Set the sampling rate on destination plane only
    /*!
      \param px horizon sampling rate (in meter) 
	  \param py vertical sampling rate (in meter)
    */
	void SetDstPitch(float px, float py, float pz=10.0e-6f); //!< Set sampling rate on destination plane
	
	void SetDstPitch(float p);

	//! Set the wavelength of light
    /*!
      \param w Wavelength of light (in meter) 
	*/
	void SetWaveLength(float w); //!< Set wavelength 

	//! Set the offset of source plane away from the origin. 
    /*!
	@param x Horizonal offset (in meter) 
	@param y Vertical offset (in meter) 
	*/
	void SetOffset(float x, float y, float z=0.0f);
	//! Set the offset of source plane away from the origin.
	/*!
	@brief The offset affects off-axis diffraction calculations (e.g. Shifted Fresnel diffraction).
	@param x Horizonal offset (in meter) 
	@param y Vertical offset (in meter) 
	*/
	void SetSrcOffset(float x, float y, float z=0.0f); 
	//! Set the offset of destination plane away from the origin.
    /*!
	@brief The offset affects off-axis diffraction calculations (e.g. Shifted Fresnel diffraction).
	@param x Horizonal offset (in meter) 
	@param y Vertical offset (in meter) 
	*/	
	void SetDstOffset(float x, float y, float z=0.0f); 

	//! set the number of threads
    /*!
	@param Nx : number of threads 
	@param Ny : number of threads (GPU only)
    */
	void SetThreads(int Nx, int Ny=1);

	//! get the number of threads
    /*!
	@return number of threads
    */
	int GetThreads();
	//! get the number of threads along to x-axis
    /*!
	@return number of threads along to x-axis
    */
	int GetThreadsX();
	//! get the number of threads along to y-axis
    /*!
	@return number of threads along to y-axis
    */
	int GetThreadsY();

	//! get x size of the field
    /*!
	@return x size (in meter unit)
    */
	size_t GetNx();
	//! get y size of the field
    /*!
	@return y size (in meter unit)
    */
	size_t GetNy(); 
	size_t GetNz();

	//! get wavelength
    /*!
	@return wavelength (in meter unit)
    */
	float GetWaveLength();
	//! get wave number
    /*!
	@return wave number (in meter unit)
    */
	float GetWaveNum();
	//! get propagation distance
    /*!
	@return propagation distance (in meter unit)
    */
	float GetDistance(); 
	//! get sampling rate along to x-axis on source plane
    /*!
	@return sampling rate along to x-axis on source plane (in meter unit)
    */
	float GetPx();
	//! get sampling rate along to y-axis on source plane
    /*!
	@return sampling rate along to y-axis on source plane (in meter unit)
    */
	float GetPy();
	float GetPz();
	//! get sampling rate along to x-axis on source plane
    /*!
	@return sampling rate along to x-axis on source plane (in meter unit)
    */
	float GetSrcPx();
	//! get sampling rate along to y-axis on source plane
    /*!
	@return sampling rate along to y-axis on source plane (in meter unit)
    */
	float GetSrcPy();
	float GetSrcPz();
	//! get sampling rate along to x-axis on destination plane
    /*!
	@return sampling rate along to x-axis on destination plane (in meter unit)
    */
	float GetDstPx();
	//! get sampling rate along to y-axis on destination plane
    /*!
	@return sampling rate along to y-axis on destination plane (in meter unit)
    */
	float GetDstPy();
	float GetDstPz();

	//! get offset along to x-axis on source plane
    /*!
	@return offset along to x-axis on source plane (in meter unit)
    */
	float GetOx();
	//! get offset along to y-axis on source plane
    /*!
	@return offset along to y-axis on source plane (in meter unit)
    */
	float GetOy();
	float GetOz();
		
	//! get offset along to x-axis on source plane
    /*!
	@return offset along to x-axis on source plane (in meter unit)
    */
	float GetSrcOx();
	//! get offset along to y-axis on source plane
    /*!
	@return offset along to y-axis on source plane (in meter unit)
    */
	float GetSrcOy();
	float GetSrcOz();

	//! get offset along to x-axis on destination plane
    /*!
	@return offset along to x-axis on destination plane (in meter unit)
    */
	float GetDstOx();
	//! get offset along to y-axis on destination plane
    /*!
	@return offset along to y-axis on destination plane (in meter unit)
    */
	float GetDstOy();
	float GetDstOz();

	//! Get current field-type.
    /*!
    @return Current field-type. (e.g. CWO_FLD_COMPLEX)
    */
	int GetFieldType();
	//! Get current field-type as string.
    /*!
    @return Current field-type as string. (e.g. string "CWO_FLD_COMPLEX")
    */
	char *GetFieldName();

	//@}
	
	//###########################################
	/** @defgroup file File I/O
	*/
	//@{
	int CheckExt(const char* fname, const char* ext);
	
	//! Load image file or cwo file
	/*! 
	@brief The function can load bitmap, jpeg, tiff, png image formats. 
	@param fname Filename
	@param c Select color to read the image file
	@return When the function success to load file, the return value is CWO_SUCCESS, otherwise CWO_FAIL.
	*/
	int Load(char* fname, int c=CWO_GREY);//OK
	int Load(char* fname_amp, char *fname_pha, int c=CWO_GREY);//OK
	
	int Save(char* fname, CWO *r=NULL, CWO *g=NULL, CWO *b=NULL);
	int SaveMonosToColor(char* fname, char *r_name, char *g_name, char *b_name);
	int SaveAsImage(char* fname, int flag=CWO_SAVE_AS_INTENSITY, CWO *r=NULL, CWO *g=NULL, CWO *b=NULL);
	//@}


	void* GetBuffer(int flag=CWO_BUFFER_FIELD);

	int CmpCtx(cwoCtx &a, cwoCtx &b);
	
	size_t GetMemSizeCplx();
	size_t GetMemSizeFloat();
	size_t GetMemSizeChar();

	//diffraction
	void Diffract(float d, int type=CWO_ANGULAR, cwoInt4 *zp=NULL, CWO *numap=NULL);
	void DiffractConv(float d, int type, cwoComplex *ape, cwoComplex *prop, cwoInt4 *zp=NULL, CWO *numap=NULL, int prop_calc=1);
	//void DiffractConvImplicit(float d, int type, cwoComplex *ape, cwoComplex *prop, int prop_calc=1);
	void DiffractFourier(float d, int type, cwoInt4 *zp=NULL);
	void DiffractDirect(float d, CWO *snumap, CWO *dnumap);

	void Diffract3D(CWO &a, float d, int type);
	void Diffract(float d, int type, int Dx, int Dy, char *dir=NULL);
	void ReleaseTmpBuffer();

	void AngularProp(float z, int iNx, int iNy);

		
	float fresnel_c(float x);
	float fresnel_s(float x);
	void FresnelInt(
		float z, int x1, int y1, int x2, int y2);


	//###########################################
	/** @defgroup waves Planar & Spherical waves
	@image html sphericalwave.jpg "Geometry for spherical wave"
	*/
	//@{

	virtual void __AddSphericalWave(cwoComplex *p, float x, float y, float z, float px, float py, float a);
	virtual void __MulSphericalWave(cwoComplex *p, float x, float y, float z, float px, float py, float a);
	virtual void __AddApproxSphWave(cwoComplex *p, float x, float y, float z, float px, float py, float a);
	virtual void __MulApproxSphWave(cwoComplex *p, float x, float y, float z, float px, float py, float a);


	//! Add spherical wave to the field. The equation is as follows:@n
	//! \f$ u(x,y)=a \exp(i k \sqrt{(x-x_0)^2+(y-y_0)^2+z^2}) \f$ where \f$ k \f$ is the wave number.
	/*! 
	@param x : center of the spherical wave 
	@param y : center of the spherical wave
	@param z : distance between the field and the spherical wave
	@param a : amplitude of the spherical wave
	
	*/
	void AddSphericalWave(float x, float y, float z, float a=1.0f);
	//! Mutily spherical wave to the field.
	/*! 
	@param x : center of the spherical wave 
	@param y : center of the spherical wave
	@param z : distance between the field and the spherical wave
	@param a : amplitude of the spherical wave
	*/
	void MulSphericalWave(float x, float y, float z, float a=1.0f);
	//! Add spherical wave to the field.
	/*! 
	@param x : center of the spherical wave 
	@param y : center of the spherical wave
	@param z : distance between the field and the spherical wave
	@param px : sampling rate along to x-axis on the field
	@param py : sampling rate along to y-axis on the field
	@param a : amplitude of the spherical wave
	*/
	void AddSphericalWave(float x, float y, float z, float px, float py, float a=1.0f);
	//! Multiply spherical wave to the field.
	/*! 
	@param x : center of the spherical wave 
	@param y : center of the spherical wave
	@param z : distance between the field and the spherical wave
	@param px : sampling rate along to x-axis on the field
	@param py : sampling rate along to y-axis on the field
	@param a : amplitude of the spherical wave
	*/
	void MulSphericalWave(float x, float y, float z, float px, float py, float a=1.0f);
	//! Add Fresnel-approximated spherical wave to the field.
	/*! 
	@param x : center of the spherical wave 
	@param y : center of the spherical wave
	@param z : distance between the field and the spherical wave
	@param px : sampling rate along to x-axis on the field
	@param py : sampling rate along to y-axis on the field
	@param a : amplitude of the spherical wave
	*/


	void AddApproxSphWave(float x, float y, float z, float a=1.0f);
	void MulApproxSphWave(float x, float y, float z, float a=1.0f);

	void AddApproxSphWave(float x, float y, float z, float px, float py, float a=1.0f);
	//! Multiply Fresnel-approximated spherical wave to the field.
	/*! 
	@param x : center of the spherical wave 
	@param y : center of the spherical wave
	@param z : distance between the field and the spherical wave
	@param px : sampling rate along to x-axis on the field
	@param py : sampling rate along to y-axis on the field
	@param a : amplitude of the spherical wave
	*/
	void MulApproxSphWave(float x, float y, float z, float px, float py, float a=1.0f);
	//! Multiply planar wave to the field.
	/*! 
	@param kx : wave number along to x-axis on the field
	@param ky : wave number along to y-axis on the field
	@param kz : wave number along to z-axis on the field
	@param px : sampling rate along to x-axis on the field
	@param py : sampling rate along to y-axis on the field
	@param a : amplitude of the planar wave
	*/

	void MulPlanarWave(float kx, float ky, float kz, float px, float py, float a=1.0f);
	//! Multiply planar wave to the field.
	/*! 
	@param kx : wave number along to x-axis on the field
	@param ky : wave number along to y-axis on the field
	@param kz : wave number along to z-axis on the field
	@param a : amplitude of the planar wave
	*/
	void MulPlanarWave(float kx, float ky, float kz, float a=1.0f);

	//@}	

	//###########################################
	/** @defgroup cplx_ope Operations for complex amplitude
	*/
	//@{
	

	void Re(); //!< Taking the real part of complex amplitude. The field type is changed to CWO_FLD_INTENSITY. 
	void Im(); //!< Taking the imagenaly part of complex amplitude. The field type is changed to CWO_FLD_INTENSITY.
	void Conj(); //!< Calculating the complex conjugation of complex amplitude. 
	void Intensity(); //!< Calculating the absolute square (light intensity) of complex amplitude.  The field type is changed to CWO_FLD_INTENSITY.
	void Amp(); //!< Calculating the amplitude of complex amplitude.  The field type is changed to CWO_FLD_INTENSITY.

	//! Calculating phase distribution from complex amplitude.
	//! After executing this function, CWO maintains phase distribution with the value \f$-\pi\f$ to \f$+\pi\f$.@n
	//! The parameter offset adjusts the phase distribution, ex. if setting offset=CWO_PI, the phase range in the phase distribution is 0 to \f$2\pi\f$.
	/*! 
	@param offset : adjust the phase distribution
	@note  The field type is changed to CWO_FLD_PHASE.
	*/
	void Phase(float offset=0.0f); //!< Calculating the phase of complex amplitude. 

	void Arg(float offset=0.0f);

	void Cplx();  //!< Convering to complex amplitude if the field is not complex amplitude(CWO_FLD_COMPLEX).
	int Cplx(CWO &amp, CWO &ph);

	void ReIm(CWO &re, CWO &im);
	//@}	

void Char(); //conver to current data type to char
void Float(); //conver to current data type to float

	//
	virtual void Sqrt();
	void Div(float a);

//	cwoComplex Mul(cwoComplex a, cwoComplex b);
	cwoComplex Conj(cwoComplex a);

	//
	cwoComplex Polar(float amp, float arg);

	void SetPixel(int x, int y, float a); 
	void SetPixel(int x, int y, float amp, float ph);
	void SetPixel(int x, int y, cwoComplex a);
	void SetPixel(int x, int y, int z, float a); 
	void SetPixel(int x, int y, int z, float amp, float ph);
	void SetPixel(int x, int y, int z, cwoComplex a);
	void SetPixel(int x, int y, CWO &a);
	
	void AddPixel(int x, int y, cwoComplex a);
	void AddPixel(int x, int y, CWO &a);

	void MulPixel(int x, int y, float a);
	void MulPixel(int x, int y, cwoComplex a);
	void GetPixel(int x, int y, float &a);
	void GetPixel(int x, int y, cwoComplex &a);
	void GetPixel(int x, int y, int z, float &a);
	void GetPixel(int x, int y, int z, cwoComplex &a);
	
	virtual void __RandPhase(cwoComplex *a, float max, float min);
	virtual void __MulRandPhase(cwoComplex *a, float max, float min);
	void RandPhase(float max=CWO_PI, float min=-CWO_PI);
	void SetRandPhase(float max=CWO_PI, float min=-CWO_PI);
	void MulRandPhase(float max=CWO_PI, float min=-CWO_PI);

	template <class T> void __Copy(
		T *src, int x1, int y1, int sNx, int sNy,
		T *dst, int x2, int y2, int dNx, int dNy, 
		int Sx, int Sy);

	void Copy(CWO &a, int x1, int y1, int x2,int y2, int Sx, int Sy);
	//CWO Extract(int x, int y, int Nx, int Ny);

	void Clear(int c=0);
	void Fill(cwoComplex a);

	template <class T> void FlipH(T *a);
	template <class T> void FlipV(T *a);
	template <class T> void FlipHV(T *a);
	void Flip(int mode=0);

	void Crop(int x1, int y1, int Sx, int Sy);

	void ShiftX(int s, int flag=0);
	void ShiftY(int s, int flag=0);
	

	//! Fill in rectangular area
	/*! 
	@param (x, y) : start position (in pixel unit)
	@param (Sx,Sy) : area size (in pixel unit)
	@param a : complex value
	@param flag : when CWO_FILL_INSIDE, the function fills inside the rectangular area with $sx \times sy$.
	*/
	void Rect(int x, int y, int Sx, int Sy, cwoComplex a, int flag=CWO_FILL_INSIDE);
	
	//! Fill in rectangular area
	/*!
	@param (x, y) : start position (in pixel unit)
	@param r : radius (in pixel unit)
	@param a : complex value
	*/
	void Circ(int x, int y, int r, cwoComplex a, int flag=CWO_FILL_INSIDE);
	void MulCirc(int x, int y, int r, cwoComplex a);

	void __Hanning(cwoComplex *a, int m, int n, int Wx, int Wy); //!< Hanning window
	void __Hamming(cwoComplex *a, int m, int n, int Wx, int Wy); //!< Hamming window

	//cwoComplex inter_nearest(cwoComplex *old, double x, double y);
	//cwoComplex inter_linear(cwoComplex *old, float x, float y);
	//cwoComplex inter_cubic(cwoComplex *old, float x, float y);
	
	virtual void __InterNearest(
		cwoComplex *p_new, int newNx, int newNy,
		cwoComplex *p_old, int oldNx, int oldNy,
		float mx, float my,
		int xx, int yy);

	virtual void __InterNearest(
		cwoComplex *p_new, int newNx, int newNy,
		cwoComplex *p_old, int oldNx, int oldNy);

	virtual void __InterLinear(
		cwoComplex *p_new, int newNx, int newNy,
		cwoComplex *p_old, int oldNx, int oldNy,
		float mx, float my,
		int xx, int yy);

	virtual void __InterLinear(
		cwoComplex *p_new, int newNx, int newNy,
		cwoComplex *p_old, int oldNx, int oldNy);

	virtual void __InterCubic(
		cwoComplex *p_new, int newNx, int newNy,
		cwoComplex *p_old, int oldNx, int oldNy,
		float mx, float my,
		int xx, int yy);

/*	virtual void __InterCubic(
		cwoComplex *p_new, int newNx, int newNy,
		cwoComplex *p_old, int oldNx, int oldNy);
*/

	void Resize(int dNx, int dNy, double mx=-1, double my=-1, int flag=CWO_INTER_LINEAR);
	void Resize(int dNx, int dNy, int flag);

	void ErrorDiffusion(CWO *a, int flag=CWO_ED_FLOYD);//!< error diffusion method

	void RGB2YCbCr(CWO *rgb, CWO *ycbcr);
	void YCbCr2RGB(CWO *rgb, CWO *ycbcr);


	//###########################################
	/** @defgroup transform Integral Transfoms and related operations (FFT etc.)
	*/
	//@{
	void FourierShift(float m, float n);

	int FFT(int flag=0); //!< Fast Fourier Transform (FFT)
	virtual void FFTShift(); //!< FFT Shift 
	
	virtual void __ScaledFFTCoeff(cwoComplex *p, float sx, float sy);
	//virtual void __ScaledFFTCoeff2(cwoComplex *p, float sx, float sy);
	virtual void __ScaledFFTKernel(cwoComplex *p, float sx, float sy);
	//void ScaledFFT(int flag=0); //!< Scaled FFT 

	virtual void __NUFFT_T1(cwoComplex *p_fld, cwoFloat2 *p_x, int R=2, int Msp=12);
	virtual void __NUFFT_T2(cwoComplex *p_fld, cwoFloat2 *p_x, int R=2, int Msp=12);
	
	void NUFFT_T1(int R=2, int Msp=12);//!< Non-uniform FFT (Type1) 
	void NUFFT_T2(int R=2, int Msp=12);//!< Non-uniform FFT (Type2) 

	void NUFFT1(CWO *map, int R=2, int Msp=12);//!< Non-uniform FFT (Type1) 
	void NUFFT2(CWO *map, int R=2, int Msp=12);//!< Non-uniform FFT (Type2) 

	void ConvertSamplingMap(int type); 
	void SamplingMapScaleOnly(int Nx, int Ny, float R, float sgn);
	//@}


	virtual void Gamma(float g);//OK
	virtual void Threshold(float max, float min=0.0);//OK
	void Binary(float th=0.0, float max=1.0, float min=0.0);//OK
	
	virtual void __PickupFloat(float *src, float *pix_p, float pix);
	virtual void __PickupCplx(cwoComplex *src, cwoComplex *pix_p, float pix);
	
	//! Pickup pixels
	/*!
	@brief if a(m,n)=pix , this(m,n)=this(m,n) otherwise this(m,n)=0 
	@param a : pointer to source image 
	@param pix : pixel value 
	*/
	void	Pickup(CWO *a, float pix);
	
	//
	virtual float Average();//OK
	virtual float Variance();//OK
	
	//! Calculate variance map
	/*! 
	@brief Variance map is 
	For example, it is useful for detecting in-focus planes.@n
	Conor P. McElhinney, John B. McDonald, Albertina Castro, Yann Frauel, Bahram Javidi, and Thomas J. Naughton, "Depth-independent segmentation of macroscopic three-dimensional objects encoded in single perspectives of digital holograms," Opt. Lett. 32, 1229-1231 (2007) 
	@param sx: 
	@param sy: 
	*/
	void VarianceMap(int sx, int sy);//OK


	/*	 

	//
	float SumTotal();
	
	float Variance(float ave);
	
	float SMDx();*/

	//
	virtual void Expand(int Nx, int Ny);
	void ExpandTwice(CWO *src);
	void ExpandHalf(CWO *dst);

	virtual void __MaxMin(cwoComplex *a, float *max, float *min, int *max_x=NULL, int *max_y=NULL,int *min_x=NULL, int *min_y=NULL);//OK


	//int MaxMin(float *a, float *max, float *min, int *max_x=NULL, int *max_y=NULL,int *min_x=NULL, int *min_y=NULL);
	float Max();//OK
	float Min();//OK
	int MaxMin(float *max, float *min, int *max_x=NULL, int *max_y=NULL,int *min_x=NULL, int *min_y=NULL);//OK
	
	virtual void Quant(float lim, float max, float min);

	int ScaleReal(float lim=1.0);
	int ScaleReal(float i1, float i2, float o1, float o2);
	int ScaleCplx(float lim=1.0);
	

	//###########################################
	/** @defgroup error Error measurement
	*/
	//@{

	float SNR(CWO &ref);
	float PSNR(CWO &ref);

	//@}

	virtual void test();
	void test2(CWO &a);

	//****************
	//For PLS
	//****************
	virtual cwoObjPoint* GetPointBuffer(){return NULL;};
	virtual int GetPointNum(){return 0;};
	virtual void SetPointNum(int num){};
	virtual void ScalePoint(float lim){};
	void PLS(int flag);

//	virtual void __PLS_Fresnel(){};
//	virtual void __PLS_CGH_Fresnel(){};

	virtual void __PLS_Huygens(float ph=0.0f){};
	virtual void __PLS_Fresnel(float ph=0.0f){};
	virtual void __PLS_CGH_Fresnel(float ph=0.0f){};


////////////////////////
//Test code
////////////////////////

void SetD1(float deg);//set displacement to source plane
void SetD2(float deg);//set displacement to destination plane
void SetX1(float deg);
void SetX2(float deg);

void SetD1(int x, int y, float d);//set displacement to source plane
void SetD2(int x, int y, float d);//set displacement to destination plane
void SetX1(int x, int y, cwoFloat2 c);
void SetX2(int x, int y, cwoFloat2 c);

virtual void __ArbitFresnelDirect(
	cwoComplex *p1, cwoComplex *p2, 
	cwoFloat2 *p_x1, cwoFloat2 *p_x2, 
	float *p_d1, float *p_d2);

virtual void __ArbitFresnelCoeff(
	cwoComplex *p, cwoFloat2 *p_x2, float *p_d2);



void SamplingMapX(cwoFloat2 *p, int Nx, int Ny, int quadrant);
void SamplingMapY(cwoFloat2 *p, int Nx, int Ny, int quadrant);


void NUDFT();

//void __ScaledAngularProp(cwoComplex *a, float px, float py);




virtual void __InvFx2Fy2(cwoComplex *a);

//###########################################
/** @defgroup	sdiff	Special diffractions
*/
//@{
float z0;
float pz;
float fresnel_c_tbl(float x, float z, float z0, float pz);
float fresnel_s_tbl(float x, float z, float z0, float pz);
void PrepareTblFresnelApproxInt(int nx, float z0, float pz, int nz);
void FresnelApproxIntTbl(
	float z, int x1, int y1, int sx1, int sy1);

void FresnelPolyAperture();	

void ParticleField(cwoFloat3 pos, float radius, float amp=1.0f, float init_ph=0.0f); //!< Complex amplitude from particle with radius
//@}


};

//!  Class for vector operations. 
/*!
This class serves vector operations.
*/
class cwoVect{
public:
    float x, y, z;
public:
	cwoVect(){};

	cwoVect(float tx, float ty, float tz) {
        x = tx; y = ty; z = tz;
    };

	void Set(float tx, float ty, float tz){
		 x = tx; y = ty; z = tz;
	}

	cwoVect operator=(cwoVect a){
        cwoVect c;
        x =a.x;
        y =a.y;
        z =a.z;
        return *this;
    };

	cwoVect operator+(cwoVect a){	//vector+vector
		cwoVect c;
		c.x = x+a.x;
		c.y = y+a.y;
		c.z = z+a.z;
		return c;
	}

    cwoVect operator-(cwoVect a){
        cwoVect c;
        c.x = x - a.x;
        c.y = y - a.y;
        c.z = z - a.z;
        return c;
    };

    cwoVect operator*(float a){
        cwoVect c;
        c.x = x * a;
        c.y = y * a;
        c.z = z * a;
        return c;
    };

	cwoVect operator/(float a){
        cwoVect c;
        if(a == 0.0f) return *this;
        c.x = x / a;
        c.y = y / a;
        c.z = z / a;
        return c;
    };

    cwoVect& operator+=(cwoVect a) {
        x += a.x;
        y += a.y;
        z += a.z;
        return *this;
    };

    cwoVect& operator-=(cwoVect a) {
		x -= a.x;
        y -= a.y;
        z -= a.z;
        return *this;
    };

    cwoVect& operator*=(float a){
        x *= a;
        y *= a;
        z *= a;
        return *this;
    };

    cwoVect& operator/=(float a){
        if(a == 0.0f) return *this;
        x /= a;
        y /= a;
        z /= a;
        return *this;
    };

	float Length(){
		double tx = (double)x;
		double ty = (double)y;
		double tz = (double)z;
		return (float)sqrt(tx*tx + ty*ty + tz*tz);
	}

	cwoVect Normalize(){
		cwoVect tmp;
		tmp=*this;

		float len=Length();
		len = 1.0f / len;
		tmp.x *= len;
		tmp.y *= len;
		tmp.z *= len;
		return tmp;
	}

	float Dot(cwoVect a){
		return (x*a.x+y*a.y+z*a.z);
	}

	cwoVect Cross(cwoVect a){
		cwoVect vec;
		double x1, y1, z1, x2, y2, z2;
        x1 = (double)x;
		y1 = (double)y;
		z1 = (double)z;
		x2 = (double)a.x;
		y2 = (double)a.y;
		z2 = (double)a.z;
 		vec.x = (float)(y1 * z2 - z1 * y2);
		vec.y = (float)(z1 * x2 - x1 * z2);
		vec.z = (float)(x1 * y2 - y1 * x2);
		return vec;
	}

};




//!  Class for look-up table
/*!
This class manages look-up tables.
*/
class cwoTbl{
	float *tbl_sin;
	float *tbl_cos;
	CWO *tbl_wrp;
	int __Nz;

	//! set the number of tables along to z
	/*! 
	@param Nz : the number of tables along to z
	@sa 
	*/
	void SetNz(int Nz);
public:
	
	cwoTbl();
	~cwoTbl();

	//! get the number of tables along to z
	/*! 
	@return the number of tables along to z
	*/
	int GetNz();

	//! Make sin table
	/*! 
	@param N : sampling number in one cycle of sin
	*/
	void MakeSin(int N);
	
	//! Make cos table
	/*! 
	@param N : sampling number in one cycle of cos
	*/
	void MakeCos(int N);

	//! Make WRP table
	/*! 
	@param z : distance from WRP
	@param Nz : number of planes along to z-axis
	@param pz : sampling rate between neighbor planes (in meter)
	@param wl : wavelength (in meter)
	@param px : sampling rate on WRP (in meter)
	@param py : sampling rate on WRP (in meter)
	*/
	void MakeWRPTbl(float z, int Nz, float pz, float wl, float px, float py);
	
	//! Clip WRP tables to the half size
	/*! 
	*/
	void ClipWRPTbl();
	

	//! Get WRP table as pointer to CWO 
	/*! 
	@param idx : index for plane
	@return pointer to CWO 
	*/
	CWO *GetWRPTbl(int idx);


};

#endif

/** @example  simple_diffraction.cpp
This code is to calculate diffraction using the angular spectrum method.
@image html lena512x512_diffract.jpg
*/

/** @example simple_diffraction_with_gpu.cpp
This code is to calculate diffraction using the angular spectrum method on a GPU.
@image html lena512x512_diffract.jpg
*/

/** @example "Amplitude CGH"
This code is to calculate an amplitude CGH.
@image html lena512x512_diffract.jpg
@code
CWO a;
a.Load("sample.bmp"); //Load image file 
a.Diffract(0.1, CWO_ANGULAR); //Calculate diffraction from the image using the angular spectrum method
a.Re(); //Taking the real part from the diffracted result
a.Scale(255); // Convert the intensity to 255-steps data
a.Save("diffract.bmp"); //Save the intensity as bitmap file
@endcode
*/

/** @example "Kinoform"
This code is to calculate a kinoform.
@image html lena512x512_diffract.jpg
@code
CWO a;
a.Load("sample.bmp"); //Load image file 
a.Diffract(0.1, CWO_ANGULAR); //Calculate diffraction from the image using the angular spectrum method
a.Phase(); //Taking the phase from the diffracted result
a.Scale(255); // Convert the intensity to 255-steps data
a.Save("diffract.bmp"); //Save the intensity as bitmap file
@endcode
*/

/** @example "Shifted-Fresnel diffraction"
This code is to calculate Shifted-Fresnel diffraction.
@code
CWO a;
a.Load("sample.bmp"); //Load image file
a.SetSrcPitch(10e-6,10e-6);
a.SetDstPitch(24e-6,24e-6);
a.Diffract(0.1, CWO_SHIFTED_FRESNEL); //Calculate diffraction from the image using the angular spectrum method
a.Intensity(); 
a.Scale(255); // Convert the intensity to 255-steps data
a.Save("diffract.bmp"); //Save the intensity as bitmap file
@endcode
*/

/** @example Using CPU threads
This code is to calculate Shifted-Fresnel diffraction.
@code
CWO a;
a.Load("sample.bmp"); //Load image file
a.SetThreads(8);
a.SetSrcPitch(10e-6,10e-6);
a.SetDstPitch(24e-6,24e-6);
a.Diffract(0.1, CWO_SHIFTED_FRESNEL); //Calculate diffraction from the image using the angular spectrum method
a.Intensity(); 
a.Scale(255); // Convert the intensity to 255-steps data
a.Save("diffract.bmp"); //Save the intensity as bitmap file
@endcode
*/