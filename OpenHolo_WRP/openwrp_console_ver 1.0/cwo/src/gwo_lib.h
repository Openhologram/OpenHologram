// Copyright (C) Tomoyoshi Shimobaba 2011-



#ifndef _GWO_LIB_H
#define _GWO_LIB_H

#include "cwo_lib.h"

#define GWO_FOR_CUDA




#ifndef __linux__
	#ifdef GWODLL
		#define GWO_EXPORT   __declspec(dllexport)
	#else
		#define GWO_EXPORT   __declspec(dllimport)
	#endif
#else
	#define GWO_EXPORT
#endif

//if you want to generate static library, omit the following comment out.
/*
#ifdef GWODLL
#define GWO_EXPORT   
#else
#define GWO_EXPORT   
#endif
*/

//if you want to generate DLL, omit the following comment out.
#ifndef __linux__
	#define GWOAPI __stdcall
#else
	#define GWOAPI 
#endif



#ifdef __cplusplus
extern "C" {
#endif






//###########################################
/** @defgroup difftype Diffraction types
*/

//@{


/**
* @def GWO_ANGULAR
* When selecting GWO_ANGULAR, the GWO library calculates the angular spectrum method.
* The Angular spectrum method is expressed as:

* @f{eqnarray*}
u(x,y) = \int \!\!\int_{-\infty}^{+\infty} & & 
A(f_x, f_y, 0) \exp(i z \sqrt{k^2 - 4 \pi^2 (f_x^2 + f_y^2)}) \\ 
& & \exp(i \ 2 \pi (f_x x + f_y y)) d f_x d f_y
* @f}

The GWO library calculates this diffraction using the FFT algorithm,

* @f{eqnarray*}
u(m_2,n_2) = FFT[ u(m_1, n_1) ] \exp(i z \sqrt{k^2 - 4 \pi^2 (f_x^2 + f_y^2)}) 
* @f}


*/
#define GWO_ANGULAR					(0x01)

#define GWO_QUASI_ANGULAR			(0x02)

/**
* @def GWO_FRESNEL_CONV
* When selecting GWO_FRESNEL_CONV, the GWO library calculates 
* the Fresnel diffraction (convolution form).
*/
#define GWO_FRESNEL_CONV			(0x03)

/**
* @def GWO_FRESNEL_FOURIER
* When selecting GWO_FRESNEL_ANALYSYS, the GWO library calculates 
* the Fresnel diffraction (fourier form).
*/
#define GWO_FRESNEL_FOURIER			(0x04)

/**
* @ingroup difftype
* @def GWO_FRESNEL_ANALYSYS
* When selecting GWO_FRESNEL_ANALYSYS, the GWO library calculates 
* the Fresnel diffraction (convolution form).
*/
#define GWO_FRESNEL_ANALYSIS		(0x05)

/**
* @ingroup difftype
* @def GWO_FRESNEL_DBL
* When selecting GWO_FRESNEL_DBL, the GWO library calculates 
* the double-step Fresnel diffraction.
*/
#define GWO_FRESNEL_DBL				(0x06)



/**
* @def GWO_FRAUNHOFER
* When selecting GWO_FRAUNHOFER, the GWO library calculates 
* the Fraunhofer diffraction.
*/
#define GWO_FRAUNHOFER				(0x07)


/**
* @def GWO_SHIFTFRESNEL
* When selecting GWO_SHIFTFRESNEL, the GWO library calculates 
* the Shfted-Fresnel diffraction.
*/
#define GWO_SHIFTED_FRESNEL			(0x08)

//@}



#define GWO_CGH_BASIC				(0x20)
#define GWO_CGH_FRESNEL				(0x21)
#define GWO_CGH_RECURRENCE			(0x22)


////////////////////////////////////////////////
//
////////////////////////////////////////////////
#define GWO_ON		1
#define GWO_OFF		0

////////////////////////////////////////////////
//
////////////////////////////////////////////////
//#define GWO_MEM_PAGELOCKED			(0x01)


////////////////////////////////////////////////
//define pi
////////////////////////////////////////////////
//#define GWO_PI (3.1415926535897932384626433832795f)


////////////////////////////////////////////////
//type
////////////////////////////////////////////////

#define GWO_RE(a) (a[0])
#define GWO_IM(a) (a[1])

//#define GWO_CONJ(a) (GWO_IM(a)*=-1.0) 

#define GWO_INTENSITY(a) (GWO_RE(a)*GWO_RE(a)+GWO_IM(a)*GWO_IM(a))
#define GWO_AMP(a) (sqrt(GWO_INTENSITY(a)))
#define GWO_PHASE(a) (atan(GWO_IM(a)/GWO_RE(a))

/*
#ifdef GWO_FOR_CUDA
	typedef float cwoComplex[2];
#endif

#ifdef GWO_FOR_CPU
	typedef float cwoComplex[2];
#endif
	*/

typedef struct{
	float	a;
	float	x;
	float	y;
	float	z;
}gwoObjPoint;



#ifdef GWO_FOR_CUDA

	#ifndef __CUFFT_H_
	#define __CUFFT_H_
		#include <cufft.h>
	#endif
	
	typedef cufftHandle gwofftPlan; //CUDA
#endif

#ifdef GWO_FOR_CPU
	#ifndef __FFTW_H_
	#define __FFTW_H_
		#include "fftw3.h"
	#endif
	typedef fftwf_plan gwofftPlan; //CPU
#endif


#ifdef GWO_FOR_CUDA
	typedef cudaStream_t cwoStream;
#else
	typedef int cwoStream; //on CPU, cwoStream is defined as type "int" instead of cudaStream_t
#endif


typedef struct gwoCtx{
	int NThreadX;
	int NThreadY;
	cwoStream stream;

	void SetThreadX(int n){NThreadX=n;};
	void SetThreadY(int n){NThreadY=n;};
	int GetThreadX(){return NThreadX;};
	int GetThreadY(){return NThreadY;};
	void SetStream(cwoStream strm){stream=strm;};
	cwoStream GetStream(){return stream;};
}gwoCtx;



	/*
#ifdef GWO_FOR_CPU
	typedef int cwoStream; //on CPU, cwoStream is defined as type "int" instead of cudaStream_t
#endif
	*/
	/*
typedef struct gwoCtx{
	cwoCtx *ctx;
	cwoStream cstream;
}gwoCtx;
*/



GWO_EXPORT void  GWOAPI gwoSetThreads(int Nx, int Ny);
GWO_EXPORT int  GWOAPI gwoGetThreadsX();
GWO_EXPORT int  GWOAPI gwoGetThreadsY();


////////////////////////////////////////////////
//timer
////////////////////////////////////////////////

GWO_EXPORT void GWOAPI gwoSetTimer();
GWO_EXPORT float GWOAPI gwoEndTimer();


////////////////////////////////////////////////
//???
////////////////////////////////////////////////


/** 
*@brief Get timer value
*/

GWO_EXPORT int GWOAPI gwoDevNum();

/** 
*@brief Get timer value
*/
GWO_EXPORT int  GWOAPI gwoSelectDev(int num);

/** 
*@brief Get timer value
*/
GWO_EXPORT int  GWOAPI gwoBarrier();





//@}


/***********************************************
Intialize functions
***********************************************/
//GWO_EXPORT int GWOAPI gwoStart();
//GWO_EXPORT void GWOAPI gwoEnd();
//
//GWO_EXPORT int GWOAPI gwoInit(cwoCtx *c, int mode, int width, int height);
//GWO_EXPORT void GWOAPI gwoFree(cwoCtx *c);




//###########################################
/** @defgroup setparam Setting parameters
*/

//@{

/**
*@ingroup setparam
*@brief Set the wavelength of light.
*@param lambda the wavelength of light. The unit is meter.
*@retval none
*/
//GWO_EXPORT void GWOAPI gwoSetWaveLength(cwoCtx *c, float lambda);

/**
*@brief Set sampling spacings on the calculation area. The units are metre, respectively.
*@param px sampling spacing along with horizontal direction.
*@param py sampling spacing along with vertical direction.
*@retval none
*/
//GWO_EXPORT void GWOAPI gwoSetPitch(cwoCtx *c, float px, float py);

/**
*@brief This funtion is reserved. Do not use in current GWO library.
*/
//GWO_EXPORT void GWOAPI gwoSetSrcPitch(cwoCtx *c, float px, float py);

/**
*@brief This funtion is reserved. Do not use in current GWO library.
*/
//GWO_EXPORT void GWOAPI gwoSetDstPitch(cwoCtx *c, float px, float py);

/**
*@brief 
	Set the total number of object points.
	The function allocate the memory space.	
*@param N Number of object points 
*@retval none
*/
//GWO_EXPORT void GWOAPI gwoSetObjNum(cwoCtx *c, int N);


/**
*@brief 
	Set offsets on the source plane.
*@param ox : offset in x-direction
*@param oy : offset in y-direction
*@retval none
*/
//GWO_EXPORT void GWOAPI gwoSetSrcOffset(cwoCtx *c, int ox, int oy);

/**
*@brief 
	Set offsets on the destination plane.
*@param ox : offset in x-direction
*@param oy : offset in y-direction
*@retval none
*/
//GWO_EXPORT void GWOAPI gwoSetDstOffset(cwoCtx *c, int ox, int oy);
//
//
//GWO_EXPORT void GWOAPI gwoSetPropDist(cwoCtx *c, float z);
//GWO_EXPORT void GWOAPI gwoSetPropDist2(cwoCtx *c, float z);


GWO_EXPORT int GWOAPI gwoAdjustObjNum(int N);



GWO_EXPORT void GWOAPI gwoSetPlanerWave(cwoCtx *c, float a, float argx, float argy);





//@}

//###########################################
/** @defgroup commun Communication between the host and the co-processor
*/
//@{
//

/**
/*ingroup commun
*@brief Send input data on the host computer to the co-processor.
*@param data input data. The type of the value is cwoComplex(complex number) or gwoObjPoint.
*@retval none
*/
//GWO_EXPORT void GWOAPI gwoSendData(cwoCtx *c, void *data);



/**
*@brief Receive calculated data from the co-processor.
*@param data calculated data. The type of the value is cwoComplex(complex number) or gwoObjPoint.
*@retval none
*/
//GWO_EXPORT void GWOAPI gwoReceiveResult(cwoCtx *c, void *data);



//GWO_EXPORT void GWOAPI gwoSendDataAsync(cwoCtx *c, void *data);
//GWO_EXPORT void GWOAPI gwoReceiveResultAsync(cwoCtx *c, void *data);

//@}

//###########################################
/** @defgroup complex Complex number functions
*/
//@{

//@ingroup complex

//GWO_EXPORT void GWOAPI gwoHostComplexAdd(cwoComplex* a, cwoComplex *b, cwoComplex *c,int sx, int sy);
//GWO_EXPORT void GWOAPI gwoHostComplexMult(cwoComplex* a, cwoComplex *b, cwoComplex *c,int sx, int sy);
//
////GWO_EXPORT void GWOAPI gwoIntensity(cwoCtx *c);
//GWO_EXPORT void GWOAPI gwoCoeff(cwoCtx *c, int flag);
//
//
////GWO_EXPORT float GWOAPI gwoHostComplexSNR(cwoComplex *a, cwoComplex *b, int Nx, int Ny);
//GWO_EXPORT float GWOAPI gwoHostFloatSNR(float *ref, float *sig, int Nx, int Ny);
//GWO_EXPORT float GWOAPI gwoHostCharSNR(unsigned char *ref, unsigned char *sig, int Nx, int Ny);

//@}

//###########################################
/** @defgroup assist Assistance funtions
*/
//@{

//@ingroup assist

GWO_EXPORT void GWOAPI gwoNormalize(cwoComplex* inbuf, unsigned char *outbuf, int sx, int sy, float max, float min);
//GWO_EXPORT void GWOAPI gwoHostSearchMaxMin(float* buf, float *max, float *min, int sx, int sy);
//
//GWO_EXPORT int  GWOAPI gwoHostCopyArea(
//						cwoComplex *src, 
//						int src_x, int src_y, int src_width, int src_height,
//						cwoComplex *dst, 
//						int dst_x, int dst_y, int dst_width, int dst_height);
//
//
//GWO_EXPORT int GWOAPI gwoHostExpandC2C(
//			cwoComplex *src, int src_width, int src_height,
//			cwoComplex *dst, int dst_width, int dst_height);
/*
GWO_EXPORT int GWOAPI gwoHostExpandF2R(
			float *src, int src_width, int src_height,
			cwoComplex *dst, int dst_width, int dst_height);

GWO_EXPORT int GWOAPI gwoHostExpandF2I(
			float *src, int src_width, int src_height,
			cwoComplex *dst, int dst_width, int dst_height);

GWO_EXPORT int GWOAPI gwoHostExpandF2F(
			float *src, int src_width, int src_height,
			float *dst, int dst_width, int dst_height);
*/




//@}

//###########################################
/** @defgroup calc Calculation diffraction integrals
*/
//@{
//

/**
*@ingroup calc
*@brief 
	Calculate the diffraction using the co-processor. 
	The type of diffraction is indicated in gwoInit function.
*@param z Propagation distance. The unit is meter.
*@retval none
*/

//GWO_EXPORT void GWOAPI gwoCalc(cwoCtx *c);

//GWO_EXPORT void GWOAPI gwoThreshold(cwoCtx *c, float threshold);

//@}




//////////////////////////////
//////////////////////////////



#define GWO_C2C	0
#define GWO_R2C	1
#define GWO_C2R	2


GWO_EXPORT int GWOAPI gwoDevMalloc(void **ptr, int size);
GWO_EXPORT int GWOAPI gwoDevFree(void *ptr);

GWO_EXPORT int GWOAPI gwoHostMalloc(void **ptr, int size);
GWO_EXPORT int GWOAPI gwoHostFree(void *ptr);

//GWO_EXPORT void GWOAPI gwoSetSize(cwoCtx *c, int Nx, int Ny);

GWO_EXPORT void GWOAPI gwoSend(cwoCtx *c, gwoCtx *gtx, void *src, void *dst, int size);
GWO_EXPORT void GWOAPI gwoRecv(cwoCtx *c, gwoCtx *gtx, void *src, void *dst, int size);

GWO_EXPORT void GWOAPI gwoCGHRecurrence(cwoCtx *c, gwoCtx *gtx, gwoObjPoint *obj, float *hol, int N);
GWO_EXPORT void GWOAPI gwoThreshold2(cwoCtx *c, gwoCtx *gtx, float *a, int threshold);

GWO_EXPORT void GWOAPI gwoFill(cwoCtx *c, gwoCtx *gtx, cwoComplex *a, cwoComplex pix);


GWO_EXPORT void GWOAPI gwoFFT(cwoCtx *c, gwoCtx *gtx, void *src, void *dst, int mode);
GWO_EXPORT void GWOAPI gwoIFFT(cwoCtx *c, gwoCtx *gtx, void *src, void *dst);
GWO_EXPORT void GWOAPI gwoFFTShift(cwoCtx *c, gwoCtx *gtx, void *src, void *dst);
GWO_EXPORT void GWOAPI gwoNUFFT_T1(cwoCtx *c, gwoCtx *gtx, cwoComplex *p_fld, cwoFloat2 *p_x, int R, int Msp);
GWO_EXPORT void GWOAPI gwoNUFFT_T2(cwoCtx *c, gwoCtx *gtx, cwoComplex *p_fld, cwoFloat2 *p_x, int R, int Msp);

GWO_EXPORT void GWOAPI gwoAngularProp(cwoCtx *c, gwoCtx *gtx, cwoComplex *buf);//Create, Center 
GWO_EXPORT void GWOAPI gwoAngularPropFS(cwoCtx *, gwoCtx *gtxc, cwoComplex *buf);//Create, FFTShift
GWO_EXPORT void GWOAPI gwoAngularPropMul(cwoCtx *c, gwoCtx *gtx, cwoComplex *buf);//Multiply, Center
GWO_EXPORT void GWOAPI gwoAngularPropMulFS(cwoCtx *c, gwoCtx *gtx, cwoComplex *buf);//Multiply, FFTShift
//GWO_EXPORT void GWOAPI gwoShiftedAngularProp(cwoCtx *c, gwoCtx *gtx, cwoComplex *buf);

GWO_EXPORT void GWOAPI gwoHuygensProp(cwoCtx *c, gwoCtx *gtx, cwoComplex *buf);
//GWO_EXPORT void GWOAPI gwoHuygensCoeff(cwoCtx *c, gwoCtx *gtx, cwoComplex *buf);

GWO_EXPORT void GWOAPI gwoFresnelConvProp(cwoCtx *c, gwoCtx *gtx, cwoComplex *buf);
GWO_EXPORT void GWOAPI gwoFresnelConvCoeff(cwoCtx *c, gwoCtx *gtx, cwoComplex *buf);
GWO_EXPORT void GWOAPI gwoShiftedFresnelAperture(cwoCtx *c, gwoCtx *gtx, cwoComplex *buf);
GWO_EXPORT void GWOAPI gwoShiftedFresnelProp(cwoCtx *c, gwoCtx *gtx, cwoComplex *buf);
GWO_EXPORT void GWOAPI gwoShiftedFresnelCoeff(cwoCtx *c, gwoCtx *gtx, cwoComplex *buf);

GWO_EXPORT void GWOAPI gwoARSSFresnelAperture(cwoCtx *c, gwoCtx *gtx, cwoComplex *buf);
GWO_EXPORT void GWOAPI gwoARSSFresnelProp(cwoCtx *c, gwoCtx *gtx, cwoComplex *buf);
GWO_EXPORT void GWOAPI gwoARSSFresnelCoeff(cwoCtx *c, gwoCtx *gtx, cwoComplex *buf);

GWO_EXPORT void GWOAPI gwoFresnelFourierProp(cwoCtx *c, gwoCtx *gtx, cwoComplex *a);
GWO_EXPORT void GWOAPI gwoFresnelFourierCoeff(cwoCtx *c, gwoCtx *gtx, cwoComplex *a);

GWO_EXPORT void GWOAPI gwoFresnelDblAperture(cwoCtx *c, gwoCtx *gtx, cwoComplex *a, float z1);
GWO_EXPORT void GWOAPI gwoFresnelDblFourierDomain(cwoCtx *c, gwoCtx *gtx, cwoComplex *a, float z1, float z2, cwoInt4 *zp);
GWO_EXPORT void GWOAPI gwoFresnelDblCoeff(cwoCtx *c, gwoCtx *gtx, cwoComplex *a, float z1, float z2);

GWO_EXPORT void GWOAPI gwoAmp(cwoCtx *c, gwoCtx *gtx, cwoComplex *a, cwoComplex *b);
GWO_EXPORT void GWOAPI gwoPhase(cwoCtx *c, gwoCtx *gtx, cwoComplex *a, cwoComplex *b, float offset);
GWO_EXPORT void GWOAPI gwoArg(cwoCtx *c, gwoCtx *gtx, cwoComplex *a, cwoComplex *b, float scale, float offset);
GWO_EXPORT void GWOAPI gwoRe(cwoCtx *c, gwoCtx *gtx, cwoComplex *a, cwoComplex *b);
GWO_EXPORT void GWOAPI gwoIm(cwoCtx *c, gwoCtx *gtx, cwoComplex *a, cwoComplex *b);
GWO_EXPORT void GWOAPI gwoIntensity(cwoCtx *c, gwoCtx *gtx, cwoComplex *a, cwoComplex *b);

GWO_EXPORT void GWOAPI gwoExpand(cwoCtx *ctx, gwoCtx *gtx, 
								 void *src, int sNx, int sNy,
								 void *dst, int dNx, int dNy, int mode);
GWO_EXPORT void GWOAPI gwoCopy(
	cwoCtx *ctx, gwoCtx *gtx, 
	cwoComplex *src, int x1, int y1, int sNx, int sNy,
	cwoComplex *dst, int x2, int y2, int dNx, int dNy, 
	int Sx, int Sy);

GWO_EXPORT void GWOAPI gwoAddCplx(cwoCtx *c, gwoCtx *gtx, cwoComplex *A, cwoComplex B, cwoComplex *C);
GWO_EXPORT void GWOAPI gwoAddCplxArry(cwoCtx *c, gwoCtx *gtx, cwoComplex *A, cwoComplex *B, cwoComplex *C);
GWO_EXPORT void GWOAPI gwoSubCplx(cwoCtx *c, gwoCtx *gtx, cwoComplex *A, cwoComplex B, cwoComplex *C);
GWO_EXPORT void GWOAPI gwoSubCplxArry(cwoCtx *c, gwoCtx *gtx, cwoComplex *A, cwoComplex *B, cwoComplex *C);
GWO_EXPORT void GWOAPI gwoMulCplx(cwoCtx *c, gwoCtx *gtx, cwoComplex *A, cwoComplex B, cwoComplex *C);
GWO_EXPORT void GWOAPI gwoMulCplxArry(cwoCtx *c, gwoCtx *gtx, cwoComplex *A, cwoComplex *B, cwoComplex *C);
GWO_EXPORT void GWOAPI gwoDivCplx(cwoCtx *c, gwoCtx *gtx, cwoComplex *A, cwoComplex B, cwoComplex *C);
GWO_EXPORT void GWOAPI gwoDivCplxArry(cwoCtx *c, gwoCtx *gtx, cwoComplex *A, cwoComplex *B, cwoComplex *C);

GWO_EXPORT void GWOAPI gwoMultComplex(cwoCtx *c, gwoCtx *gtx, void *A, void *B, void *C);
GWO_EXPORT void GWOAPI gwoMultComplexFlt(cwoCtx *c, gwoCtx *gtx, void *A, float B, void *C);
GWO_EXPORT void GWOAPI gwoDiv(cwoCtx *c, gwoCtx *gtx, void* a, float b); // c=a/b

GWO_EXPORT void GWOAPI gwoSqrtReal(cwoCtx *c, gwoCtx *gtx, cwoComplex* a);
GWO_EXPORT void GWOAPI gwoSqrtCplx(cwoCtx *c, gwoCtx *gtx, cwoComplex* a);

GWO_EXPORT void GWOAPI gwoGamma(cwoCtx *c, gwoCtx *gtx, cwoComplex* buf, float gamma);

GWO_EXPORT void GWOAPI gwoRandSeed(cwoCtx *c, gwoCtx *gtx,unsigned long s);
//GWO_EXPORT void GWOAPI gwoSetRandReal(cwoCtx *c, gwoCtx *gtx, cwoComplex *a, float max, float min);
//GWO_EXPORT void GWOAPI gwoRandPhase(cwoCtx *c, gwoCtx *gtx, cwoComplex *a, float max, float min);
//GWO_EXPORT void GWOAPI gwoMulRandPhase(cwoCtx *c, gwoCtx *gtx, cwoComplex *a, float max, float min);
GWO_EXPORT void GWOAPI gwoSetRandReal(cwoCtx *c, gwoCtx *gtx, float *p_rnd, cwoComplex *a, long long int seed, float max, float min);
GWO_EXPORT void GWOAPI gwoSetRandPhase(cwoCtx *c, gwoCtx *gtx, float *p_rnd, cwoComplex *a, long long int seed, float max, float min);
GWO_EXPORT void GWOAPI gwoMulRandPhase(cwoCtx *c, gwoCtx *gtx, float *p_rnd, cwoComplex *a, long long int seed, float max, float min);

GWO_EXPORT void GWOAPI gwoReal2Complex(cwoCtx *c, gwoCtx *gtx, float* src, cwoComplex *dst);
GWO_EXPORT void GWOAPI gwoPhase2Complex(cwoCtx *c, gwoCtx *gtx, float* src, cwoComplex *dst);
GWO_EXPORT void GWOAPI gwoArg2Cplx(cwoCtx *ctx, gwoCtx *gtx,cwoComplex *src,cwoComplex *dst,float scale,float offset);
GWO_EXPORT void GWOAPI gwoPolar(cwoCtx* ctx, gwoCtx *gtx, float *amp, float *ph, cwoComplex *c);
GWO_EXPORT void GWOAPI gwoReIm(cwoCtx* ctx, gwoCtx *gtx, cwoComplex *re, cwoComplex *im, cwoComplex *c);

//GWO_EXPORT void GWOAPI gwoFltRe(cwoCtx* ctx, gwoCtx *gtx,float *a, cwoComplex *b);

//GWO_EXPORT void GWOAPI gwoThreshold(cwoCtx *c, gwoCtx *gtx, float* a, float max, float min);
GWO_EXPORT void GWOAPI gwoThreshold(cwoCtx *c, gwoCtx *gtx, cwoComplex* a, float max, float min);
GWO_EXPORT void GWOAPI gwoPickupFloat(cwoCtx *c, gwoCtx *gtx, float *src, float* pix_p, float pix);
GWO_EXPORT void GWOAPI gwoPickupCplx(cwoCtx *c, gwoCtx *gtx, cwoComplex *src, cwoComplex* pix_p, float pix);


GWO_EXPORT float GWOAPI gwoSumTotal(cwoCtx *c, gwoCtx *gtx, float *a);
GWO_EXPORT float GWOAPI gwoAverage(cwoCtx *c, gwoCtx *gtx, cwoComplex *a);
GWO_EXPORT float GWOAPI gwoVariance(cwoCtx *c, gwoCtx *gtx, cwoComplex *a, float ave);
GWO_EXPORT void GWOAPI gwoVarianceMap(cwoCtx *c, gwoCtx *gtx, float *src, float *dst, int sx, int sy);
GWO_EXPORT float GWOAPI gwoSMDx(cwoCtx *c, gwoCtx *gtx, float *a);


GWO_EXPORT void GWOAPI gwoResizeNearest(cwoCtx *c, gwoCtx *gtx, cwoComplex *p_new, int dNx, int dNy,	cwoComplex *p_old, int sNx, int sNy);
GWO_EXPORT void GWOAPI gwoResizeLinear(cwoCtx *c, gwoCtx *gtx, cwoComplex *p_new, int dNx, int dNy, cwoComplex *p_old, int sNx, int sNy);
GWO_EXPORT void GWOAPI gwoResizeCubic(cwoCtx *c, gwoCtx *gtx, cwoComplex *p_new, int dNx, int dNy, cwoComplex *p_old, int sNx, int sNy);
GWO_EXPORT void GWOAPI gwoResizeLanczos(cwoCtx *c, gwoCtx *gtx, cwoComplex *p_new, int dNx, int dNy, cwoComplex *p_old, int sNx, int sNy);


GWO_EXPORT void GWOAPI gwoErrorDiffusion(cwoCtx *c, gwoCtx *gtx, cwoComplex *p_i, cwoComplex *p_o);


GWO_EXPORT void GWOAPI gwoAddSphericalWave(cwoCtx* ctx, gwoCtx *gtx, cwoComplex *p, float x, float y, float z, float px, float py, float a);
GWO_EXPORT void GWOAPI gwoMulSphericalWave(cwoCtx* ctx, gwoCtx *gtx, cwoComplex *p, float x, float y, float z, float px, float py, float a);
GWO_EXPORT void GWOAPI gwoAddApproxSphWave(cwoCtx *ctx, gwoCtx *gtx,cwoComplex *p, float x, float y, float z, float zx, float zy, float px, float py, float a);
GWO_EXPORT void GWOAPI gwoMulApproxSphWave(cwoCtx *ctx, gwoCtx *gtx,cwoComplex *p, float x, float y, float z, float zx, float zy, float px, float py, float a);


GWO_EXPORT void GWOAPI gwoMaxMin(cwoCtx *c, gwoCtx *gtx, cwoComplex* buf, float *max, float *min);
GWO_EXPORT cwoComplex GWOAPI gwoTotalSum(cwoCtx *c, gwoCtx *gtx, cwoComplex* a);


GWO_EXPORT void GWOAPI gwoScaleReal(
	cwoCtx *ctx, gwoCtx *gtx, cwoComplex *pi, cwoComplex *po, 
	float i1,float i2, float o1, float o2);
GWO_EXPORT void GWOAPI gwoScaleCplx(
	cwoCtx *ctx, gwoCtx *gtx, cwoComplex *pi, cwoComplex *po, 
	float i1,float i2, float o1, float o2);


GWO_EXPORT void GWOAPI gwoRectFillInside(cwoCtx *c, gwoCtx *gtx, cwoComplex *p, int x, int y, int Sx, int Sy, cwoComplex a);
GWO_EXPORT void GWOAPI gwoRectFillOutside(cwoCtx *c, gwoCtx *gtx, cwoComplex *p, int x, int y, int Sx, int Sy, cwoComplex a);

GWO_EXPORT void GWOAPI gwoFloatToChar(cwoCtx *c, gwoCtx *gtx, char *dst, float *src);
GWO_EXPORT void GWOAPI gwoCharToFloat(cwoCtx *c, gwoCtx *gtx, float *dst, char *src);


GWO_EXPORT float GWOAPI gwoMSE(cwoCtx *c, gwoCtx *gtx,cwoComplex *p_tar,cwoComplex *p_ref);


/****************
** For PLS
*****************/
GWO_EXPORT void GWOAPI gwoPLSFresnel(cwoCtx* c, gwoCtx *gtx, gwoObjPoint *obj, cwoComplex *cgh, float ph);
GWO_EXPORT void GWOAPI gwoPLSCGHFresnel(cwoCtx* c, gwoCtx *gtx, gwoObjPoint *obj, float *cgh, float ph);

/****************
** For Direct display using OpenGL
*****************/
/*
GWO_EXPORT void GWOAPI gwoDirectDispPhase(uchar4 *p_disp, float2 *p_a, int Nx, int Ny);
GWO_EXPORT void GWOAPI gwoDirectDispFloat(uchar4 *p_disp, float *p_a, int Nx, int Ny);
GWO_EXPORT void GWOAPI gwoDirectDispFloatRGB(uchar4 *p_disp, float *p_r, float *p_g, float *p_b, int Nx, int Ny);
*/



	/****************
** Test code
*****************/

GWO_EXPORT void GWOAPI gwoArbitFresnelDirect(
	cwoCtx *c, gwoCtx *gtx, cwoComplex *p1, cwoComplex *p2, 
	cwoFloat2 *p_x1, cwoFloat2 *p_x2, 
	float *p_d1, float *p_d2);

GWO_EXPORT void GWOAPI gwoArbitFresnelCoeff(
	cwoCtx* c, gwoCtx *gtx, cwoComplex *p, cwoFloat2 *p_x2, float *p_d2);



#ifdef __cplusplus
}
#endif

#endif


/////////////////////////////

