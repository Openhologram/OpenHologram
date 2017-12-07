// Copyright (C) Tomoyoshi Shimobaba 2011-


#ifndef _GWO_H
#define _GWO_H

#include "cwo.h"
#include "cwoCu.h"
#include "gwo_lib.h"
//#ifndef _GWO_LIB_H
//#define _GWO_LIB_H
//#include "gwo_lib.h"

class GWO : public CWO
{
	int stream_mode;
	float *p_rnd;

public:
	//cwoStream cstream;
	gwoCtx gtx;

	GWO();
	~GWO();

	GWO(GWO &tmp);
	GWO(int Nx, int Ny, int Nz = 1);//!< constructor
	
	void SetDev(int dev);
	void SetThreads(int Nx, int Ny);
	int GetThreadsX();
	int GetThreadsY();

	//void Create(int dev, int Nx, int Ny);	
	//int Create(int Nx, int Ny=1, int Nz=1); 
	void Delete();

	void Send(CWO &a);
	void Recv(CWO &a);

	void CreateStream();
	cwoStream GetStream();
	void DestroyStream();
	void SyncStream();

/*	void SetMemcpyFlag(int i=0);
	int GetMemcpyFlag();
	void SetPagelockFlag(int i=0);
	int GetPagelockFlag();
*/	void SetStreamMode(int mode);
	int GetStreamMode();

	int Load(char* fname, int c=CWO_GREY);
	int Load(char* fname_amp, char *fname_pha, int c=CWO_GREY);
	int Save(char* fname, int bmp_8_24=24);
	int Save(char* fname, CWO *r, CWO *g=NULL, CWO *b=NULL);
	//int SaveMonosToColor(char* fname, char *r_name, char *g_name, char *b_name);
	int SaveAsImage(char* fname, float i1, float i2, float o1, float o2, int flag=CWO_SAVE_AS_RE);
	int SaveAsImage(char* fname, int flag=CWO_SAVE_AS_RE, CWO *r=NULL, CWO *g=NULL, CWO *b=NULL);
	int SaveAsImage(cwoComplex *p, int Nx, int Ny, char* fname, int flag=CWO_SAVE_AS_RE, int bmp_8_24=24);
	
public:
	void* __Malloc(size_t size);
	void __Free(void **a);
	void __Memcpy(void *dst, void *src, size_t size);
	void __Memset(void *p, int c, size_t  size);

	void Fill(cwoComplex pix);

	/*void __Expand(void *src, int srcNx, int srcNy, 
						void *dst, int dstNx, int dstNy, int type);*/
	void __Expand(
		void *src, int sx, int sy, int srcNx, int srcNy, 
		void *dst, int dx, int dy, int dstNx, int dstNy,
		int type);


	void __ShiftedFresnelAperture(cwoComplex *a);
	void __ShiftedFresnelProp(cwoComplex *a);
	void __ShiftedFresnelCoeff(cwoComplex *a);

	void __ARSSFresnelAperture(cwoComplex *a);
	void __ARSSFresnelProp(cwoComplex *a);
	void __ARSSFresnelCoeff(cwoComplex *a);

	void __FresnelConvProp(cwoComplex *a);
	void __FresnelConvCoeff(cwoComplex *a, float const_val=1.0f);

	void __AngularProp(cwoComplex *a, int flag);
	//void __ShiftedAngularProp(cwoComplex *a);

	void __HuygensProp(cwoComplex *a);
//	void __HuygensCoeff(cwoComplex *a);

	void __FresnelFourierProp(cwoComplex *a);
	void __FresnelFourierCoeff(cwoComplex *a);
	
	void __FresnelDblAperture(cwoComplex *a, float z1);
	void __FresnelDblFourierDomain(cwoComplex *a, float z1, float z2, cwoInt4 *zp);
	void __FresnelDblCoeff(cwoComplex *a, float z1, float z2);

	void __FFT(void *src, void *dst, int type);
	void __IFFT(void *src, void *dst);
	void __FFTShift(void *src);

	void __NUFFT_T1(cwoComplex *p_fld, cwoFloat2 *p_x, int R=2, int Msp=12);
	void __NUFFT_T2(cwoComplex *p_fld, cwoFloat2 *p_x, int R=2, int Msp=12);

void __Add(cwoComplex *a, cwoComplex b, cwoComplex *c);//c=a+b
void __Add(cwoComplex *a, cwoComplex *b, cwoComplex *c);//c=a+b
void __Sub(cwoComplex *a, cwoComplex b, cwoComplex *c);//c=a-b
void __Sub(cwoComplex *a, cwoComplex *b, cwoComplex *c);//c=a-b
void __Mul(cwoComplex *a, cwoComplex b, cwoComplex *c);//c=a*b
void __Mul(cwoComplex *a, cwoComplex *b, cwoComplex *c);//c=a*b
void __Div(cwoComplex *a, cwoComplex b, cwoComplex *c);//c=a/b
void __Div(cwoComplex *a, cwoComplex *b, cwoComplex *c);//c=a/b

	void __AddSphericalWave(cwoComplex *p, float x, float y, float z, float px, float py, float a);
	void __MulSphericalWave(cwoComplex *p, float x, float y, float z, float px, float py, float a);
	void __AddApproxSphWave(cwoComplex *p, float x, float y, float z, float zx, float zy, float px, float py, float a);
	void __MulApproxSphWave(cwoComplex *p, float x, float y, float z, float zx, float zy, float px, float py, float a);

	void __Re(cwoComplex*a , cwoComplex *b);
	void __Im(cwoComplex*a , cwoComplex *b);
	void __Intensity(cwoComplex*a , cwoComplex *b);
	void __Amp(cwoComplex*a , cwoComplex *b);
	void __Phase(cwoComplex*a , cwoComplex *b, float offset);
	void __Arg(cwoComplex *a , cwoComplex *b, float scale, float offset);
	void __Real2Complex(float *src, cwoComplex *dst);
	void __Phase2Complex(float *src, cwoComplex *dst);
	void __Arg2Cplx(cwoComplex *src, cwoComplex *dst, float scale, float offset);
	void __Polar(float *amp, float *ph, cwoComplex *c);
	void __ReIm(cwoComplex *re, cwoComplex *im, cwoComplex *c);
//	void __Gamma(float *src, float gamma);
	void __RectFillInside(cwoComplex *p, int x, int y, int Sx, int Sy, cwoComplex a);
	void __RectFillOutside(cwoComplex *p, int x, int y, int Sx, int Sy, cwoComplex a);

	void __FloatToChar(char *dst, float *src, int N);
	void __CharToFloat(float *dst, char *src, int N);

	void __Copy(
			cwoComplex *src, int x1, int y1, int sNx, int sNy,
			cwoComplex *dst, int x2, int y2, int dNx, int dNy, 
			int Sx, int Sy);

	void FFTShift();

	void SqrtReal();
	void SqrtCplx();

	void Gamma(float g);
	void Threshold(float max, float min=0.0);
	void __PickupFloat(float *src, float *pix_p, float pix);
	void __PickupCplx(cwoComplex *src, cwoComplex *pix_p, float pix);

	float Average();
	float Variance();


	void SetRandSeed(long long s);
	void RandReal(float max = 1.0f, float min = 0.0f);
	void __RandPhase(cwoComplex *a, float max, float min);
	void __MulRandPhase(cwoComplex *a, float max, float min);


	void __MaxMin(cwoComplex *a, float *max, float *min, int *max_x=NULL, int *max_y=NULL,int *min_x=NULL, int *min_y=NULL);
	//int MaxMin(float *a, float *max, float *min, int *max_x=NULL, int *max_y=NULL,int *min_x=NULL, int *min_y=NULL);
	//int MaxMin(float *max, float *min, int *max_x=NULL, int *max_y=NULL,int *min_x=NULL, int *min_y=NULL);


	cwoComplex TotalSum();



	int __ScaleReal(float i1, float i2, float o1, float o2);
	int __ScaleCplx(float i1, float i2, float o1, float o2);


	void __ResizeNearest(
		cwoComplex *p_new, int dNx, int dNy, cwoComplex *p_old, int sNx, int sNy);
	void __ResizeLinear(
		cwoComplex *p_new, int dNx, int dNy, cwoComplex *p_old, int sNx, int sNy);
	void __ResizeCubic(
		cwoComplex *p_new, int dNx, int dNy, cwoComplex *p_old, int sNx, int sNy);
	void __ResizeLanczos(
		cwoComplex *p_new, int dNx, int dNy, cwoComplex *p_old, int sNx, int sNy);


	void ErrorDiffusion(CWO *a = NULL, int flag = CWO_ED_FS);
	void ErrorDiffusionSegmented(CWO *a = NULL, int flag = CWO_ED_FS);

	float MSE(CWO &ref);


////////////////////////
//Test code
////////////////////////
virtual void __ArbitFresnelDirect(
	cwoComplex *p1, cwoComplex *p2, 
	cwoFloat2 *p_x1, cwoFloat2 *p_x2, 
	float *p_d1, float *p_d2);

virtual void __ArbitFresnelCoeff(
	cwoComplex *p, cwoFloat2 *p_x2, float *p_d2);

	void test();
	
};


#endif
