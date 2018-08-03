// Copyright (C) Tomoyoshi Shimobaba 2011-

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <ctype.h>

#include <iostream>
#include <vector>
#include <string>

#ifndef __linux__
#include <windows.h>
#endif

#ifdef _OPENMP
#include<omp.h>
#endif

#include <limits.h>
#include <float.h>


#include "cwo.h"
//#include "CImg.h"




CWO::CWO()
{
	cwoInitFFT();

	InitParams();
	InitBuffers();
		
}


CWO::CWO(int Nx, int Ny, int Nz)
{
	cwoInitFFT();

	InitParams();
	InitBuffers();

	SetSize(Nx,Ny,Nz);
	p_field=(cwoComplex*)__Malloc(Nx*Ny*sizeof(cwoComplex));
}

CWO::~CWO()
{
	if(p_field!=NULL){
		free(p_field);
		p_field=NULL;
	}
	if(p_diff_a!=NULL){
		free(p_diff_a);
		p_diff_a=NULL;
	}
	if(p_diff_b!=NULL){
		free(p_diff_b);
		p_diff_b=NULL;
	}


	if(fre_s_tbl!=NULL) delete []fre_s_tbl;
	if(fre_c_tbl!=NULL) delete []fre_c_tbl;

}

void CWO::InitParams()
{

	SetFieldType(CWO_FLD_COMPLEX);
	SetSize(0,0,1);
	SetWaveLength((float)633.0e-9);
	SetPitch((float)10.0e-6,(float)10.0e-6,(float)10.0e-6);
	SetDstPitch((float)10.0e-6,(float)10.0e-6,(float)10.0e-6);
	SetOffset(0,0);
	SetDstOffset(0,0);
	SetPropDist((float)0.1);
	SetCalcType(-1);
	
	//SetThreads(1);

	memset(&prev_ctx,0,sizeof(cwoCtx));
	//prev_ctx=0;
}


void CWO::InitBuffers()
{
	p_field=NULL;
	p_diff_a=NULL;
	p_diff_b=NULL;
	fre_s_tbl=NULL;
	fre_c_tbl=NULL;

}

void CWO::FreeBuffers()
{

	__Free((void**)&p_field);
	__Free((void**)&p_diff_a);
	__Free((void**)&p_diff_b);
	__Free((void**)&fre_s_tbl);
	__Free((void**)&fre_c_tbl);

}

void CWO::SetThreads(int Nx, int Ny)
{
	//ctx.nthread_x=Nx;
	//ctx.nthread_y=Ny;
	cwoSetThreads(Nx);
}
int CWO::GetThreads()
{
	//return ctx.nthread_x;
	return cwoGetThreads();
}/*
int CWO::GetThreadsX()
{
	return ctx.nthread_x;
}
int CWO::GetThreadsY()
{
	return ctx.nthread_y;
}*/


CWO::CWO(CWO &tmp)
{
	//copy constructor	
	InitParams();
	InitBuffers();

	ctx=tmp.ctx;
	
	size_t size=GetMemSizeCplx();
	p_field=(cwoComplex*)__Malloc(size);

	__Memcpy(GetBuffer(),tmp.GetBuffer(),size);	
}


CWO& CWO::operator=(CWO &tmp) 
{
	size_t size;

	InitParams();
	FreeBuffers();

	SetFieldType(tmp.GetFieldType());

	if(GetNx()!=tmp.GetNx() || GetNy()!=tmp.GetNy() || GetNz()!=tmp.GetNz() ||
		GetBuffer()==NULL){
		
		SetSize(tmp.GetNx(),tmp.GetNy(),tmp.GetNz());	

		//calculate new memory size
		size=GetMemSizeCplx();

		__Free((void**)&p_field);
		
		p_field=__Malloc(size);

	}
	else{
		
		SetSize(tmp.GetNx(),tmp.GetNy(),tmp.GetNz());	
		//calculate new memory size
		if(GetFieldType()==CWO_FLD_COMPLEX)
			size=GetMemSizeCplx();
		else
			size=GetMemSizeFloat();
	
	}
	__Memcpy(GetBuffer(),tmp.GetBuffer(),size);	
	
	//copy context
	ctx=tmp.ctx;

	return *this;
}

CWO CWO::operator+(CWO &a)
{
	CWO *tmp=NULL;
	tmp=(CWO*)new CWO;
	tmp->Create(GetNx(),GetNy());
	(*tmp).ctx=ctx;
	//(*tmp)=(*this);
	__Add((cwoComplex*)GetBuffer(),(cwoComplex*)a.GetBuffer(),(cwoComplex*)tmp->GetBuffer());
	return (*tmp);
}
CWO CWO::operator+(float a)
{
	CWO *tmp=NULL;
	tmp=(CWO*)new CWO;
	tmp->Create(GetNx(),GetNy());
	(*tmp).ctx=ctx;
	__Add((cwoComplex*)GetBuffer(),a,(cwoComplex*)tmp->GetBuffer());
	return (*tmp);
}
CWO CWO::operator+(cwoComplex a)
{
	CWO *tmp=NULL;
	tmp=(CWO*)new CWO;
	tmp->Create(GetNx(),GetNy());
	(*tmp).ctx=ctx;
	__Add((cwoComplex*)GetBuffer(),a,(cwoComplex*)tmp->GetBuffer());
	return (*tmp);
}

CWO CWO::operator-(CWO &a)
{
	CWO *tmp=NULL;
	tmp=(CWO*)new CWO;
	tmp->Create(GetNx(),GetNy());
	(*tmp).ctx=ctx;
	//(*tmp)=(*this);
	__Sub((cwoComplex*)GetBuffer(),(cwoComplex*)a.GetBuffer(),(cwoComplex*)tmp->GetBuffer());
	return (*tmp);
}
CWO CWO::operator-(float a)
{
	CWO *tmp=NULL;
	tmp=(CWO*)new CWO;
	tmp->Create(GetNx(),GetNy());
	(*tmp).ctx=ctx;
	//(*tmp)=(*this);
	__Sub((cwoComplex*)GetBuffer(),a,(cwoComplex*)tmp->GetBuffer());
	return (*tmp);
}
CWO CWO::operator-(cwoComplex a)
{
	CWO *tmp=NULL;
	tmp=(CWO*)new CWO;
	tmp->Create(GetNx(),GetNy());
	(*tmp).ctx=ctx;
	//(*tmp)=(*this);
	__Sub((cwoComplex*)GetBuffer(),a,(cwoComplex*)tmp->GetBuffer());
	return (*tmp);
}
CWO CWO::operator*(CWO &a)
{
	CWO *tmp=NULL;
	tmp=(CWO*)new CWO;
	tmp->Create(GetNx(),GetNy());
	(*tmp).ctx=ctx;
	//(*tmp)=(*this);
	__Mul((cwoComplex*)GetBuffer(),(cwoComplex*)a.GetBuffer(),(cwoComplex*)tmp->GetBuffer());
	return (*tmp);
}
CWO CWO::operator*(float a)
{
	CWO *tmp=NULL;
	tmp=(CWO*)new CWO;
	tmp->Create(GetNx(),GetNy());
	(*tmp).ctx=ctx;
	//(*tmp)=(*this);
	__Mul((cwoComplex*)GetBuffer(),a,(cwoComplex*)tmp->GetBuffer());

	return (*tmp);
}
CWO CWO::operator*(cwoComplex a)
{
	CWO *tmp=NULL;
	tmp=(CWO*)new CWO;
	tmp->Create(GetNx(),GetNy());
	(*tmp).ctx=ctx;
	//(*tmp)=(*this);
	__Mul((cwoComplex*)GetBuffer(),a,(cwoComplex*)tmp->GetBuffer());

	return (*tmp);
}
CWO CWO::operator/(CWO &a)
{
	CWO *tmp=NULL;
	tmp=(CWO*)new CWO;
	tmp->Create(GetNx(),GetNy());
	(*tmp).ctx=ctx;
	//(*tmp)=(*this);
	__Div((cwoComplex*)GetBuffer(),(cwoComplex*)a.GetBuffer(),(cwoComplex*)tmp->GetBuffer());
	return (*tmp);
}
CWO CWO::operator/(float a)
{
	CWO *tmp=NULL;
	tmp=(CWO*)new CWO;
	tmp->Create(GetNx(),GetNy());
	(*tmp).ctx=ctx;
	//(*tmp)=(*this);
	__Div((cwoComplex*)GetBuffer(),a,(cwoComplex*)tmp->GetBuffer());
	return (*tmp);
}
CWO CWO::operator/(cwoComplex a)
{
	CWO *tmp=NULL;
	tmp=(CWO*)new CWO;
	tmp->Create(GetNx(),GetNy());
	(*tmp).ctx=ctx;
	//(*tmp)=(*this);
	__Div((cwoComplex*)GetBuffer(),a,(cwoComplex*)tmp->GetBuffer());
	return (*tmp);
}

CWO& CWO::operator+=(CWO &a)
{
	cwoComplex *src=(cwoComplex *)a.GetBuffer();
	cwoComplex *dst=(cwoComplex *)GetBuffer();

	__Add(dst,src,dst);
		
	return *this;
}
CWO& CWO::operator+=(float a)
{

	cwoComplex *dst=(cwoComplex *)GetBuffer();
	__Add(dst,a,dst);

	return *this;
}
CWO& CWO::operator+=(cwoComplex a)
{
	Cplx();
	cwoComplex *dst=(cwoComplex *)GetBuffer();

	__Add(dst,a,dst);
		
	return *this;
}


CWO& CWO::operator-=(CWO &a)
{

	cwoComplex *dst=(cwoComplex *)GetBuffer();
	cwoComplex *src=(cwoComplex *)a.GetBuffer();

	__Sub(dst,src,dst);

	return *this;
}
CWO& CWO::operator-=(float a)
{
	cwoComplex *dst=(cwoComplex *)GetBuffer();
	__Sub(dst,a,dst);

	return *this;
}
CWO& CWO::operator-=(cwoComplex a)
{
	Cplx();
	cwoComplex *dst=(cwoComplex *)GetBuffer();
		
	__Sub(dst,a,dst);
		
	return *this;
}

CWO& CWO::operator*=(CWO &a)
{
	int fld1=GetFieldType();
	int fld2=a.GetFieldType();

	Cplx();
	a.Cplx();
	
	cwoComplex *dst=(cwoComplex *)GetBuffer();
	cwoComplex *src=(cwoComplex *)a.GetBuffer();
	
	__Mul(dst,src,dst);


	return *this;
}

CWO& CWO::operator*=(float a)
{
	cwoComplex *dst=(cwoComplex *)GetBuffer();
	__Mul(dst,a,dst);
		
	return *this;
}
CWO& CWO::operator*=(cwoComplex a)
{
	Cplx();
	cwoComplex *dst=(cwoComplex *)GetBuffer();
	__Mul(dst,a,dst);

	return *this;
}



CWO& CWO::operator/=(CWO &a)
{
	int fld1=GetFieldType();
	int fld2=a.GetFieldType();

	Cplx();
	a.Cplx();
	
	cwoComplex *dst=(cwoComplex *)GetBuffer();
	cwoComplex *src=(cwoComplex *)a.GetBuffer();
	
	__Div(dst,src,dst);

	return *this;
}

CWO& CWO::operator/=(float a)
{
	cwoComplex *dst=(cwoComplex *)GetBuffer();
	__Div(dst,a,dst);

	return *this;
}

CWO& CWO::operator/=(cwoComplex a)
{
	Cplx();
	cwoComplex *dst=(cwoComplex *)GetBuffer();
	__Div(dst,a,dst);

	return *this;
}

void CWO::Send(CWO &a)
{
	*this=a;
}
void CWO::Recv(CWO &a)
{
	a=*this;
}

int CWO::Create(int Nx, int Ny, int Nz)
{
	if(GetBuffer()==NULL || GetNx()!=Nx || GetNy()!=Ny || GetNz()!=Nz){
		
		SetFieldType(CWO_FLD_COMPLEX);
		SetSize(Nx,Ny,Nz);	
		//calculate new memory size
		size_t size=GetMemSizeCplx();
		__Free((void**)&p_field);
		p_field=__Malloc(size);
		
		if(p_field==NULL) return -1;
	}

	Clear();

	return 0;

}

void CWO::Destroy()
{

	__Free((void**)&p_field);
	__Free((void**)&p_diff_a);//temporary buffer for diffraction (convolution type)
	__Free((void**)&p_diff_b);//temporary buffer for diffraction (convolution type)

}


void CWO::Attach(CWO &a)
{
	ctx=a.ctx;

	p_field=a.p_field;
	p_diff_a=a.p_diff_a;
	p_diff_b=a.p_diff_b;

	
}

#ifdef _WIN32 
LARGE_INTEGER nFreq, nBefore, nAfter;
#else
double nBefore, nAfter; 
#endif 
void CWO::SetTimer() 
{ 
    #ifdef _WINDOWS_ 
        memset(&nFreq,   0x00, sizeof nFreq); 
        memset(&nBefore, 0x00, sizeof nBefore); 
        memset(&nAfter,  0x00, sizeof nAfter); 
        QueryPerformanceFrequency(&nFreq); 
        QueryPerformanceCounter(&nBefore);    
    #else
        struct timeval tv;
        gettimeofday(&tv, NULL);
        nBefore=tv.tv_sec + (double)tv.tv_usec*1e-6; 
    #endif 
 } 

double CWO::EndTimer() 
{ 
    #ifdef _WINDOWS_ 
        float dwTime; 
        QueryPerformanceCounter(&nAfter); 
        dwTime = (float)((nAfter.QuadPart - nBefore.QuadPart) * 1000 / nFreq.QuadPart); 
        return dwTime;
    #else
        struct timeval tv;
        gettimeofday(&tv, NULL);
        nAfter=tv.tv_sec + (double)tv.tv_usec*1e-6;
        return (nAfter - nBefore)*1e+3; 
    #endif 
}
void* CWO::__Malloc(size_t size)
{
	return malloc(size);
}
void CWO::__Free(void **p)
{
	if(*p!=NULL) free(*p);
	*p=NULL;
}
void CWO::__Memcpy(void *dst, void *src, size_t size)
{
	memcpy(dst,src,size);

}
void CWO::__Memset(void *p, int c, size_t size)
{
	memset(p, c, size);
}

void CWO::__Expand(
	void *src, int sx, int sy, int srcNx, int srcNy, 
	void *dst, int dx, int dy, int dstNx, int dstNy,
	int type)
{
	//Nz
	cwoExpand(&ctx,
		src,sx,sy,srcNx,srcNy,
		dst,dx,dy,dstNx,dstNy,type);
	SetSize(dstNx,dstNy);
}

void CWO::__ShiftedFresnelAperture(cwoComplex *a)
{
	cwoShiftedFresnelAperture(&ctx,a);
}
void CWO::__ShiftedFresnelProp(cwoComplex *a)
{
	cwoShiftedFresnelProp(&ctx, a);
}
void CWO::__ShiftedFresnelCoeff(cwoComplex *a)
{
	cwoShiftedFresnelCoeff(&ctx,a);
}

void CWO::__ARSSFresnelAperture(cwoComplex *a)
{
	cwoARSSFresnelAperture(&ctx,a);
}
void CWO::__ARSSFresnelProp(cwoComplex *a)
{
	cwoARSSFresnelProp(&ctx, a);
}
void CWO::__ARSSFresnelCoeff(cwoComplex *a)
{
	cwoARSSFresnelCoeff(&ctx,a);
}

void CWO::__FresnelConvProp(cwoComplex *a)
{
	cwoFresnelConvProp(&ctx, a);
}

void CWO::__FresnelConvCoeff(cwoComplex *a, float const_val)
{
	cwoFresnelConvCoeff(&ctx,a, const_val);
}

void CWO::__FresnelAnalysisTransfer(cwoComplex *a, cwoComplex *b)
{
	cwoFresnelAnalysisTransfer(&ctx, a, b);
}

void CWO::__AngularProp(cwoComplex *a, int flag)
{
	switch(flag){
		case CWO_PROP_CENTER:
			cwoAngularProp(&ctx,a);		
			break;
		case CWO_PROP_FFT_SHIFT:
			cwoAngularPropFS(&ctx,a);		
			break;
		case CWO_PROP_MUL_CENTER:
			cwoAngularPropMul(&ctx,a);	
			break;
		case CWO_PROP_MUL_FFT_SHIFT:
			cwoAngularPropMulFS(&ctx,a);	
			break;
		default:
			cwoAngularProp(&ctx,a);		
	}
			 
		 //cwoAngularProp(&ctx, a/*, px, py*/);
}

void CWO::__AngularLim(float *fx_c, float *fx_w, float *fy_c, float *fy_w)
{
	cwoAngularLim(&ctx, fx_c, fx_w, fy_c, fy_w);

}
/*
void CWO::__ShiftedAngularProp(cwoComplex *a)
{
	cwoShiftedAngularProp(&ctx,a);
}*/

void CWO::__HuygensProp(cwoComplex *a)
{
	cwoHuygensProp(&ctx, a);

}
void CWO::__FresnelFourierProp(cwoComplex *a)
{
	cwoFresnelFourierProp(&ctx, a);
}
void CWO::__FresnelFourierCoeff(cwoComplex *a)
{
	cwoFresnelFourierCoeff(&ctx, a);
}

void CWO::__FresnelDblAperture(cwoComplex *a, float z1)
{
	cwoFresnelDblAperture(&ctx, a, z1);
}
void CWO::__FresnelDblFourierDomain(cwoComplex *a, float z1, float z2, cwoInt4 *zp)
{
	cwoFresnelDblFourierDomain(&ctx, a, z1, z2, zp);
}
void CWO::__FresnelDblCoeff(cwoComplex *a, float z1, float z2)
{
	cwoFresnelDblCoeff(&ctx, a, z1, z2);
}

void CWO::__FFT(void *src, void *dst, int type)
{
	cwoFFT(&ctx, src, dst, type);
}
void CWO::__IFFT(void *src, void *dst)
{
	cwoIFFT(&ctx, src, dst);
}
void CWO::__FFTShift(void *a)
{
	cwoFFTShift(&ctx,a);

}

void CWO::__Add(cwoComplex *a, cwoComplex b, cwoComplex *c)
{
#ifdef _OPENMP
	omp_set_num_threads(GetThreads());
	#pragma omp parallel for	
#endif
	for(long long int i=0;i<GetNx()*GetNy()*GetNz();i++){
		c[i]=a[i]+b;
	}
}

void CWO::__Add(cwoComplex *a, float b, cwoComplex *c)
{
	cwoComplex tmp;
	CWO_RE(tmp)=b;
	CWO_IM(tmp)=0.0f;
	__Add(a,tmp,a);
}

void CWO::__Add(cwoComplex *a, cwoComplex *b, cwoComplex *c)
{
#ifdef _OPENMP
	omp_set_num_threads(GetThreads());
	#pragma omp parallel for	
#endif
	for(long long int i=0;i<GetNx()*GetNy()*GetNz();i++){
		CWO_RE(c[i])=CWO_RE(a[i])+CWO_RE(b[i]);
		CWO_IM(c[i])=CWO_IM(a[i])+CWO_IM(b[i]);
	}
}

void CWO::__Sub(cwoComplex *a, cwoComplex b, cwoComplex *c)
{
#ifdef _OPENMP
	omp_set_num_threads(GetThreads());
	#pragma omp parallel for
#endif
	for(long long int i=0;i<GetNx()*GetNy()*GetNz();i++){
		c[i]=a[i]-b;
	}

}
void CWO::__Sub(cwoComplex *a, float b, cwoComplex *c)
{
	cwoComplex tmp;
	CWO_RE(tmp)=b;
	CWO_IM(tmp)=0.0f;
	__Sub(a,tmp,a);
}

void CWO::__Sub(cwoComplex *a, cwoComplex *b, cwoComplex *c)
{
#ifdef _OPENMP
	omp_set_num_threads(GetThreads());
	#pragma omp parallel for	
#endif	
	for(long long int i=0;i<GetNx()*GetNy()*GetNz();i++){
		CWO_RE(c[i])=CWO_RE(a[i])-CWO_RE(b[i]);
		CWO_IM(c[i])=CWO_IM(a[i])-CWO_IM(b[i]);
	}
}

void CWO::__Mul(cwoComplex *a, cwoComplex *b, cwoComplex *c)
{
	cwoMultComplex(&ctx, a, b, c);

}

void CWO::__Mul(cwoComplex *a, float b, cwoComplex *c)
{
	cwoComplex tmp;
	CWO_RE(tmp)=b;
	CWO_IM(tmp)=0.0f;
	__Mul(a,tmp,c);

}

void CWO::__Mul(cwoComplex *a, cwoComplex b, cwoComplex *c)
{
#ifdef _OPENMP
	omp_set_num_threads(GetThreads());
	#pragma omp parallel for
#endif
	for(long long int i=0;i<GetNx()*GetNy()*GetNz();i++){
		c[i]=a[i]*b;
	}

}

void CWO::__Div(cwoComplex *a, cwoComplex b, cwoComplex *c)
{
#ifdef _OPENMP
	omp_set_num_threads(GetThreads());
	#pragma omp parallel for
#endif
	for(long long int i=0;i<GetNx()*GetNy()*GetNz();i++){
		c[i]=a[i]/b;
	}
}


void CWO::__Div(cwoComplex *a, float b, cwoComplex *c)
{
	cwoComplex tmp;
	CWO_RE(tmp)=b;
	CWO_IM(tmp)=0.0f;
	__Div(a,tmp,c);
	
}

void CWO::__Div(cwoComplex *a, cwoComplex *b, cwoComplex *c)
{
#ifdef _OPENMP
	omp_set_num_threads(GetThreads());
	#pragma omp parallel for
#endif
	for(long long int i=0;i<GetNx()*GetNy()*GetNz();i++){
		c[i]=a[i]/b[i];
	}
}


void CWO::__Re(cwoComplex*a , cwoComplex *b)
{
#ifdef _OPENMP
	omp_set_num_threads(GetThreads());
	#pragma omp parallel for
#endif
	for(long long int i=0;i<GetNx()*GetNy()*GetNz();i++){
		CWO_RE(b[i])=CWO_RE(a[i]);
		CWO_IM(b[i])=0.0f;
	}

}
void CWO::__Im(cwoComplex *a , cwoComplex *b)
{
#ifdef _OPENMP
	omp_set_num_threads(GetThreads());
	#pragma omp parallel for
#endif
	for(long long int i=0;i<GetNx()*GetNy()*GetNz();i++){
		CWO_RE(b[i])=CWO_IM(a[i]);
		CWO_IM(b[i])=0.0f;
	}

}
void CWO::__Conj(cwoComplex *a)
{
#ifdef _OPENMP
	omp_set_num_threads(GetThreads());
	#pragma omp parallel for
#endif
	for(long long int i=0;i<GetNx()*GetNy()*GetNz();i++) 
		CWO_IM(a[i])*=-1.0f;
}


void CWO::__Intensity(cwoComplex* a , cwoComplex *b)
{
	cwoIntensity(&ctx, a, b);
}
void CWO::__Amp(cwoComplex *a , cwoComplex *b)
{
	cwoAmp(&ctx, a, b);

}
void CWO::__Phase(cwoComplex *a, cwoComplex *b, float offset)
{
	cwoPhase(&ctx,a, b, offset);
}
void CWO::__Arg(cwoComplex *a, cwoComplex *b, float offset)
{
	cwoArg(&ctx,a, b, offset);
}

void CWO::__Real2Complex(float *src, cwoComplex *dst)
{
#ifdef _OPENMP
	omp_set_num_threads(GetThreads());
	#pragma omp parallel for schedule(dynamic)	
#endif
	for(long long int i=GetNx()*GetNy()-1;i>=0;i--){
		CWO_RE(dst[i])=src[i];
		CWO_IM(dst[i])=0.0;
	}
}
void CWO::__Phase2Complex(float *src, cwoComplex *dst)
{
#ifdef _OPENMP
	omp_set_num_threads(GetThreads());
	#pragma omp parallel for schedule(dynamic)	
#endif
	for(long long int i=GetNx()*GetNy()-1;i>=0;i--){
		CWO_RE(dst[i])=cos(src[i]);
		CWO_IM(dst[i])=sin(src[i]);
	}
}

void CWO::__Arg2Cplx(cwoComplex *src, cwoComplex *dst, float scale, float offset)
{
#ifdef _OPENMP
	omp_set_num_threads(GetThreads());
	#pragma omp parallel for schedule(dynamic)	
#endif
	for(long long int i=GetNx()*GetNy()-1;i>=0;i--){
		float ph=CWO_RE(src[i])*scale+offset;
		CWO_RE(dst[i])=cos(ph);
		CWO_IM(dst[i])=sin(ph);
	}
}

void CWO::__Polar(float *amp, float *ph, cwoComplex *c)
{
#ifdef _OPENMP
	omp_set_num_threads(GetThreads());
	#pragma omp parallel for schedule(dynamic)	
#endif
	for(long long int i=GetNx()*GetNy()-1;i>=0;i--){
		c[i]=Polar(amp[i],ph[i]);
	}

}
void CWO::__ReIm(cwoComplex *re, cwoComplex *im, cwoComplex *c)
{
#ifdef _OPENMP
	omp_set_num_threads(GetThreads());
	#pragma omp parallel for schedule(dynamic)	
#endif
	for(long long int i=GetNx()*GetNy()-1;i>=0;i--){
		CWO_RE(c[i])=CWO_RE(re[i]);
		CWO_IM(c[i])=CWO_RE(im[i]);
	}

}

void CWO::SetSize(int Nx, int Ny, int Nz)
{
	ctx.Nx=Nx;
	ctx.Ny=Ny;
	ctx.Nz=Nz;
}
void CWO::SetCalcType(int type)
{
	ctx.calc_type=type;
}
void CWO::SetPitch(float px, float py, float pz)
{
	//SetSrcPitch(px,py,pz);
	SetSrcPitch(px,py,pz);
	SetDstPitch(px,py,pz);
}
void CWO::SetPitch(float p)
{
	SetSrcPitch(p,p,p);
	SetDstPitch(p,p,p);
}
void CWO::SetSrcPitch(float px, float py, float pz)
{
	ctx.src_px=px;
	ctx.src_py=py;
	ctx.src_pz=pz;
}
void CWO::SetSrcPitch(float p)
{
	SetSrcPitch(p,p,p);
}
void CWO::SetDstPitch(float px, float py, float pz)
{
	ctx.dst_px=px;
	ctx.dst_py=py;
	ctx.dst_pz=pz;
}
void CWO::SetDstPitch(float p)
{
	SetDstPitch(p,p,p);
}
void CWO::SetWaveLength(float w)
{
	ctx.wave_length=w;
}
void CWO::SetOffset(float x, float y, float z)
{
	SetSrcOffset(x,y,z);
	SetDstOffset(x,y,z);

}
void CWO::SetSrcOffset(float x, float y, float z)
{
	ctx.src_ox=x;
	ctx.src_oy=y;
	ctx.src_oz=z;

}
void CWO::SetDstOffset(float x, float y, float z)
{

	ctx.dst_ox=x;
	ctx.dst_oy=y;
	ctx.dst_oz=z;

}
void CWO::SetPropDist(float z)
{
	ctx.z=z;
}
float CWO::GetPropDist()
{
	return ctx.z;
}
void CWO::SetFieldType(int type)
{
	ctx.field_type=type;
}


void* CWO::GetBuffer(int flag)
{
	switch(flag){
		case CWO_BUFFER_FIELD:
			return p_field;
			break;
		case CWO_BUFFER_DIFF_A:
			return p_diff_a;
			break;
		case CWO_BUFFER_DIFF_B:
			return p_diff_b;
			break;

		case CWO_BUFFER_PLS:
			return p_pnt;
			break;
		default:
			return p_field;
	}
}

int CWO::GetFieldType()
{
	return ctx.field_type;
}
char* CWO::GetFieldName()
{
	char *name[]={"CWO_FLD_COMPLEX","CWO_FLD_INTENSITY","CWO_FLD_PHASE","CWO_FLD_CHAR"};
	return name[GetFieldType()];
}
size_t CWO::GetNx()
{
	return ctx.Nx;
}
size_t CWO::GetNy()
{
	return ctx.Ny;
}
size_t CWO::GetNz()
{
	return ctx.Nz;
}

float CWO::GetWaveLength()
{
	return ctx.wave_length;
}
float CWO::GetWaveNum()
{
	return 2.0f*CWO_PI/GetWaveLength();
}
float CWO::GetDistance()
{
	return ctx.z;
}
float CWO::GetPx()
{
	return ctx.src_px;	
}
float CWO::GetPy()
{
	return ctx.src_py;
}
float CWO::GetPz()
{
	return ctx.src_pz;
}
float CWO::GetSrcPx()
{
	return GetPx();	
}
float CWO::GetSrcPy()
{
	return GetPy();
}
float CWO::GetSrcPz()
{
	return GetPz();
}
float CWO::GetDstPx()
{
	return ctx.dst_px;
}
float CWO::GetDstPy()
{
	return ctx.dst_py;
}
float CWO::GetDstPz()
{
	return ctx.dst_pz;
}

float CWO::GetOx()
{
	return GetSrcOx();

}
float CWO::GetOy()
{
	return GetSrcOy();

}
float CWO::GetOz()
{
	return GetSrcOz();
}
float CWO::GetSrcOx()
{
	return ctx.src_ox;
}
float CWO::GetSrcOy()
{
	return ctx.src_oy;
}
float CWO::GetSrcOz()
{
	return ctx.src_oz;
}
float CWO::GetDstOx()
{
	return ctx.dst_ox;
}
float CWO::GetDstOy()
{
	return ctx.dst_oy;
}
float CWO::GetDstOz()
{
	return ctx.dst_oz;
}

float CWO::GetLx()
{
	return GetSrcLx();
}
float CWO::GetLy()
{
	return GetSrcLy();
}
float CWO::GetSrcLx()
{
	return GetSrcPx()*GetNx();
}
float CWO::GetSrcLy()
{
	return GetSrcPy()*GetNy();
}
float CWO::GetDstLx()
{
	return GetDstPx()*GetNx();
}
float CWO::GetDstLy()
{
	return GetDstPy()*GetNy();
}

size_t CWO::GetMemSizeCplx()
{
	return GetNx()*GetNy()*GetNz()*sizeof(cwoComplex);
}

size_t CWO::GetMemSizeFloat()
{
	return GetNx()*GetNy()*GetNz()*sizeof(float);
}

size_t CWO::GetMemSizeChar()
{
	return GetNx()*GetNy()*GetNz()*sizeof(char);
}


int CWO::CheckExt(const char* fname, const char* ext)
{
	//return	1	: the extension of "fname" and "ext" is the same
	//			0	: the extension of "fname" and "ext" is not the same
	char buf1[256],buf2[256];

	strcpy(buf1,strrchr(fname, '.')+1);//extarct extension 
	strcpy(buf2,ext);

	for(int i=0;buf1[i]!='\0';i++) buf1[i]=tolower(buf1[i]); //convert lowercase letter
	for(int i=0;buf2[i]!='\0';i++) buf2[i]=tolower(buf2[i]); //convert lowercase letter

	return (strcmp(buf1,buf2)==0);

}
		

int CWO::Load(char *fname, int c)
{
	//return 0: success
	//return -1: error

	if(CheckExt(fname,"cwo")){
		//load as cwo file 

		FILE *fp=fopen(fname,"rb");
		if(fp==NULL) return -1;
		
		fread(&ctx,sizeof(cwoCtx),1,fp);//context
		Create(GetNx(),GetNy(),GetNz());

		if(GetFieldType()==CWO_FLD_COMPLEX){
			fread((cwoComplex*)GetBuffer(), sizeof(cwoComplex), GetNx()*GetNy()*GetNz(),fp); 
		}else{
			fread((float*)GetBuffer(), sizeof(float), GetNx()*GetNy()*GetNz(),fp); 
		}

		fclose(fp);


	}
	else{
		//load as bitmap file
		int tNx,tNy;
		cwoPrepareCimg(fname,&tNx,&tNy);
		Create(tNx,tNy);
		cwoComplex *p=(cwoComplex*)GetBuffer();
		cwoLoadCimg(&ctx,fname,c,p);
	}

	return CWO_SUCCESS;
}

int CWO::Load(char *fname_amp, char *fname_pha, int c)
{
	CWO *amp=(CWO*)new CWO;
	CWO *arg=(CWO*)new CWO;

	amp->Load(fname_amp,c);	
	arg->Load(fname_pha,c);	

	arg->ScaleReal(0,255,-CWO_PI,CWO_PI);
	arg->Arg2Cplx();

	(*amp)*=(*arg);

	(*this)=(*amp);

	delete amp;
	delete arg;

	return 0;
}


int CWO::Save(char *fname, int bmp_8_24)
{
	//return 0: success
	//return -1: error (field type is not real number etc...)

//	int type=GetFieldType();

	if(CheckExt(fname,"cwo")){
		//save as cwo file

		FILE *fp=fopen(fname,"wb");
		if(fp==NULL) return -1;
		
		fwrite(&ctx,sizeof(cwoCtx),1,fp);

		cwoComplex *p=(cwoComplex*)GetBuffer();
		fwrite(p, sizeof(cwoComplex), GetNx()*GetNy()*GetNz(),fp); 
	
		fclose(fp);
		
	}
	else{
		//save as monochrome image
	//	if(bmp_8_24==24)
			cwoSaveCimgMono(&ctx,fname,(cwoComplex *)GetBuffer());
	/*	else{
			int Nx=GetNx();
			int Ny=GetNy();
			unsigned char *save_img=(unsigned char *)new char[Nx*Ny];
			for(int i=0;i<Nx*Ny;i++){
				cwoComplex *p=(cwoComplex*)GetBuffer();
				save_img[i]=(unsigned char)CWO_RE(p[i]);
			}
			cwoSaveBmp(fname,save_img,Nx,Ny,8,-1);
			delete []save_img;
		}*/

		
	}
	
	return CWO_SUCCESS;

	
}

int CWO::Save(char *fname, CWO *r, CWO *g, CWO *b)
{
	//return 0: success
	//return -1: error (field type is not real number etc...)

	//save as bitmap file etc...
	if(r==NULL && g==NULL && b==NULL)
		//save as monochrome image
		cwoSaveCimgMono(&ctx,fname,(cwoComplex *)GetBuffer());
	else{
		//save as monochrome image
		cwoComplex *pr=(cwoComplex*)(r->GetBuffer());
		cwoComplex *pg=(cwoComplex*)(g->GetBuffer());
		cwoComplex *pb=(cwoComplex*)(b->GetBuffer());
		cwoSaveCimgColor(&ctx,fname,pr,pg,pb);
	}

	return CWO_SUCCESS;

}


int CWO::SaveMonosToColor(char* fname, char *r_name, char *g_name, char *b_name)
{
	cwoSaveCimgMonosToColor(&ctx, fname, r_name, g_name, b_name);
	return 0;
}


int CWO::SaveAsImage(char* fname, float i1, float i2, float o1, float o2, int flag)
{

	CWO *tmp=NULL;
	tmp=(CWO*)new CWO;
	if(tmp==NULL) return CWO_ERROR;

	(*tmp)=(*this);
	
	switch(flag & 0xff){
		case CWO_SAVE_AS_INTENSITY:
			tmp->Intensity();
			break;
		case CWO_SAVE_AS_PHASE:
			tmp->Phase();
			break;
		case CWO_SAVE_AS_AMP:
			tmp->Amp();
			break;
		case CWO_SAVE_AS_RE:
			tmp->Re();
			break;
		case CWO_SAVE_AS_IM:
			tmp->Im();
			break;
		case CWO_SAVE_AS_ARG:
			tmp->Arg();
			break;
	}
	
	if(flag & CWO_SAVE_AS_LOG) tmp->Log();

	tmp->ScaleReal(i1,i2,o1,o2);
	tmp->Save(fname);

	delete tmp;

	return CWO_SUCCESS;
}

int CWO::SaveAsImage(char* fname, int flag, CWO *r, CWO *g, CWO *b)
{
	CWO *tmp=NULL;
	tmp=(CWO*)new CWO;
	if(tmp==NULL) return CWO_ERROR;

	(*tmp)=(*this);
	
	switch(flag & 0xff){
		case CWO_SAVE_AS_INTENSITY:
			tmp->Intensity();
			break;
		case CWO_SAVE_AS_PHASE:
			tmp->Phase();
			break;
		case CWO_SAVE_AS_AMP:
			tmp->Amp();
			break;
		case CWO_SAVE_AS_RE:
			tmp->Re();
			break;
		case CWO_SAVE_AS_IM:
			tmp->Im();
			break;
		case CWO_SAVE_AS_ARG:
			tmp->Arg();
			break;
	}

	if(flag & CWO_SAVE_AS_LOG){
		tmp->Log(10);
	}


	tmp->ScaleReal(255);
	tmp->Save(fname);

	delete tmp;

	return CWO_SUCCESS;
}


int CWO::SaveLineAsText(char* fname, int flag, int x1, int y1, int x2, int y2)
{
	CWO *tmp=NULL;
	tmp=(CWO*)new CWO;
	if(tmp==NULL) return CWO_ERROR;

	(*tmp)=(*this);
	
	switch(flag){
		case CWO_SAVE_AS_INTENSITY:
			tmp->Intensity();
			break;
		case CWO_SAVE_AS_PHASE:
			tmp->Phase();
			break;
		case CWO_SAVE_AS_AMP:
			tmp->Amp();
			break;
		case CWO_SAVE_AS_RE:
			tmp->Re();
			break;
		case CWO_SAVE_AS_IM:
			tmp->Im();
			break;
		case CWO_SAVE_AS_ARG:
			tmp->Arg();
			break;
	}

	float dx=(x2-x1);
	float dy=(y2-y1);
	int len=(int)sqrt(dx*dx+dy*dy);
	float ang=atan2(dy,dx);
	int Nx=GetNx();

	FILE *fp=fopen(fname,"wt");
	cwoComplex *p=(cwoComplex*)tmp->GetBuffer();
	for(int i=0;i<len;i++){
		int x=(int)(x1+i*cos(ang));
		int y=(int)(y1+i*sin(ang));
		fprintf(fp,"%f\n",CWO_RE(p[x+y*Nx]));
	}
	fclose(fp);

	delete tmp;

	return CWO_SUCCESS;
}


int CWO::CmpCtx(cwoCtx &a, cwoCtx &b)
{
	if(a.Nx!=b.Nx) return 0;
	if(a.Ny!=b.Ny) return 0;
	if(a.Nz!=b.Nz) return 0;

	if(a.z!=b.z) return 0;
	if(a.wave_length!=b.wave_length) return 0;

	if(a.src_px!=b.src_px) return 0;
	if(a.src_py!=b.src_py) return 0;
	if(a.src_pz!=b.src_pz) return 0;

	if(a.dst_px!=b.dst_px) return 0;
	if(a.dst_py!=b.dst_py) return 0;
	if(a.dst_pz!=b.dst_pz) return 0;

	if(a.src_ox!=b.src_ox) return 0;
	if(a.src_oy!=b.src_oy) return 0;
	if(a.src_oz!=b.src_oz) return 0;

	if(a.dst_ox!=b.dst_ox) return 0;
	if(a.dst_oy!=b.dst_oy) return 0;
	if(a.dst_oz!=b.dst_oz) return 0;

	return 1;
}

/*
void CWO::SamplingMapX(cwoFloat2 *p, int Nx, int Ny, int quadrant)
{
	//quadrant	0 : first quadrant
	//quadrant	1 : second
	//			2 : third
	//quadrant	3 : forth

#ifdef _OPENMP
	omp_set_num_threads(GetThreads());
	#pragma omp parallel for schedule(dynamic)	
#endif
	for(int i=0;i<Ny/2;i++){
		float lx,ly;
		switch(quadrant){
			case 0:
				lx=Nx/2.0f;
				break;
			case 1:
				lx=Nx/2.0f-1;
				break;
			case 2:
				lx=Nx/2.0f-1;
				break;
			case 3:
				lx=Nx/2.0f;
				break;	
		}
		for(int j=0;j<Nx/2;j++){
			int x,y;
			switch(quadrant){
				case 0:
					x=Nx/2+j;
					y=Ny/2-i-1;
					lx+=p[x+y*Nx].x;
					break;
				case 1:
					x=Nx/2-j-1;
					y=Ny/2-i-1;
					lx-=p[x+y*Nx].x;
					break;
				case 2:
					x=Nx/2-j-1;
					y=Ny/2+i;
					lx-=p[x+y*Nx].x;
					break;
				case 3:
					x=Nx/2+j;
					y=Ny/2+i;
					lx+=p[x+y*Nx].x;
					break;	
			}
			
			p[x+y*Nx].x=lx*2*CWO_PI/Nx;
		}

	}

}

void CWO::SamplingMapY(cwoFloat2 *p, int Nx, int Ny, int quadrant)
{
	//quadrant	0 : first quadrant
	//quadrant	1 : second
	//			2 : third
	//quadrant	3 : forth

#ifdef _OPENMP
	omp_set_num_threads(GetThreads());
	#pragma omp parallel for schedule(dynamic)	
#endif
	for(int j=0;j<Nx/2;j++){
		float lx, ly;
		switch(quadrant){
			case 0:
				ly=Ny/2.0f-1;
				break;
			case 1:
				ly=Ny/2.0f-1;
				break;
			case 2:
				ly=Ny/2.0f;
				break;
			case 3:
				ly=Ny/2.0f;
				break;	
		}
		for(int i=0;i<Ny/2;i++){
			int x,y;
			switch(quadrant){
				case 0:
					x=Nx/2+j;
					y=Ny/2-i-1;
					ly-=p[x+y*Nx].y;
					break;
				case 1:
					x=Nx/2-j-1;
					y=Ny/2-i-1;
					ly-=p[x+y*Nx].y;
					break;
				case 2:
					x=Nx/2-j-1;
					y=Ny/2+i;
					ly+=p[x+y*Nx].y;
					break;
				case 3:
					x=Nx/2+j;
					y=Ny/2+i;
					ly+=p[x+y*Nx].y;
					break;	
			}
			
			p[x+y*Nx].y=ly*2*CWO_PI/Ny;
		}
	
	}

}
*/

/*
void CWO::DiffractDirect(float d, CWO *snumap, CWO *dnumap)
{
	//NU_RS_1
	int Nx=GetNx();
	int Ny=GetNy();
	int Nx_h=Nx>>1;
	int Ny_h=Ny>>1;
	double wl=GetWaveLength();
	double wn=GetWaveNum();
	double spx=GetSrcPx();
	double spy=GetSrcPy();
	double dpx=GetDstPx();
	double dpy=GetDstPy();
	
	Cplx();
	//SetCalcType(type);
	SetPropDist(d);
	
	cwoComplex* p_s=(cwoComplex*)GetBuffer();
	size_t size=Nx*Ny*sizeof(cwoComplex);
	cwoComplex* p_d=(cwoComplex*)__Malloc(size);
	
	cwoFloat2 *smp=(cwoFloat2*)snumap->GetBuffer();
	

#ifdef _OPENMP
			omp_set_num_threads(GetThreads());
			#pragma omp parallel for schedule(static)	
#endif	
	for(int i2=0;i2<Ny;i2++){
		for(int j2=0;j2<Nx;j2++){
			p_d[j2+i2*Nx]=0.0f;

			for(int i1=0;i1<Ny;i1++){
				for(int j1=0;j1<Nx;j1++){
					int adr1=j1+i1*Nx;
					int adr2=j2+i2*Nx;

					double x1=smp[adr1].x*spx;
					double y1=smp[adr1].y*spy;
					double x2=(j2-Nx_h)*dpx;
					double y2=(i2-Ny_h)*dpy;
					double dx=x2-x1;
					double dy=y2-y1;
					
					double Ax=wl/(2*dpx);
					double Ay=wl/(2*dpy);
					double limx=sqrt(Ax*Ax/(1-Ax*Ax)*((dy*dy)+d*d));
					double limy=sqrt(Ay*Ay/(1-Ay*Ay)*((dx*dx)+d*d));

					double r=sqrt(dx*dx+dy*dy+d*d);
				
					if(fabs(x2-x1)<limx && fabs(y2-y1)<limy){
						cwoComplex e;
						e.Re(cos(wn*r));
						e.Im(sin(wn*r));
						cwoComplex cff;
						cff.Re(d/(r*r*r));
						cff.Im(-wn/(r*r));
						p_d[adr2]+=p_s[adr1] * e *cff /(Nx*Ny);
					}
				
				}
			}
		}
	}

	__Free(&p_field);
	p_field=p_d;

}
*/

void CWO::DiffractDirect(float d, CWO *snumap, CWO *dnumap)
{
	//NU_RS_2
	int Nx=GetNx();
	int Ny=GetNy();
	int Nx_h=Nx>>1;
	int Ny_h=Ny>>1;
	double wl=GetWaveLength();
	double wn=GetWaveNum();
	double spx=GetSrcPx();
	double spy=GetSrcPy();
	double dpx=GetDstPx();
	double dpy=GetDstPy();
	
	Cplx();
	//SetCalcType(type);
	SetPropDist(d);
	
	cwoComplex* p_s=(cwoComplex*)GetBuffer();
	size_t size=Nx*Ny*sizeof(cwoComplex);
	cwoComplex* p_d=(cwoComplex*)__Malloc(size);
	
	cwoFloat2 *smp=(cwoFloat2*)snumap->GetBuffer();
	

#ifdef _OPENMP
	omp_set_num_threads(GetThreads());
	#pragma omp parallel for schedule(static)	
#endif	
	for(long long int i2=0;i2<Ny;i2++){
		for(long long int j2=0;j2<Nx;j2++){
			p_d[j2+i2*Nx]=0.0f;

			for(long long int i1=0;i1<Ny;i1++){
				for(long long int j1=0;j1<Nx;j1++){
					int adr1=j1+i1*Nx;
					int adr2=j2+i2*Nx;

					double x1=(j1-Nx_h)*spx;
					double y1=(i1-Ny_h)*spy;
					double x2=smp[adr2].x*dpx;
					double y2=smp[adr2].y*dpy;
					double dx=x2-x1;
					double dy=y2-y1;
					
					double Ax=wl/(2*dpx);
					double Ay=wl/(2*dpy);
					double limx=sqrt(Ax*Ax/(1-Ax*Ax)*((dy*dy)+d*d));
					double limy=sqrt(Ay*Ay/(1-Ay*Ay)*((dx*dx)+d*d));

					double r=sqrt(dx*dx+dy*dy+d*d);
				
					if(fabs(x2-x1)<limx && fabs(y2-y1)<limy){
						cwoComplex e;
						e.Re(cos(wn*r));
						e.Im(sin(wn*r));
						cwoComplex cff;
						cff.Re(d/(r*r*r));
						cff.Im(-wn/(r*r));
						p_d[adr2]+=p_s[adr1] * e * cff /(Nx*Ny);
					}
				
				}
			}
		}
	}

	__Free(&p_field);
	p_field=p_d;

}

//void CWO::Diffract(float d, int type, cwoInt4 *zp, CWO *numap)
//{
//	Cplx();
//	SetCalcType(type);
//	
//	//diffraction (convolution type)	
//	if(	type==CWO_SCALED_FFT ||
//		type==CWO_SHIFTED_FRESNEL || type==CWO_ARSS_FRESNEL || 
//		type==CWO_FRESNEL_CONV || type==CWO_FRESNEL_ANALYSIS ||
//		type==CWO_ANGULAR || type==CWO_SHIFTED_ANGULAR || type==CWO_SCALED_ANGULAR || 
//		type==CWO_HUYGENS || type==CWO_LAPLACIAN_WAVE_RETV || 
//		type==CWO_NU_ANGULAR1 || type==CWO_NU_ANGULAR2 ||
//		type==CWO_NU_FRESNEL1 || type==CWO_NU_FRESNEL2){
//			
//		int Nx=GetNx();
//		int Ny=GetNy();
//		int Nx2=Nx<<1;
//		int Ny2=Ny<<1;
//		size_t i_size;
//		size_t t_size;
//		size_t o_size;
//
//		t_size=Nx2*Ny2*sizeof(cwoComplex);
//		o_size=Nx*Ny*sizeof(cwoComplex);
//
//		SetSize(Nx,Ny);	
//
//		//int IsRecalc=(!CmpCtx(ctx, prev_ctx)) | (p_diff_a==NULL) | (p_diff_b==NULL);
//		//prev_ctx=ctx;
//
//
//		p_diff_a=(cwoComplex*)__Malloc(t_size);
//		cwoComplex *a=p_diff_a;
//		
//	
//	
//		if(GetFieldType()==CWO_FLD_COMPLEX){
//			__Expand(
//				GetBuffer(),0,0,Nx,Ny,
//				a,Nx2/2-Nx/2,Ny2/2-Ny/2,Nx2,Ny2,
//				CWO_C2C);
//
//		}
//		else{
//			__Expand(
//				GetBuffer(),0,0,Nx,Ny,
//				a,Nx2/2-Nx/2,Ny2/2-Ny/2,Nx2,Ny2,
//				CWO_R2C);
//		}
//
//		
//		__Free((void**)&p_field);
//		p_diff_b=(cwoComplex*)__Malloc(t_size);	
//		cwoComplex *b=p_diff_b;
//
//		DiffractConv(d,type,a,b,zp,numap);
//		
//		__Free((void**)&p_diff_b);
//
//	//	if(GetFieldType()!=CWO_FLD_COMPLEX){
//	//		SetFieldType(CWO_FLD_COMPLEX);
//			__Free((void**)&p_field);
//			p_field=__Malloc(o_size);
//	//	}
//		
//		__Expand(
//			a,Nx2/2-Nx/2,Ny2/2-Ny/2,Nx2,Ny2,
//			GetBuffer(),0,0,Nx,Ny,
//			CWO_C2C);
//	
//		__Free((void**)&p_diff_a);
//		
//		
//	}
//	else {
//		//diffraction (Fourier type)
//		DiffractFourier(d,type,zp);
//	}
//
//	SetFieldType(CWO_FLD_COMPLEX);
//}

void CWO::Diffract(float d, int type, cwoInt4 *zp, cwoComplex *knl_mask, CWO *numap)
{
//	Cplx();
//	SetCalcType(type);
	
	//diffraction (convolution type)	
	if(	type==CWO_SCALED_FFT ||
		type==CWO_SHIFTED_FRESNEL || type==CWO_ARSS_FRESNEL || 
		type==CWO_FRESNEL_CONV || type==CWO_FRESNEL_ANALYSIS ||
		type==CWO_ANGULAR || type==CWO_SHIFTED_ANGULAR || type==CWO_SCALED_ANGULAR || 
		type==CWO_HUYGENS || type==CWO_LAPLACIAN_WAVE_RETV || 
		type==CWO_NU_ANGULAR1 || type==CWO_NU_ANGULAR2 ||
		type==CWO_NU_FRESNEL1 || type==CWO_NU_FRESNEL2){
			
		long long int Nx=GetNx();
		long long int Ny=GetNy();
		long long int Nx2=Nx<<1;
		long long int Ny2=Ny<<1;
		long long int i_size;
		long long int t_size;
		long long int o_size;

		t_size=Nx2*Ny2*sizeof(cwoComplex);
		o_size=Nx*Ny*sizeof(cwoComplex);

		SetSize(Nx,Ny);	

		//int IsRecalc=(!CmpCtx(ctx, prev_ctx)) | (p_diff_a==NULL) | (p_diff_b==NULL);
		//prev_ctx=ctx;

		p_diff_a=(cwoComplex*)__Malloc(t_size);
		p_diff_b=(cwoComplex*)__Malloc(t_size);	

		cwoComplex *a=p_diff_a;
		cwoComplex *b=p_diff_b;
	
		if(GetFieldType()==CWO_FLD_COMPLEX){
			__Expand(
				GetBuffer(),0,0,Nx,Ny,
				a,Nx2/2-Nx/2,Ny2/2-Ny/2,Nx2,Ny2,
				CWO_C2C);
		}
		else{
			__Expand(
				GetBuffer(),0,0,Nx,Ny,
				a,Nx2/2-Nx/2,Ny2/2-Ny/2,Nx2,Ny2,
				CWO_R2C);
		}

		DiffractConv(d,type,a,b,zp,knl_mask,numap);
		
		if(GetFieldType()!=CWO_FLD_COMPLEX){
			SetFieldType(CWO_FLD_COMPLEX);
			__Free((void**)&p_field);
			p_field=__Malloc(o_size);
		}
		
		__Expand(
			a,Nx2/2-Nx/2,Ny2/2-Ny/2,Nx2,Ny2,
			GetBuffer(),0,0,Nx,Ny,
			CWO_C2C);
	
		__Free((void**)&p_diff_a);
		__Free((void**)&p_diff_b);
		
	}
	else {
		//diffraction (Fourier type)
		DiffractFourier(d,type,zp);
	}

	SetFieldType(CWO_FLD_COMPLEX);
}



void CWO::DiffractConv(float d, int type, cwoComplex *a, cwoComplex *b, cwoInt4 *zp, cwoComplex *knl_mask, CWO *numap, int prop_flag)
{
	
	SetPropDist(d);

	if(type==CWO_SCALED_FFT){
		float ax=GetSrcPx()/GetDstPx();
		float ay=GetSrcPy()/GetDstPy();
		
		float sx=-CWO_PI*ax/(GetNx()/2);
		float sy=-CWO_PI*ay/(GetNy()/2);

		__ScaledFFTCoeff(a, sx, sy);
		__FFT(a,a,CWO_C2C);
		__ScaledFFTKernel(b, -sx, -sy);
		__FFT(b,b,CWO_C2C);
		__Mul(a, b, a);
		
		__IFFT(a,a);
		__FFTShift(a);//important
	
		__ScaledFFTCoeff(a, sx, sy);
		
		//
	}
	else if(type==CWO_SHIFTED_FRESNEL){
		////Shifted-Fresnel diffraction
		__ShiftedFresnelAperture(a);
		__FFT(a,a,CWO_C2C);
		//__Mul(a,GetPx()*GetPy(),a);
		if(prop_flag==1){
			__ShiftedFresnelProp(b);
			__FFT(b,b,CWO_C2C);
		}
		__Mul(a, b, a);
		__IFFT(a,a);
		__ShiftedFresnelCoeff(a);//include the multiplication of 1/(Nx*Ny)
		
	}
	else if(type==CWO_ARSS_FRESNEL){
		////Aliasing-reduced scaled and shifted Fresnel diffraction
		double wl=GetWaveLength();
		double spx=GetSrcPx();
		double spy=GetSrcPy();
		double dpx=GetDstPx();
		double dpy=GetDstPy();
		double ox=GetSrcOx()+GetDstOx();
		double oy=GetSrcOy()+GetDstOy();
		double z=d;
		double sx1=spx/dpx;
		double sy1=spy/dpy;
		double sx2=dpx/spx;
		double sy2=dpy/spy;

		int cz1x=(int)fabs((wl*fabs(z)/dpx - fabs(2*sx1*ox))/(2*fabs(1-sx1)*dpx));
		int cz1y=(int)fabs((wl*fabs(z)/dpy - fabs(2*sy1*oy))/(2*fabs(1-sy1)*dpy));
		int u1x=(int)fabs((wl*fabs(z)/dpx - fabs(2*sx1*ox))/(2*(sx1*sx1-sx1)*dpx));
		int u1y=(int)fabs((wl*fabs(z)/dpy - fabs(2*sy1*oy))/(2*(sy1*sy1-sy1)*dpy));
		int h1x=(int)(wl*fabs(z)/(2*sx1*dpx*dpx));
		int h1y=(int)(wl*fabs(z)/(2*sy1*dpy*dpy));

		int cz2x=(int)fabs((wl*fabs(z)/spx - fabs(2*sx2*ox))/(2*(sx2*sx2-sx2)*spx));
		int cz2y=(int)fabs((wl*fabs(z)/spy - fabs(2*sy2*oy))/(2*(sy2*sy2-sy2)*spy));
		int u2x=(int)fabs((wl*fabs(z)/spx - fabs(2*sx2*ox))/(2*fabs(1-sx2)*spx));
		int u2y=(int)fabs((wl*fabs(z)/spy - fabs(2*sy2*oy))/(2*fabs(1-sy2)*spy));
		int h2x=(int)(wl*fabs(z)/(2*sx2*spx*spx));
		int h2y=(int)(wl*fabs(z)/(2*sy2*spy*spy));

		__ARSSFresnelAperture(a);
		__FFT(a,a,CWO_C2C);
		
		if(prop_flag==1){
			
			__ARSSFresnelProp(b);
			__FFTShift(b);
		
			__RectFillOutside(
				b,
				GetNx()/2-h1x, GetNy()/2-h1y,
				h1x*2, h1y*2,
				Polar(0,0));

			__FFTShift(b);
			
			__FFT(b,b,CWO_C2C);
		
		}
		__Mul(a, b, a);
		__IFFT(a,a);
		__ARSSFresnelCoeff(a);//include the multiplication of 1/(Nx*Ny)
	}
	else if(type==CWO_FRESNEL_CONV){
		////Fresnel diffraction(Convolution)
	/*	float nxny=GetNx()*GetNy();
		__FFT(a,a,CWO_C2C);
		if(prop_flag==1){
			__FresnelConvProp(b);
			__FFT(b,b,CWO_C2C);
		}
		__Mul(a, b, a);
		__IFFT(a,a);
		__FresnelConvCoeff(a,1/(nxny*nxny));
*/

		float nxny=(float)GetNx()*(float)GetNy();
		__FFT(a,a,CWO_C2C);
		if(prop_flag==1){
			__FresnelConvProp(b);
			if(knl_mask!=NULL) __Mul(b, knl_mask, b);
			__FFT(b,b,CWO_C2C);
		}

		__Mul(a, b, a);
		__IFFT(a,a);
		__FresnelConvCoeff(a,1/(nxny));
	
	}
	else if(type==CWO_FRESNEL_ANALYSIS){
		////Fresnel diffraction(Analysis form)
		__FFT(a,a,CWO_C2C);
		__FresnelAnalysisTransfer(a,a);
		__IFFT(a,a);
		__Mul(a,1.0f/(GetNx()*GetNy()),a);
	}
	else if(type==CWO_ANGULAR){
		//Angular spectrum method
		cwoCtx ctx_tmp=ctx;//imporatant

		__FFT(a,a,CWO_C2C);

		//__Mul(a,1.0f/(GetNx()*GetNy()),a);
		if(/*IsRecalc*/1){
			//__AngularProp(a,GetPx(),GetPy());
			//SetOffset(GetDstOx()-GetSrcOx(),-(GetDstOy()-GetSrcOy()));
			__AngularProp(a,CWO_PROP_MUL_FFT_SHIFT);

			//__Hanning(a,0,0,GetNx(),GetNy());
			if(zp!=NULL){
				int x1=zp->x1;
				int y1=zp->x2;
				int x2=zp->x3;
				int y2=zp->x4;
				__RectFillInside(a,x1,y1,x2,y2,Polar(0,0));
			}
		}

		__IFFT(a,a);

		__Mul(a,1.0f/(GetNx()*GetNy()),a);
		ctx=ctx_tmp;//imporatant
	}
	else if(type==CWO_SHIFTED_ANGULAR){
		//Shifted angular spectrum method
	//	cwoCtx ctx_tmp=ctx;//imporatant
		
		__FFT(a,a,CWO_C2C);
		__FFTShift(a);
		if(prop_flag){
			__AngularProp(a,CWO_PROP_MUL_CENTER);
			float fx_c, fy_c, fx_w,fy_w;
			__AngularLim(&fx_c, &fx_w, &fy_c, &fy_w);

			__RectFillOutside(
				a,
				GetNx()/2+fx_c-fx_w/2, GetNy()/2+fy_c-fy_w/2,
				fx_w, fy_w,
				Polar(0,0));
		}
	
		__FFTShift(a);
		__IFFT(a,a);
		__Mul(a,1.0f/(GetNx()*GetNy()),a);
		//ctx=ctx_tmp;//imporatant
	}
	else if(type==CWO_SCALED_ANGULAR){
		//Scaled angular spectrum method

		cwoCtx ctx_tmp=ctx;//imporatant

		double spx=GetSrcPx();
		double dpx=GetDstPx();
		float R=spx/dpx;
		int Nx=GetNx();
		int Ny=GetNy();

	//	cwoFloat2 *p_map=(cwoFloat2*)__Malloc(GetMemSizeFloat()*2);

		if(R<1){
			CWO *mp=(CWO *)new CWO;
			mp->SamplingMapScaleOnly(Nx,Ny,R,1.0f);	
			cwoFloat2 *p_map=(cwoFloat2*)mp->GetBuffer();
			__NUFFT_T1(a,p_map,1,2);
						
			delete mp;//important ! avoiding meory leak

			SetPitch(GetDstPx(),GetDstPy());
			//SetOffset(GetDstOx()-GetSrcOx(),-(GetDstOy()-GetSrcOy()));
			__AngularProp(a,CWO_PROP_MUL_CENTER);
			
			//Band-limit
			float fx_c, fy_c, fx_w,fy_w;
			__AngularLim(&fx_c, &fx_w, &fy_c, &fy_w);
			__RectFillOutside(
				a,
				GetNx()/2+fx_c-fx_w/2, GetNy()/2+fy_c-fy_w/2,
				fx_w, fy_w,
				Polar(0,0));			
		
			__FFTShift(a);
			__IFFT(a,a);
			__Mul(a,1.0f/(GetNx()*GetNy()),a);
			
		}
		else{//R>1
			__FFT(a,a,CWO_C2C);
		
			SetPitch(GetSrcPx(),GetSrcPy());
			//SetOffset(GetDstOx()-GetSrcOx(),-(GetDstOy()-GetSrcOy()));
			__AngularProp(a,CWO_PROP_MUL_FFT_SHIFT);
			__FFTShift(a);
	
			//Band-limit
			float fx_c, fy_c, fx_w,fy_w;
			__AngularLim(&fx_c, &fx_w, &fy_c, &fy_w);
			__RectFillOutside(
				a,
				GetNx()/2+fx_c-fx_w/2, GetNy()/2+fy_c-fy_w/2,
				fx_w, fy_w,
				Polar(0,0));			
				
			CWO *mp=(CWO*)new CWO;
			mp->SamplingMapScaleOnly(Nx,Ny,1/R,-1.0f);	
			cwoFloat2 *p_map=(cwoFloat2*)mp->GetBuffer();

			

			//__NUFFT_T2(a,p_x1,2,12);
			__NUFFT_T2(a,p_map,1,2);
	
			delete mp;//important ! avoiding meory leak

			__FFTShift(a);
			__Mul(a,1.0f/(GetNx()*GetNy()),a);
			

		}
		
		//__Free((void**)&p_map);

		ctx=ctx_tmp;//important
	
	}
	else if(type==CWO_NU_ANGULAR1){
	
		numap->ConvertSamplingMap(type);

		cwoCtx ctx_tmp=ctx;//imporatant
		int Nx=GetNx();
		int Ny=GetNy();
		cwoFloat2 *p=(cwoFloat2 *)numap->GetBuffer();
		//__NUFFT_T1(a,p_x1,1,2);
		__NUFFT_T1(a,p,1,2);
		
		SetPitch(GetDstPx(),GetDstPy());
		__AngularProp(a,CWO_PROP_MUL_CENTER);

		//Band-limit
		float fx_c, fy_c, fx_w,fy_w;
		__AngularLim(&fx_c, &fx_w, &fy_c, &fy_w);
		__RectFillOutside(
			a,
			GetNx()/2+fx_c-fx_w/2, GetNy()/2+fy_c-fy_w/2,
			fx_w, fy_w,
			Polar(0,0));			
		
		__FFTShift(a);
		__IFFT(a,a);
	
		ctx=ctx_tmp;//important
	}
	else if(type==CWO_NU_ANGULAR2){

		numap->ConvertSamplingMap(type);

		cwoCtx ctx_tmp=ctx;//imporatant
		__FFT(a,a,CWO_C2C);
		
		SetPitch(GetSrcPx(),GetSrcPy());
		__AngularProp(a,CWO_PROP_MUL_FFT_SHIFT);
		__FFTShift(a);
	
		//Band-limit
		float fx_c, fy_c, fx_w,fy_w;
		__AngularLim(&fx_c, &fx_w, &fy_c, &fy_w);
		__RectFillOutside(
			a,
			GetNx()/2+fx_c-fx_w/2, GetNy()/2+fy_c-fy_w/2,
			fx_w, fy_w,
			Polar(0,0));			
		
		cwoFloat2 *p=(cwoFloat2 *)numap->GetBuffer();
		__NUFFT_T2(a,p,1,2);
	
		__FFTShift(a);
		//__Free((void**)&p_x1);

		ctx=ctx_tmp;//important
	}
	else if(type==CWO_NU_FRESNEL1){
	
		numap->ConvertSamplingMap(type);

		cwoCtx ctx_tmp=ctx;//imporatant
		long long int Nx=GetNx();
		long long int Ny=GetNy();
		cwoFloat2 *p=(cwoFloat2 *)numap->GetBuffer();
		//__NUFFT_T1(a,p_x1,1,2);
		__NUFFT_T1(a,p,1,2);
		
		__FFTShift(a);

		SetPitch(GetDstPx(),GetDstPy());

		__FresnelConvProp(b);
		__FFT(b,b,CWO_C2C);

		__Mul(a, b, a);

		__IFFT(a,a);

		__FresnelConvCoeff(a,1);
		
		//__Free((void**)&p_x1);
		ctx=ctx_tmp;//important
	}
	else if(type==CWO_NU_FRESNEL2){

		numap->ConvertSamplingMap(type);

		cwoCtx ctx_tmp=ctx;//imporatant
		__FFT(a,a,CWO_C2C);
		
		SetPitch(GetSrcPx(),GetSrcPy());
		__FresnelConvProp(b);
		__FFT(b,b,CWO_C2C);
		__Mul(a, b, a);
		__FFTShift(a);

		cwoFloat2 *p=(cwoFloat2 *)numap->GetBuffer();
		__NUFFT_T2(a,p,1,2);
	
		__FFTShift(a);
		//__Free((void**)&p_x1);

		ctx=ctx_tmp;//important
	}
	else if(type==CWO_LAPLACIAN_WAVE_RETV){	
		/*__FFT(a,a,CWO_C2C);
		__Mul(a,GetPx()*GetPy(),a);
		__FresnelConvProp(b);
		__FFT(b,b,CWO_C2C);
		__Mul(a, b, a);
		__FresnelConvCoeff(a);
		__InvFx2Fy2(a);
		__IFFT(a,a);
		__Div(a,GetNx()*GetNy(),a);*/
		
	//	__FFTShift(a);
		__FFT(a,a,CWO_C2C);
	//	__FFTShift(a);
		__Mul(a,GetPx()*GetPy(),a);
		
		__FresnelConvProp(b);
		__FFT(b,b,CWO_C2C);
				
		__Mul(a, b, a);
		__FresnelConvCoeff(a);
		
		__InvFx2Fy2(a);
		
	//	__FFTShift(a);
		__IFFT(a,a);
	//	__FFTShift(a);
		__Div(a,GetNx()*GetNy(),a);
	}
	else if(type==CWO_HUYGENS){
		//Huygens-Fresnel diffraction
		__FFT(a,a,CWO_C2C);
		__Div(a,GetNx()*GetNy(),a);
		if(/*IsRecalc*/1){
			__HuygensProp(b);
			__FFT(b,b,CWO_C2C);
		}
				
		__Mul(a, b, a);
		__IFFT(a,a);

	}
	
}


void CWO::DiffractFourier(float d, int type, cwoInt4 *zp)
{
	long long int Nx=GetNx();
	long long int Ny=GetNy();
	SetSize(Nx,Ny);	
		
	Cplx();//change source plane to complex data
	cwoComplex *a=(cwoComplex*)GetBuffer();

	SetPropDist(d);

	/*	if(type==CWO_FRESNEL_FOURIER){

			__FresnelFourierProp(a);
			
			if(d>=0.0f){
				__FFT(a,a,CWO_C2C);
				//__FFTShift(a);
			}
			else{
				__IFFT(a,a);
				//__FFTShift(a);
			}
			
				
			__FresnelFourierCoeff(a);//include the multiplication of 1/(Nx*Ny)

			float px=GetPx();
			float py=GetPy();
			px=(GetWaveLength()*fabs(d))/(px*GetNx());
			py=(GetWaveLength()*fabs(d))/(py*GetNy());

			printf("%e %e\n",px,py);
			SetSrcPitch(px,py);
			
		}
		else */
		if(type==CWO_FFT){
			cwoComplex *p=(cwoComplex*)GetBuffer();
			__FFTShift(p);
			(d>0.0f)?(__FFT(p,p,CWO_C2C)):(__IFFT(p,p));
			__FFTShift(p);
			__Mul(p,sqrt(1.0f/(GetNx()*GetNy())),p);

		}
		else if(type==CWO_FRESNEL_FOURIER){
		
		/*	float z1=d/2+500;
			float z2=d/2-500;
			
			float z=z1+z2;
			int Nx=GetNx();
			int Ny=GetNy();

			float wl=GetWaveLength();
			float px=GetPx();
			float py=GetPy();
			float sox=GetSrcOx();
			float soy=GetSrcOy();
			float dox=GetDstOx();
			float doy=GetDstOy();
			float p1x=fabs(wl*z1/(Nx*px));
			float p1y=fabs(wl*z1/(Ny*py));
			float p2x=fabs(wl*z2/(Nx*p1x));
			float p2y=fabs(wl*z2/(Ny*p1y));

			__FresnelDblAperture(a,z1);
			
			(z1>0.0f) ? (__FFT(a,a,CWO_C2C)) : (__IFFT(a,a));
		
			__FresnelDblFourierDomain(a, z1, z2, zp);
	
			//band-limited area			
			int f1x=(int)fabs(wl*z1*z2/(2*z*p1x*p1x));
			int f1y=(int)fabs(wl*z1*z2/(2*z*p1y*p1y));
		
			cwoComplex mask;
			mask.Re(0.0f);
			mask.Im(0.0f);
			__RectFillOutside(a, Nx/2-f1x, Ny/2-f1y, f1x*2, f1y*2, mask);
			
			(z2>0.0f) ? (__FFT(a,a,CWO_C2C)) : (__IFFT(a,a));

			__FresnelDblCoeff(a, z1, z2);
					
*/
		}
		else if(type==CWO_FRESNEL_DBL){
			float z1=d/2+500;
			float z2=d/2-500;
			
			float z=z1+z2;
			int Nx=GetNx();
			int Ny=GetNy();

			float wl=GetWaveLength();
			float px=GetPx();
			float py=GetPy();
			float sox=GetSrcOx();
			float soy=GetSrcOy();
			float dox=GetDstOx();
			float doy=GetDstOy();
			float p1x=fabs(wl*z1/(Nx*px));
			float p1y=fabs(wl*z1/(Ny*py));
			float p2x=fabs(wl*z2/(Nx*p1x));
			float p2y=fabs(wl*z2/(Ny*p1y));

			//printf("p2x %e\n", p2x);

			__FresnelDblAperture(a,z1);
			
			(z1>0.0f) ? (__FFT(a,a,CWO_C2C)) : (__IFFT(a,a));
		
			__FresnelDblFourierDomain(a, z1, z2, zp);
	
			//band-limited area			
			int f1x=(int)fabs(wl*z1*z2/(2*z*p1x*p1x));
			int f1y=(int)fabs(wl*z1*z2/(2*z*p1y*p1y));
		
			cwoComplex mask;
			mask.Re(0.0f);
			mask.Im(0.0f);
			__RectFillOutside(a, Nx/2-f1x, Ny/2-f1y, f1x*2, f1y*2, mask);
			
			(z2>0.0f) ? (__FFT(a,a,CWO_C2C)) : (__IFFT(a,a));

			__FresnelDblCoeff(a, z1, z2);//important

		}
		else if(type==CWO_FRESNEL_ARBITRARY){
						
			int Nx=GetNx();
			int Ny=GetNy();
			
			double wl=GetWaveLength();
			double k=GetWaveNum();
			double z=GetDistance();

			double spx=GetSrcPx();
			double spy=GetSrcPy();
			double dpx=GetDstPx();
			double dpy=GetDstPy();

			double ox1=GetSrcOx();
			double oy1=GetSrcOy();
			double ox2=GetDstOx();
			double oy2=GetDstOy();
		
			cwoComplex *p_fld=(cwoComplex *)GetBuffer();
			float *p_d1=(float *)GetBuffer(CWO_BUFFER_D1);
			float *p_d2=(float *)GetBuffer(CWO_BUFFER_D2);
			cwoFloat2 *p_x1=(cwoFloat2 *)GetBuffer(CWO_BUFFER_X1);
			cwoFloat2 *p_x2=(cwoFloat2 *)GetBuffer(CWO_BUFFER_X2);

			//cwoComplex *p_tmp=(cwoComplex*)__Malloc(Nx*Ny*sizeof(cwoComplex));
			//__Memcpy(p_tmp,p,Nx*Ny*sizeof(cwoComplex));
		
			#pragma omp parallel for schedule(static) num_threads(4)
			for(int i=0;i<Ny;i++){
				for(int j=0;j<Nx;j++){
					int adr=j+i*Nx;
					
					double x1=(p_x1[adr].x+ox1);
					double y1=(p_x1[adr].y+oy1);
					double d1=p_d1[adr];

					double r=-d1+(x1*x1+y1*y1)/(2*(z-d1));
						
					cwoComplex tmp;					
					CWO_RE(tmp)=cos(k*r);
					CWO_IM(tmp)=sin(k*r);

					double offset=-2*CWO_PI*(ox2*x1+oy2*y1 +(ox1*ox2+oy1*oy2) )/(wl*(z-d1));

					cwoComplex tmp3;					
					CWO_RE(tmp3)=cos(offset);
					CWO_IM(tmp3)=sin(offset);

					//planar wave
					cwoComplex tmp2;
					CWO_RE(tmp2)=cos(k*d1);
					CWO_IM(tmp2)=sin(k*d1);

					cwoComplex f;
					CWO_RE(f)=CWO_RE(p_fld[adr]);
					CWO_IM(f)=CWO_IM(p_fld[adr]);
					f*=tmp3;
					f*=tmp2;
					f*=tmp;

					CWO_RE(p_fld[adr])=CWO_RE(f);
					CWO_IM(p_fld[adr])=CWO_IM(f);


				}
	
			}
				

		//	__Free((void**)&p_tmp);


			for(long long int i=0;i<Ny;i++){
				for(long long int j=0;j<Nx;j++){
					long long int adr=j+i*Nx;
					double d1=p_d1[adr];
					float scalex=2*CWO_PI*dpx/(wl*(z-d1));
					float scaley=2*CWO_PI*dpy/(wl*(z-d1));
					//printf("scale %e\n",scale);
					p_x1[adr].x*=scalex;
					p_x1[adr].y*=scaley;
				/*	p_x1[adr].x/=(spx*Nx)*scale;
					p_x1[adr].y/=(spy*Ny)*scale;*/

				}
			}

			NUFFT_T1();
			//NUDFT();

		/*	float scalex=2*CWO_PI*GetSrcPx()*GetDstPx()*GetNx()/(GetWaveLength()*z);
			float scaley=2*CWO_PI*GetSrcPy()*GetDstPy()*GetNy()/(GetWaveLength()*z);
			//float scalex=2*GetSrcPx()*GetDstPx()*GetNx()/(CWO_PI*GetWaveLength()*z);
			//float scaley=2*GetSrcPy()*GetDstPy()*GetNy()/(CWO_PI*GetWaveLength()*z);
			printf("interpolate %f %f \n",1/scalex,1/scaley);
			
			Interpolate(1/scalex,1/scaley);
	*/
			p_fld=(cwoComplex*)GetBuffer();
			for(long long int i=0;i<Ny;i++){
				for(long long int j=0;j<Nx;j++){
					long long int adr=j+i*Nx;
			
					float x2=p_x2[adr].x+ox2;
					float y2=p_x2[adr].y+oy2;
					float d2=p_d2[adr];

					//ñ{óàÇÕz+d2ÅEÅEÇ∆ÇµÇΩÇ¢Ç∆Ç±ÇÎÇæÇ™ zÇ™ëÂÇ´Ç∑Ç¨ÇÈèÍçáÇ™Ç†ÇË
					//rÇÃê∏ìxÇ™ë´ÇËÇ»Ç≠Ç»ÇÈèÍçáÇ™Ç†ÇÈÇΩÇﬂÅCzÇÕï™ó£ÇµÇƒÇ†ÇÈ
					float r=d2+(x2*x2+y2*y2)/(2*(z+d2));
							
					cwoComplex tmp;					
					CWO_RE(tmp)=cos(k*r);
					CWO_IM(tmp)=sin(k*r);
					
					cwoComplex tmp2;	
					CWO_RE(tmp2)=cos(k*z);
					CWO_IM(tmp2)=sin(k*z);

					cwoComplex tmp3;	
					CWO_RE(tmp3)=-cos(CWO_PI/2);
					CWO_IM(tmp3)=-sin(CWO_PI/2);

					float offset=-2*CWO_PI*((ox1*x2+oy1*y2)/(wl*z));
					cwoComplex tmp4;	
					CWO_RE(tmp4)=cos(offset);
					CWO_IM(tmp4)=sin(offset);

					p_fld[adr]*=tmp;
					p_fld[adr]*=tmp2;
					p_fld[adr]*=tmp3;
					p_fld[adr]*=tmp4;



				}
	
			}

			SetFieldType(CWO_FLD_COMPLEX);
		}
		else if(type==CWO_FRAUNHOFER_ARBITRARY){
			int Nx=GetNx();
			int Ny=GetNy();
			
			double wl=GetWaveLength();
			double k=GetWaveNum();
			double z=GetDistance();

			double spx=GetSrcPx();
			double spy=GetSrcPy();
			double dpx=GetDstPx();
			double dpy=GetDstPy();

			double ox1=GetSrcOx();
			double oy1=GetSrcOy();
			double ox2=GetDstOx();
			double oy2=GetDstOy();
		
			cwoComplex *p_fld=(cwoComplex *)GetBuffer();
			float *p_d1=(float *)GetBuffer(CWO_BUFFER_D1);
			float *p_d2=(float *)GetBuffer(CWO_BUFFER_D2);
			cwoFloat2 *p_x1=(cwoFloat2 *)GetBuffer(CWO_BUFFER_X1);
			cwoFloat2 *p_x2=(cwoFloat2 *)GetBuffer(CWO_BUFFER_X2);
	
			NUFFT_T1();

			p_fld=(cwoComplex*)GetBuffer();
			for(long long int i=0;i<Ny;i++){
				for(long long int j=0;j<Nx;j++){
					long long int adr=j+i*Nx;
			
					float x2=p_x2[adr].x;
					float y2=p_x2[adr].y;
					float d2=p_d2[adr];

					float ph=CWO_PI*(x2*x2+y2*y2)/(wl*d2);
					float ph2=k*d2-CWO_PI/2;

					cwoComplex tmp;
					tmp.Re(ph+ph2);
					tmp.Im(ph+ph2);
					tmp/=wl*d2;
	
					p_fld[adr]*=tmp;


				}
	
			}


	
			

			SetFieldType(CWO_FLD_COMPLEX);
		}



		SetFieldType(CWO_FLD_COMPLEX);



}

void CWO::Diffract3D(CWO &a, float d, int type)
{
	//a : input data(2D image)
	
	long long int Nz=GetNz();
	long long int sgn=(d>0)?(1):(-1);

	for(int i=0;i<Nz;i++){

		long long int Nx=a.GetNx();
		long long int Ny=a.GetNy();
		long long int size_2d=Nx*Ny*sizeof(cwoComplex);
		cwoComplex *p_tmp_fld=(cwoComplex*)__Malloc(size_2d); 
		
		cwoComplex *src=(cwoComplex*)a.GetBuffer();
		cwoComplex *dst=(cwoComplex*)GetBuffer();

		__Memcpy(p_tmp_fld, src, size_2d);

		float pz=a.GetPz();
		float prop_d=d+(float)i*pz*sgn;//propagation distance
		a.Diffract(prop_d,type);

		__Memcpy(dst+i*Nx*Ny, src, size_2d);

		__Memcpy(src, p_tmp_fld, size_2d);	
	
		__Free((void**)&p_tmp_fld);

		//printf("%s %d\n",__FUNCTION__, i);
	
	}
}



void CWO::Diffract(float d, int type, int Dx, int Dy, char *dir)
{
	/*
		d		: propagation distance
		type	: type of diffraction
		Dx		: Divided size
		Dy		: Divided size
		dir		: temporary directory path (ex. dir="e:\\test\") 
	*/
	using namespace std;
	vector<string> f_list;

	long long int Nx=GetNx();
	long long int Ny=GetNy();

	int Tx=Nx/Dx; //division number
	int Ty=Ny/Dy; //division number

	CWO *wo, *wo1;
	wo=(CWO *)new CWO;
	wo->Create(Dx,Dy);
	wo->Clear();

	wo1=(CWO *)new CWO;
	wo1->Create(Dx,Dy);


	for(long long int dy=0;dy<Ty;dy++){
		for(long long int dx=0;dx<Tx;dx++){
			
			char tmp[1024];
			if(dir!=NULL)
				sprintf(tmp,"%scwo_%03d_%03d.cwo",dir,dy,dx);
			else
				sprintf(tmp,"cwo_%03d_%03d.cwo",dy,dx);

			f_list.push_back(tmp); //save filename list

			//printf("%s\n",tmp);

			wo1->Clear();

			for(long long int sy=0;sy<Ty;sy++){
				for(long long int sx=0;sx<Tx;sx++){
					
					wo->Copy(*this,sx*Dx,sy*Dy,0,0,Dx,Dy);
					
					wo->SetWaveLength(this->GetWaveLength());
					wo->SetSrcPitch(this->GetPx(),this->GetPy(),this->GetPz());
					wo->SetDstPitch(this->GetDstPx(),this->GetDstPy(),this->GetDstPz());
					wo->SetSrcOffset(sx*Dx*wo->GetSrcPx(),sy*Dy*wo->GetSrcPy());
					wo->SetDstOffset(dx*Dx*wo->GetDstPx(),dy*Dy*wo->GetDstPy());
					
					wo->Diffract(d,type);

					(*wo1) += (*wo);
				
									
				}
			}	

			wo1->Save((char*)f_list[dx+Tx*dy].c_str());
		}
	}

	//accumulate 
	wo->Destroy();
	wo1->Destroy();
	wo->Create(Dx,Dy);
	wo1->Create(Tx*Dx,Ty*Dy);

	for(long long int i=0;i<Ty;i++){
		for(long long int j=0;j<Tx;j++){
			printf("fname read %s\n",(char *)f_list[j+i*Tx].c_str());
			wo->Load((char *)f_list[j+i*Tx].c_str());
			wo1->Copy(*wo,0,0,j*Dx,i*Dy,Dx,Dy);
		}
	}

	Copy(*wo1,0,0,0,0,GetNx(),GetNy());


	delete wo,wo1;

}

void CWO::ParticleField(cwoFloat3 pos, float radius, float amp, float init_ph)
{
	SetFieldType(CWO_FLD_COMPLEX);
	cwoParticleField(&ctx,(cwoComplex*)GetBuffer(),pos, radius, amp, init_ph);
	

}

void CWO::ReleaseTmpBuffer()
{
	 __Free((void **)&p_diff_a);
	 __Free((void **)&p_diff_b);

	 p_diff_a=NULL;
	 p_diff_b=NULL;
}

void CWO::FresnelCoeff(float z1, float z2, float wl1, float wl2){
	int Nx=GetNx();
	int Ny=GetNy();
	float _wlz=1/(z2*wl2);
	float ph=2*CWO_PI*z1/wl1 - CWO_PI/2;

	cwoComplex *p=(cwoComplex*)GetBuffer();
	cwoComplex t;
	CWO_RE(t)=cosf(ph)*_wlz;
	CWO_IM(t)=sinf(ph)*_wlz;
	
	for(long long int i=0;i<Nx*Ny;i++) p[i]*=t;


}


/////////////////////////////////////
//Planar & Spherical waves
/////////////////////////////////////

void CWO::__AddSphericalWave(
	cwoComplex *p, float x, float y, float z, float px, float py, float a)
{
	cwoAddSphericalWave(&ctx,p,x,y,z,px,py,a);
}
void CWO::__MulSphericalWave(cwoComplex *p, float x, float y, float z, float px, float py, float a)
{
	cwoMulSphericalWave(&ctx,p,x,y,z,px,py,a);
}

void CWO::AddSphericalWave(float x, float y, float z, float px, float py, float a)
{	
	Cplx();
	__AddSphericalWave((cwoComplex*)GetBuffer(), x, y, z, px, py, a);
}

void CWO::AddSphericalWave(float x, float y, float z, float a)
{	
	Cplx();
	float px=GetPx();
	float py=GetPy();
	__AddSphericalWave((cwoComplex*)GetBuffer(), x, y, z, px, py, a);
}
void CWO::MulSphericalWave(float x, float y, float z, float px, float py, float a)
{
	Cplx();
	__MulSphericalWave((cwoComplex*)GetBuffer(), x, y, z, px, py, a);
}

void CWO::MulSphericalWave(float x, float y, float z, float a)
{	
	Cplx();
	float px=GetPx();
	float py=GetPy();
	__MulSphericalWave((cwoComplex*)GetBuffer(), x, y, z, px, py, a);

}

void CWO::__AddApproxSphWave( cwoComplex *p, float x, float y, float z, float px, float py, float a)
{
	//cwoAddApproxSphWave(&ctx,p,x,y,z,px,py,a);
	cwoAddApproxSphWave(&ctx,p,x,y,z,z,z,px,py,a);

}
void CWO::__MulApproxSphWave(cwoComplex *p, float x, float y, float z, float px, float py, float a)
{
	//cwoMulApproxSphWave(&ctx,p,x,y,z,px,py,a);
	cwoMulApproxSphWave(&ctx,p,x,y,z,z,z,px,py,a);
}

void CWO::__AddApproxSphWave(cwoComplex *p, float x, float y, float z, float zx, float zy, float px, float py, float a)
{
	cwoAddApproxSphWave(&ctx,p,x,y,z,zx,zy,px,py,a);
}
void CWO::__MulApproxSphWave(cwoComplex *p, float x, float y, float z, float zx, float zy, float px, float py, float a)
{
	cwoMulApproxSphWave(&ctx,p,x,y,z,zx, zy,px,py,a);
}


void CWO::AddApproxSphWave(float x, float y, float z, float px, float py, float a)
{	
	Cplx();
	__AddApproxSphWave((cwoComplex*)GetBuffer(), x, y, z, px, py, a);
}

void CWO::AddApproxSphWave(float x, float y, float z, float a)
{
	Cplx();
	float px=GetPx();
	float py=GetPy();
	__AddApproxSphWave((cwoComplex*)GetBuffer(), x, y, z, px, py, a);
}

void CWO::MulApproxSphWave(float x, float y, float z, float px, float py, float a)
{	
	Cplx();
	__MulApproxSphWave((cwoComplex*)GetBuffer(), x, y, z, px, py, a);
}

void CWO::MulApproxSphWave(float x, float y, float z, float a)
{
	Cplx();
	float px=GetPx();
	float py=GetPy();
	__MulApproxSphWave((cwoComplex*)GetBuffer(), x, y, z, px, py, a);
}


void CWO::AddApproxSphWave(float x, float y, float z, float zx, float zy, float px, float py, float a)
{	
	Cplx();
	__AddApproxSphWave((cwoComplex*)GetBuffer(), x, y, z, zx, zy, px, py, a);
}

void CWO::MulApproxSphWave(float x, float y, float z, float zx, float zy, float px, float py, float a)
{	
	Cplx();
	__MulApproxSphWave((cwoComplex*)GetBuffer(), x, y, z, zx, zy, px, py, a);
}



void CWO::AddPlanarWave(float kx, float ky, float px, float py, float a)
{
//	Cplx();	
	cwoComplex *p=(cwoComplex*)GetBuffer();
	long long int Nx=GetNx();
	long long int Ny=GetNy();

#ifdef _OPENMP
	omp_set_num_threads(GetThreads());
	#pragma omp parallel for
#endif
	for(long long int i=0;i<Ny;i++){
		for(long long int j=0;j<Nx;j++){
			long long int adr=j+i*Nx;
			float dx=(j-Nx/2)*px;
			float dy=(i-Ny/2)*py;
			float ph=kx*dx+ky*dy;
			cwoComplex tmp;
			tmp.Re(a*cos(ph));
			tmp.Im(a*sin(ph));
			p[adr]+=tmp;
		}
	}

}

void CWO::AddPlanarWave(float kx, float ky, float a)
{
	float px=GetPx();
	float py=GetPy();
	AddPlanarWave(kx, ky, px, py, a);
}

void CWO::MulPlanarWave(float kx, float ky, float px, float py, float a)
{
	Cplx();	
	cwoComplex *p=(cwoComplex*)GetBuffer();
	long long int Nx=GetNx();
	long long int Ny=GetNy();

#ifdef _OPENMP
	omp_set_num_threads(GetThreads());
	#pragma omp parallel for schedule(static)	
#endif
	for(long long int i=0;i<Ny;i++){
		for(long long int j=0;j<Nx;j++){
			long long int adr=j+i*Nx;
			float dx=(j-Nx/2)*px;
			float dy=(i-Ny/2)*py;
			float ph=kx*dx+ky*dy;
			cwoComplex tmp;
			tmp.Re(a*cos(ph));
			tmp.Im(a*sin(ph));
			p[adr]*=tmp;
		}
	}

}

void CWO::MulPlanarWave(float kx, float ky, float a)
{
	float px=GetPx();
	float py=GetPy();
	MulPlanarWave(kx, ky, px, py, a);
}




void CWO::Re()
{
	SetFieldType(CWO_FLD_COMPLEX);
	__Re((cwoComplex*)GetBuffer(), (cwoComplex*)GetBuffer());

}
void CWO::Im()
{
	SetFieldType(CWO_FLD_COMPLEX);
	__Im((cwoComplex*)GetBuffer(), (cwoComplex*)GetBuffer());
}
void CWO::Conj()
{
	if(GetFieldType()!=CWO_FLD_COMPLEX) return;
	
	cwoComplex *p=(cwoComplex*)GetBuffer();
#ifdef _OPENMP
	omp_set_num_threads(GetThreads());
	#pragma omp parallel for schedule(static)	
#endif
	for(long long int i=0;i<GetNx()*GetNy()*GetNz();i++){
		CWO_IM(p[i])*=-1.0f;
	}
	
}
void CWO::Intensity()
{
	SetFieldType(CWO_FLD_COMPLEX);
	__Intensity((cwoComplex*)GetBuffer(),(cwoComplex*)GetBuffer());

}
void CWO::Amp()
{
	SetFieldType(CWO_FLD_COMPLEX);
	__Amp((cwoComplex*)GetBuffer(),(cwoComplex*)GetBuffer());
}
void CWO::Phase(float offset)
{
	SetFieldType(CWO_FLD_COMPLEX);
	__Phase((cwoComplex*)GetBuffer(),(cwoComplex*)GetBuffer(),offset);
}

void CWO::Arg(float offset)
{
	SetFieldType(CWO_FLD_COMPLEX);
	__Arg((cwoComplex*)GetBuffer(),(cwoComplex*)GetBuffer(),offset);
}

void CWO::Cplx()
{
	if(GetFieldType()==CWO_FLD_COMPLEX) return;
	
	float *src=(float*)GetBuffer();
	cwoComplex *dst=(cwoComplex*)GetBuffer();

	float *src_tmp=(float*)__Malloc(GetMemSizeFloat());
	__Memcpy(src_tmp,src,GetMemSizeFloat());

	switch(GetFieldType()){
		case CWO_FLD_INTENSITY:
			__Real2Complex(src_tmp,dst);
			break;
		case CWO_FLD_PHASE:
			__Phase2Complex(src_tmp,dst);
			break;
	}

	__Free((void**)&src_tmp);

	SetFieldType(CWO_FLD_COMPLEX);
}



int CWO::Cplx(CWO &a, CWO &b)
{
	//Make complex ampltude
	//a		: intensity field
	//ph	: phase field or intensity field
	//return 

	int a_type=a.GetFieldType();
	int b_type=b.GetFieldType();
	if(a_type==CWO_FLD_COMPLEX || b_type==CWO_FLD_COMPLEX)
		return CWO_ERROR;

	float *p_a=(float*)a.GetBuffer();
	float *p_b=(float*)b.GetBuffer();
	cwoComplex *p=(cwoComplex *)GetBuffer();

	//note: "p_a" or "p_b" is the same pointer as "p".
	//So, we need to save "p_a" or "p_b" temporally.
	float *p_tmp_a,*p_tmp_b;
	if((long int)p_a==(long int)p){
		p_tmp_a=(float*)__Malloc(GetMemSizeFloat());
		__Memcpy(p_tmp_a, p_a, GetMemSizeFloat());
		p_tmp_b=p_b;
	}
	else{// if((long int)p_b==(long int)p){
		p_tmp_a=p_a;
		p_tmp_b=(float*)__Malloc(GetMemSizeFloat());
		__Memcpy(p_tmp_b, p_b, GetMemSizeFloat());
	}
	if(GetFieldType()!=CWO_FLD_COMPLEX){
		Cplx();
		p=(cwoComplex *)GetBuffer();
		p_a=(float*)a.GetBuffer();
		p_b=(float*)b.GetBuffer();
	}

	if(b_type==CWO_FLD_PHASE)
		__Polar(p_tmp_a, p_tmp_b, p);
	/*else
		__ReIm(p_tmp_a, p_tmp_b, p);
	*/

	if((long int)p_a==(long int)p)
		__Free((void**)&p_tmp_a);
	else
		__Free((void**)&p_tmp_b);

	SetFieldType(CWO_FLD_COMPLEX);

	return CWO_SUCCESS;
}

void CWO::Arg2Cplx(float scale, float offset)
{
	__Arg2Cplx((cwoComplex*)GetBuffer(),(cwoComplex*)GetBuffer(), scale, offset);

}

void CWO::Cart2Polar()
{
	long long int Nx=GetNx();
	long long int Ny=GetNy();
	cwoComplex *p=(cwoComplex*)GetBuffer();

	for(long long int i=0;i<Nx*Ny;i++){
		float re=CWO_RE(p[i]);
		float im=CWO_IM(p[i]);
		CWO_RE(p[i])=sqrt(re*re+im*im);
		CWO_IM(p[i])=atan2(im,re);
	}

}

void CWO::Polar2Cart()
{
	long long int Nx=GetNx();
	long long int Ny=GetNy();
	cwoComplex *p=(cwoComplex*)GetBuffer();

	for(long long int i=0;i<Nx*Ny;i++){
		float amp=CWO_RE(p[i]);
		float arg=CWO_IM(p[i]);
		CWO_RE(p[i])=amp*cos(arg);
		CWO_IM(p[i])=amp*sin(arg);
	}

}

void CWO::ReIm(CWO &re, CWO &im)
{
	__ReIm((cwoComplex*)re.GetBuffer(),(cwoComplex*)im.GetBuffer(),(cwoComplex*)GetBuffer());
}

void CWO::Char()
{
	if(GetFieldType()== CWO_FLD_COMPLEX) return;

	SetFieldType(CWO_FLD_CHAR);

	float *src=(float*)GetBuffer();
	char *src_tmp=(char *)__Malloc(GetMemSizeFloat());
	
	size_t N=GetMemSizeChar();

	__FloatToChar((char *)src_tmp,(float*)src, N);

	__Memcpy((void *)src,(void *)src_tmp,N);

	__Free((void**)&src_tmp);

}

void CWO::Float()
{
	if(GetFieldType()== CWO_FLD_COMPLEX) return;

	SetFieldType(CWO_FLD_INTENSITY);

	char *src=(char*)GetBuffer();
	float *src_tmp=(float *)__Malloc(GetMemSizeFloat());
	
	size_t N=GetMemSizeFloat();

	__CharToFloat(src_tmp,src, N);

	__Memcpy((void *)src,(void *)src_tmp,N);

	__Free((void**)&src_tmp);

}


void CWO::Div(float a)
{
	cwoComplex *p=(cwoComplex*)GetBuffer();
	__Div(p,a,p);
}

void CWO::SqrtReal()
{
	cwoComplex *p=(cwoComplex *)GetBuffer();
	
	cwoSqrtReal(&ctx,p,p);
/*
#ifdef _OPENMP
	omp_set_num_threads(GetThreads());
	#pragma omp parallel for
#endif
	for(long long int i=0; i<GetNx()*GetNy()*GetNz();i++){
		CWO_RE(p[i])=sqrt(CWO_RE(p[i]));
		CWO_IM(p[i])=0.0f;
	}*/
}

void CWO::SqrtCplx()
{
	cwoComplex *p=(cwoComplex *)GetBuffer();

	cwoSqrtCplx(&ctx,p,p);
/*
#ifdef _OPENMP
	omp_set_num_threads(GetThreads());
	#pragma omp parallel for
#endif
	for(long long int i=0; i<GetNx()*GetNy()*GetNz();i++){
		float amp=sqrt(CWO_AMP(p[i]));
		float theta=CWO_PHASE(p[i])/2.0f;
		CWO_RE(p[i])=amp*cosf(theta);
		CWO_IM(p[i])=amp*sinf(theta);
	}*/
}

cwoComplex CWO::Polar(float amp, float arg)
{
	cwoComplex a;
	CWO_RE(a)=amp*cos(arg);
	CWO_IM(a)=amp*sin(arg);
	return a;
}

void CWO::SetPixel(int x, int y, float a)
{
	cwoComplex *p=(cwoComplex*)GetBuffer();
	CWO_RE(p[x+y*GetNx()])=a;
	CWO_IM(p[x+y*GetNx()])=0.0f;
}
void CWO::SetPixel(int x, int y, float amp, float ph)
{
	cwoComplex *p=(cwoComplex*)GetBuffer();
	CWO_RE(p[x+y*GetNx()])=amp*cos(ph);
	CWO_IM(p[x+y*GetNx()])=amp*sin(ph);
}
void CWO::SetPixel(int x, int y, cwoComplex a)
{	
	cwoComplex *p=(cwoComplex*)GetBuffer();
	p[x+y*GetNx()]=a;
}




void CWO::SetPixel(int x, int y, CWO &a)
{
	int sNx=a.GetNx();
	int sNy=a.GetNy();
	int dNx=GetNx();
	int dNy=GetNy();

	//SetFieldType(a.GetFieldType());
	
	cwoComplex *src=(cwoComplex*)a.GetBuffer();
	cwoComplex *dst=(cwoComplex*)GetBuffer();

	__Copy(src,0,0,sNx,sNy, dst,x,y,dNx,dNy, sNx,sNy);

/*	for(long long int i=0;i<sNy;i++){
		for(long long int j=0;j<sNx;j++){
			long long int sadr=j+i*sNx;
			long long int dadr=(x+j)+(y+i)*dNx;
			CWO_RE(dst[dadr])=CWO_RE(src[sadr]);
			CWO_IM(dst[dadr])=CWO_IM(src[sadr]);
		}
	}*/
}

void CWO::SetPixel(int x, int y, int z, float a)
{
	cwoComplex *p=(cwoComplex*)GetBuffer();
	CWO_RE(p[x+(y+z*GetNy())*GetNx()])=a;
	CWO_IM(p[x+(y+z*GetNy())*GetNx()])=0.0f;

}
void CWO::SetPixel(int x, int y, int z, float amp, float ph)
{
	cwoComplex *p=(cwoComplex*)GetBuffer();
	CWO_RE(p[x+(y+z*GetNy())*GetNx()])=amp*cos(ph);
	CWO_IM(p[x+(y+z*GetNy())*GetNx()])=amp*sin(ph);
}
void CWO::SetPixel(int x, int y, int z, cwoComplex a)
{	
	cwoComplex *p=(cwoComplex*)GetBuffer();
	p[x+(y+z*GetNy())*GetNx()]=a;
}

void CWO::AddPixel(int x, int y, cwoComplex a)
{
	long long int Nx=GetNx();
	long long int Ny=GetNy();
	cwoComplex *p=(cwoComplex*)GetBuffer();

	if(x>=0 && x<Nx && y>=0 && y< Ny){
		CWO_RE(p[x+y*GetNx()])+=CWO_RE(a);
		CWO_IM(p[x+y*GetNx()])+=CWO_IM(a);
	}
}
void CWO::AddPixel(int x, int y, CWO &a)
{
	long long int sNx=a.GetNx();
	long long int sNy=a.GetNy();
	long long int dNx=GetNx();
	long long int dNy=GetNy();

	SetFieldType(a.GetFieldType());
	
	if(GetFieldType()==CWO_FLD_COMPLEX){
		cwoComplex *src=(cwoComplex*)a.GetBuffer();
		cwoComplex *dst=(cwoComplex*)GetBuffer();
		for(long long int i=0;i<sNy;i++){
			for(long long int j=0;j<sNx;j++){
				cwoComplex tmp;
				a.GetPixel(j,i,tmp);
				AddPixel(x+j-sNx/2,y+i-sNy/2,tmp);
			}
		}
	}
	else{
		float *src=(float*)a.GetBuffer();
		float *dst=(float*)GetBuffer();
		for(long long int i=0;i<sNy;i++){
			for(long long int j=0;j<sNx;j++){
				long long int sadr=j+i*sNx;
				long long int dadr=(x+j)+(y+i)*dNx;
				dst[dadr]=src[sadr];
				
			}
		}
	}
}
void CWO::MulPixel(int x, int y, float a)
{
	long long int Nx=GetNx();
	long long int Ny=GetNy();
	cwoComplex *p=(cwoComplex*)GetBuffer();
	if(x>=0 && x<Nx && y>=0 && y< Ny){
		p[x+y*Nx]*=a;
	}
}
void CWO::MulPixel(int x, int y, cwoComplex a)
{
	long long int Nx=GetNx();
	long long int Ny=GetNy();
	cwoComplex *p=(cwoComplex*)GetBuffer();
	if(x>=0 && x<Nx && y>=0 && y< Ny)
		p[x+y*Nx]*=a;
}
void CWO::GetPixel(int x, int y, cwoComplex &a)
{
	long long int Nx=GetNx();
	long long int Ny=GetNy();
	cwoComplex *p=(cwoComplex*)GetBuffer();
	if(x>=0 && x<Nx && y>=0 && y< Ny)
		a=p[x+y*Nx];
}
void CWO::GetPixel(int x, int y, float &a)
{
	long long int Nx=GetNx();
	long long int Ny=GetNy();	
	float *p=(float*)GetBuffer();
	if(x>=0 && x<Nx && y>=0 && y< Ny)
		a=p[x+y*Nx];	
}
void CWO::GetPixel(int x, int y, int z, cwoComplex &a)
{
	long long int Nx=GetNx();
	long long int Ny=GetNy();			
	cwoComplex *p=(cwoComplex*)GetBuffer();
	if(x>=0 && x<Nx && y>=0 && y< Ny)
		a=p[x+Nx*(y+z*Ny)];
}
void CWO::GetPixel(int x, int y, int z, float &a)
{
	long long int Nx=GetNx();
	long long int Ny=GetNy();	
	float *p=(float*)GetBuffer();
	if(x>=0 && x<Nx && y>=0 && y< Ny)
		a=p[x+Nx*(y+z*Ny)];
}
cwoComplex CWO::GetPixel(int m, int n)
{
	cwoComplex ret;
	ret=0.0f;
	long long int Nx=GetNx(),Ny=GetNy();	
	cwoComplex *p=(cwoComplex*)GetBuffer();
	if(m>=0 && m<Nx && n>=0 && n< Ny) ret=p[m+Nx*n];
	return ret;
}



void CWO::__Copy(
	cwoComplex *src, int x1, int y1, int sNx, int sNy,
	cwoComplex *dst, int x2, int y2, int dNx, int dNy, 
	int Sx, int Sy)
{
#ifdef _OPENMP
	omp_set_num_threads(GetThreads());
	#pragma omp parallel for schedule(dynamic)	
#endif
	for(long long int i=0;i<Sy;i++){
		for(long long int j=0;j<Sx;j++){
			long long int sadr=(x1+j)+(y1+i)*sNx;
			long long int dadr=(x2+j)+(y2+i)*dNx;
			dst[dadr]=src[sadr];
		}
	}

}

void CWO::Copy(CWO &a, int x1, int y1, int x2, int y2, int Sx, int Sy)
{
	//a				: source
	//(x1,y1)		: start positon on source
	//(x2,y2)		: start positon on destination
	//(Sx, Sy)		: area size

	//ctx=a.ctx;
	//SetSize(Nx1,Ny1);
	
	SetFieldType(a.GetFieldType());

	int sNx=a.GetNx();
	int sNy=a.GetNy();
	int dNx=GetNx();
	int dNy=GetNy();

//	if(a.GetFieldType()==CWO_FLD_COMPLEX){
		
		cwoComplex *src=(cwoComplex*)a.GetBuffer();
		cwoComplex *dst=(cwoComplex*)GetBuffer();

		
		__Copy(
			src,x1,y1,sNx,sNy,
			dst,x2,y2,dNx,dNy,
			Sx,Sy);

/*	}
	else{
		float *src=(float*)a.GetBuffer();
		float *dst=(float*)GetBuffer();

		//__CopyFloat(
		__Copy(
			src,x1,y1,sNx,sNy,
			dst,x2,y2,dNx,dNy,
			Sx,Sy);

	}*/
}

void CWO::SetRandSeed(unsigned long s)
{
	cwoSetRandSeed(s);
}
float CWO::GetRandVal()
{
	return cwoRandVal();
}
cwoComplex CWO::GetRandComplex()
{
	cwoComplex c;
	float r=cwoRandVal();
	float ph=2*CWO_PI*r-CWO_PI;
	CWO_RE(c)=cos(ph);
	CWO_IM(c)=sin(ph);
	return c;
}

void CWO::__RandPhase(cwoComplex *a, float max, float min)
{
	cwoSetRandPhase(&ctx,a, max, min);
}
void CWO::__MulRandPhase(cwoComplex *a, float max, float min)
{
	cwoMulRandPhase(&ctx,a, max, min);
}

void CWO::RandPhase(float max, float min)
{
	__RandPhase((cwoComplex*)GetBuffer(), max, min);
}
void CWO::SetRandPhase(float max, float min)
{
	//__MulRandPhase((cwoComplex *)GetBuffer(), max, min);
	__RandPhase((cwoComplex*)GetBuffer(), max, min);
}
void CWO::MulRandPhase(float max, float min)
{
	__MulRandPhase((cwoComplex *)GetBuffer(), max, min);
}


void CWO::Clear(int c)
{
	if(GetBuffer()==NULL) return;
	__Memset(GetBuffer(),c,GetMemSizeCplx());
}


void CWO::Fill(cwoComplex a)
{
	long long int nx=GetNx();
	long long int ny=GetNy();
	long long int nz=GetNz();

	Cplx();

	cwoComplex *p=(cwoComplex *)GetBuffer();
	if(p==NULL) return;
	for(long long int i=0;i<nx*ny*nz;i++){
		p[i]=a;	
	}
}

template <class T> 
void CWO::FlipH(T *a)
{
	long long int Nx=GetNx();
	long long int Ny=GetNy();
#ifdef _OPENMP
	omp_set_num_threads(GetThreads());
	#pragma omp parallel for
#endif
	for(long long int i=0;i<Ny;i++){
		for(long long int j=0;j<Nx/2;j++){
			T tmp1=a[j+i*Nx];		
			T tmp2=a[(Nx-j-1)+i*Nx];		
			a[j+i*Nx]=tmp2;
			a[(Nx-j-1)+i*Nx]=tmp1;
		}
	}
}
template <class T> 
void CWO::FlipV(T *a)
{
	long long int Nx=GetNx();
	long long int Ny=GetNy();
#ifdef _OPENMP
	omp_set_num_threads(GetThreads());
	#pragma omp parallel for
#endif
	for(long long int i=0;i<Ny/2;i++){
		for(long long int j=0;j<Nx;j++){
			T tmp1=a[j+i*Nx];		
			T tmp2=a[j+(Ny-i-1)*Nx];		
			a[j+i*Nx]=tmp2;
			a[j+(Ny-i-1)*Nx]=tmp1;
		}
	}
}


void CWO::Flip(int mode)
{
	//if(GetFieldType()==CWO_FLD_COMPLEX){
		cwoComplex *p=(cwoComplex *)GetBuffer();
		if(mode==0){
			FlipH(p);
		}
		else if(mode==1){
			FlipV(p);			
		}

/*	}
	else{
		float *p=(float *)GetBuffer();
		if(mode==0){
			FlipH(p);		
		}
		else if(mode==1){
			FlipV(p);			
		}
		else if(mode==-1){
			FlipHV(p);
		}

	}*/

}

void CWO::Crop(int x1, int y1, int Sx, int Sy)
{
	size_t size=Sx*Sy*sizeof(cwoComplex);

	//if(GetFieldType()==CWO_FLD_COMPLEX){
		cwoComplex *p=(cwoComplex*)GetBuffer();
		cwoComplex *p_tmp=(cwoComplex*)__Malloc(size);
		//__CopyComplex(
		__Copy(
			p,x1,y1,GetNx(),GetNy(),
			p_tmp,0,0,Sx,Sy,
			Sx,Sy);
		
		__Free((void**)&p);
		SetSize(Sx,Sy);
		p_field=(void*)__Malloc(size);
		__Memcpy((cwoComplex*)GetBuffer(),p_tmp,size);
		__Free((void**)&p_tmp);
/*	}
	else{
		float *p=(float*)GetBuffer();
		float *p_tmp=(float*)__Malloc(size);
		//__CopyFloat(
		__Copy(
			p,x1,y1,GetNx(),GetNy(),
			p_tmp,0,0,Sx,Sy,
			Sx,Sy);
		
		__Free((void**)&p);
		SetSize(Sx,Sy);
		p_field=(void*)__Malloc(size);
		__Memcpy((cwoComplex*)GetBuffer(),p_tmp,size);
		__Free((void**)&p_tmp);

	}*/

}

void CWO::ShiftX(int s, int flag)
{
	int Nx=GetNx();
	int Ny=GetNy();
	int s1=abs(s);

//	if(GetFieldType()==CWO_FLD_COMPLEX){

		cwoComplex *p=(cwoComplex*)GetBuffer();
		cwoComplex *p_new=(cwoComplex*)__Malloc(GetMemSizeCplx());
		cwoComplex *p_edg=(cwoComplex*)__Malloc(s1*Ny*sizeof(cwoComplex));
		if(flag==0) __Memset(p_edg, 0, s1*Ny*sizeof(cwoComplex));

		if(s>=0){
			if(flag!=0){
				//__CopyComplex(
				__Copy(
					p,Nx-s1,0,Nx,Ny,
					p_edg,0,0,s1,Ny,
					s1,Ny
				);
				//__CopyComplex(
				__Copy(
					p_edg,0,0,s1,Ny,
					p_new,0,0,Nx,Ny,
					s1,Ny
				);
			}

			//__CopyComplex(
			__Copy(
				p,0,0,Nx,Ny,
				p_new,s1,0,Nx,Ny,
				Nx-s1,Ny
			);


		}
		else{
			if(flag!=0){
				//__CopyComplex(
				__Copy(
					p,0,0,Nx,Ny,
					p_edg,0,0,s1,Ny,
					s1,Ny
				);
				//__CopyComplex(
				__Copy(
					p_edg,0,0,s1,Ny,
					p_new,Nx-s1,0,Nx,Ny,
					s1,Ny
				);
			}

			//__CopyComplex(
			__Copy(
				p,s1,0,Nx,Ny,
				p_new,0,0,Nx,Ny,
				Nx-s1,Ny
			);


		}
			
		__Free((void**)&p_field);
		__Free((void**)&p_edg);
		
		p_field=p_new;
/*	}
	else{
		float *p=(float*)GetBuffer();
		float *p_new=(float*)__Malloc(GetMemSizeCplx());
		float *p_edg=(float*)__Malloc(s1*Ny*sizeof(float));
		if(flag==0) __Memset(p_edg, 0, s1*Ny*sizeof(float));

		if(s>=0){
			if(flag!=0){
				//__CopyComplex(
				__Copy(
					p,Nx-s1,0,Nx,Ny,
					p_edg,0,0,s1,Ny,
					s1,Ny
				);
				//__CopyComplex(
				__Copy(
					p_edg,0,0,s1,Ny,
					p_new,0,0,Nx,Ny,
					s1,Ny
				);
			}

			//__CopyComplex(
			__Copy(
				p,0,0,Nx,Ny,
				p_new,s1,0,Nx,Ny,
				Nx-s1,Ny
			);


		}
		else{
			if(flag!=0){
				//__CopyComplex(
				__Copy(
					p,0,0,Nx,Ny,
					p_edg,0,0,s1,Ny,
					s1,Ny
				);
				//__CopyComplex(
				__Copy(
					p_edg,0,0,s1,Ny,
					p_new,Nx-s1,0,Nx,Ny,
					s1,Ny
				);
			}

			//__CopyComplex(
			__Copy(
				p,s1,0,Nx,Ny,
				p_new,0,0,Nx,Ny,
				Nx-s1,Ny
			);


		}
			
		__Free((void**)&p_field);
		__Free((void**)&p_edg);
		
		p_field=p_new;

	}*/

}

void CWO::ShiftY(int s, int flag)
{
	long long int Nx=GetNx();
	long long int Ny=GetNy();
	int s1=abs(s);

//	if(GetFieldType()==CWO_FLD_COMPLEX){

		cwoComplex *p=(cwoComplex*)GetBuffer();
		cwoComplex *p_new=(cwoComplex*)__Malloc(GetMemSizeCplx());
		cwoComplex *p_edg=(cwoComplex*)__Malloc(s1*Nx*sizeof(cwoComplex));
		if(flag==0) __Memset(p_edg, 0, s1*Nx*sizeof(cwoComplex));

		if(s>=0){
			if(flag!=0){
				//__CopyComplex(
				__Copy(
					p,0,Ny-s1,Nx,Ny,
					p_edg,0,0,Nx,s1,
					Nx,s1
				);
				//__CopyComplex(
				__Copy(
					p_edg,0,0,Nx,s1,
					p_new,0,0,Nx,Ny,
					Nx,s1
				);
			}

			//__CopyComplex(
			__Copy(
				p,0,0,Nx,Ny,
				p_new,0,s1,Nx,Ny,
				Nx,Ny-s1
			);


		}
		else{
			if(flag!=0){
				//__CopyComplex(
				__Copy(
					p,0,0,Nx,Ny,
					p_edg,0,0,Nx,s1,
					Nx,s1
				);
				//__CopyComplex(
				__Copy(
					p_edg,0,0,Nx,s1,
					p_new,0,Ny-s1,Nx,Ny,
					Nx,s1
				);
			}

			//__CopyComplex(
			__Copy(
				p,0,s1,Nx,Ny,
				p_new,0,0,Nx,Ny,
				Nx,Ny-s1
			);


		}
			
		__Free((void**)&p_field);
		__Free((void**)&p_edg);
		
		p_field=p_new;
/*	}
	else{
		float *p=(float*)GetBuffer();
		float *p_new=(float*)__Malloc(GetMemSizeCplx());
		float *p_edg=(float*)__Malloc(s1*Nx*sizeof(float));
		if(flag==0) __Memset(p_edg, 0, s1*Nx*sizeof(float));

		if(s>=0){
			if(flag!=0){
				//__CopyComplex(
				__Copy(
					p,0,Ny-s1,Nx,Ny,
					p_edg,0,0,Nx,s1,
					Nx,s1
				);
				//__CopyComplex(
				__Copy(
					p_edg,0,0,Nx,s1,
					p_new,0,0,Nx,Ny,
					Nx,s1
				);
			}

			//__CopyComplex(
			__Copy(
				p,0,0,Nx,Ny,
				p_new,0,s1,Nx,Ny,
				Nx,Ny-s1
			);


		}
		else{
			if(flag!=0){
				//__CopyComplex(
				__Copy(
					p,0,0,Nx,Ny,
					p_edg,0,0,Nx,s1,
					Nx,s1
				);
				//__CopyComplex(
				__Copy(
					p_edg,0,0,Nx,s1,
					p_new,0,Ny-s1,Nx,Ny,
					Nx,s1
				);
			}

			//__CopyComplex(
			__Copy(
				p,0,s1,Nx,Ny,
				p_new,0,0,Nx,Ny,
				Nx,Ny-s1
			);


		}
			
		__Free((void**)&p_field);
		__Free((void**)&p_edg);
		
		p_field=p_new;


	}*/

}



void CWO::Transpose()
{
/*	int Nx=GetNx();
	int Ny=GetNy();
	cwoComplex *p=(cwoComplex*)GetBuffer();
	cwoComplex *p_new=(cwoComplex *)__Malloc(GetMemSizeCplx());

	for(int i=0;i<Ny;i++){
		for(int j=0;j<Nx;j++){
			p_new[i+j*Nx]=p[j+i*Nx];
		}
	}

	p_field=p_new;
	__Free((void**)&p);*/

	cwoComplex *pi=(cwoComplex*)GetBuffer();
	cwoComplex *po=(cwoComplex *)__Malloc(GetMemSizeCplx());
	__Transpose(pi,po);
	p_field=po;
	__Free((void**)&pi);

}

void CWO::__Transpose(cwoComplex *pi, cwoComplex *po){
	int Nx=GetNx();
	int Ny=GetNy();
#ifdef _OPENMP
	omp_set_num_threads(GetThreads());
	#pragma omp parallel for
#endif
	for(long long int i=0;i<Ny;i++){
		for(long long int j=0;j<Nx;j++){
			po[i+j*Nx]=pi[j+i*Nx];
		}
	}
}


void CWO::__FloatToChar(char *dst, float *src, int N)
{
	for(long long int i=0;i<N; i++){
		dst[i]=(char)src[i];
	}
}
void CWO::__CharToFloat(float *dst, char *src, int N)
{
	for(long long int i=0;i<N; i++){
		dst[i]=(float)src[i];
	}
}


void CWO::__RectFillInside(cwoComplex *p, int x, int y, int Sx, int Sy, cwoComplex a)
{
#ifdef _OPENMP
	omp_set_num_threads(GetThreads());
	#pragma omp parallel for schedule(static)	
#endif
	for(long long int i=0;i<Sy;i++){
		for(long long int j=0;j<Sx;j++){
			p[(x+j)+(y+i)*GetNx()]=a;
		}
	}
}
void CWO::__RectFillOutside(cwoComplex *p, int x, int y, int Sx, int Sy, cwoComplex a)
{
	int Nx=GetNx();
	int Ny=GetNy();

#ifdef _OPENMP
	omp_set_num_threads(GetThreads());
	#pragma omp parallel for schedule(static)	
#endif
	for(long long int i=0;i<Ny;i++){
		for(long long int j=0;j<Nx;j++){
			if(! (j>=x && j<x+Sx && i>=y && i<y+Sy) ){
				p[j+i*Nx]=a;
			}
		}
	}
}

void CWO::Rect(int x, int y, int Sx, int Sy, cwoComplex a, int flag)
{
	cwoComplex *p=(cwoComplex*)GetBuffer();
	long long int Nx=GetNx();
	long long int Ny=GetNy();

	int x1=x,y1=y;
	int Sx1=Sx,Sy1=Sy;

	if(x1<0) x1=0;
	if(x1>=Nx) x1=Nx-1;
	if(y1<0) y1=0;
	if(y1>=Ny) y1=Ny-1;

	if(Sx1>=Nx-x) Sx1=Nx-x;
	if(Sy1>=Ny-y) Sy1=Ny-y;

	if(flag==CWO_FILL_INSIDE){
		__RectFillInside(p, x1, y1, Sx1, Sy1, a);
	}
	else if(flag==CWO_FILL_OUTSIDE){
		__RectFillOutside(p, x1, y1, Sx1, Sy1, a);
	}

}

void CWO::Circ(int m, int n, int r, cwoComplex a, int flag)
{
	cwoComplex *p=(cwoComplex*)GetBuffer();
	long long int Nx=GetNx();
	long long int Ny=GetNy();
	
	if(flag==CWO_FILL_INSIDE){
		for(long long int i=0;i<Ny;i++){
			for(long long int j=0;j<Nx;j++){
				if((j-m)*(j-m)+(i-n)*(i-n)<r*r){
					SetPixel(j,i,a);
				}
			}
		}
	}
	else{
		for(long long int i=0;i<Ny;i++){
			for(long long int j=0;j<Nx;j++){
				if(!((j-m)*(j-m)+(i-n)*(i-n)<r*r)){
					SetPixel(j,i,a);
				}
			}
		}
	}
}

void CWO::MulCirc(int m, int n, int r, cwoComplex a)
{
	cwoComplex *p=(cwoComplex*)GetBuffer();
	long long int Nx=GetNx();
	long long int Ny=GetNy();

	for(long long int i=0;i<Ny;i++){
		for(long long int j=0;j<Nx;j++){
			if((j-m)*(j-m)+(i-n)*(i-n)<r*r){
				MulPixel(j,i,a);
			}
		}
	}

}


void CWO::__Hanning(cwoComplex *a, int m, int n, int Wx, int Wy) 
{
	cwoHanning(&ctx,a,m,n,Wx,Wy);
}


void CWO::__Hamming(cwoComplex *a, int m, int n, int Wx, int Wy) 
{
	cwoHamming(&ctx,a,m,n,Wx,Wy);
}

void CWO::FourierShift(float m, float n)
{
	long long int Nx=GetNx();
	long long int Ny=GetNy();

	Cplx();
	cwoComplex *p=(cwoComplex*)GetBuffer();
	for(long long int i=0;i<Ny;i++){
		for(long long int j=0;j<Nx;j++){
			float fx=(float)(j-Nx/2)/(float)(Nx);
			float fy=(float)(i-Ny/2)/(float)(Ny);
			float ph=-2.0f*CWO_PI*(fx*m+fy*n);
			cwoComplex tmp;
			tmp.Re(cos(ph));
			tmp.Im(sin(ph));
			p[j+i*Nx]*=tmp;
		}
	}
}


int CWO::FFT(int flag)
{
	Cplx();
	if(flag==-1)
		__IFFT((cwoComplex*)GetBuffer(),(cwoComplex*)GetBuffer());
	else
		__FFT((cwoComplex*)GetBuffer(),(cwoComplex*)GetBuffer(),CWO_C2C);
	
	return 0;
}


void CWO::FFTShift()
{
	__FFTShift(GetBuffer());

}

float CWO::Average()
{
	return cwoAverage(&ctx,(cwoComplex*)GetBuffer());
}

float CWO::Variance()
{
	float ave=cwoAverage(&ctx,(cwoComplex*)GetBuffer());
	return cwoVariance(&ctx,(cwoComplex*)GetBuffer(),ave);
}

void CWO::VarianceMap(int sx, int sy)
{
	size_t size=GetMemSizeCplx();
	cwoComplex *tmp_p=(cwoComplex*)__Malloc(size);
	cwoVarianceMap(&ctx,(cwoComplex*)GetBuffer(),tmp_p,sx,sy);
	void *p=GetBuffer();

	__Memcpy(p,tmp_p,size);
	__Free((void**)&tmp_p);
}


float CWO::Histogram(int *hist, int N)
{
	int Nx=GetNx();
	int Ny=GetNy();
	
	//for(int i=0;i<N;i++) hist[i]=0;
	__Memset(hist,0,sizeof(int)*N);

	float max,min;
	MaxMin(&max,&min);
	float binw=(max-min)/N;//interval of histgram

	if(binw==0.0f) return 0.0f;

	cwoComplex *p=(cwoComplex*)GetBuffer();
	for(long long int i=0;i<Nx*Ny;i++){
		int idx=(int)((CWO_RE(p[i])-min)/binw);			
		if(idx>=N) idx=N-1;
		hist[idx]++;
	}

	return binw;
}



float CWO::TotalSum()
{
	float ret=0.0f;
	cwoComplex *p=(cwoComplex*)GetBuffer();
	for(long long int i=0;i<GetNx()*GetNy();i++) ret+=CWO_RE(p[i]);		
	return ret;
}


void CWO::Log(float base)
{
	cwoComplex *a=(cwoComplex*)GetBuffer();
#ifdef _OPENMP
	omp_set_num_threads(GetThreads());
	#pragma omp parallel for schedule(static)	
#endif
	for(long long int i=0;i<GetNx()*GetNy()*GetNz();i++){
		CWO_RE(a[i])=log(CWO_RE(a[i]+1.0f))/log(base);
		CWO_IM(a[i])=0.0f;
	}
}

void CWO::Gamma(float g)
{
	cwoComplex *a=(cwoComplex*)GetBuffer();
#ifdef _OPENMP
	omp_set_num_threads(GetThreads());
	#pragma omp parallel for schedule(static)	
#endif
	for(long long int i=0;i<GetNx()*GetNy()*GetNz();i++){
		float I=pow(CWO_RE(a[i]),g);
		CWO_RE(a[i])=I;
		CWO_IM(a[i])=0.0f;
	}
}



void CWO::Threshold(float max, float min)
{
	

	cwoComplex *p=(cwoComplex*)GetBuffer();
#ifdef _OPENMP
	omp_set_num_threads(GetThreads());
	#pragma omp parallel for schedule(static)	
#endif
	for(long long int i=0;i<GetNx()*GetNy()*GetNz();i++){
		float r=CWO_RE(p[i]);
		if(r>=max) 
			CWO_RE(p[i])=max;
		else if(r<=min)
			CWO_RE(p[i])=min;
	}

}

void CWO::ThresholdAmp(float max_amp, float min_amp=0.0f){
	cwoComplex *p=(cwoComplex*)GetBuffer();
#ifdef _OPENMP
	omp_set_num_threads(GetThreads());
	#pragma omp parallel for schedule(static)	
#endif
	for(long long int i=0;i<GetNx()*GetNy()*GetNz();i++){
		float amp=CWO_AMP(p[i]);
		if(amp>=max_amp){
			float ph=CWO_ARG(p[i]);
			CWO_RE(p[i])=max_amp*cos(ph);
			CWO_IM(p[i])=max_amp*sin(ph);
		}
		else if(amp<=min_amp){
			float ph=CWO_ARG(p[i]);
			CWO_RE(p[i])=min_amp*cos(ph);
			CWO_IM(p[i])=min_amp*sin(ph);
		}
	}
}


void CWO::Binary(float th, float max, float min)
{

	cwoComplex *p=(cwoComplex*)GetBuffer();
#ifdef _OPENMP
	omp_set_num_threads(GetThreads());
	#pragma omp parallel for schedule(static)	
#endif
	for(long long int i=0;i<GetNx()*GetNy()*GetNz();i++){

		if(CWO_RE(p[i])>=th) 
			CWO_RE(p[i])=max;
		else
			CWO_RE(p[i])=min;
	}

}

void CWO::__PickupCplx(cwoComplex *src, cwoComplex *pix_p, float pix)
{
	long long int Nx=GetNx();
	long long int Ny=GetNy();
#ifdef _OPENMP
		omp_set_num_threads(GetThreads());
		#pragma omp parallel for schedule(static)	
#endif
	for(long long int i=0;i<Ny;i++){
		for(long long int j=0;j<Nx;j++){
			long long int adr=j+i*Nx;
			float pix_a=CWO_RE(pix_p[adr]);
			if(pix_a!=pix){
				CWO_RE(src[adr])=0.0f;
				CWO_IM(src[adr])=0.0f;
			}
		}
	}
}


void CWO::__PickupFloat(float *src, float *pix_p, float pix)
{
	long long int Nx=GetNx();
	long long int Ny=GetNy();
#ifdef _OPENMP
		omp_set_num_threads(GetThreads());
		#pragma omp parallel for schedule(static)	
#endif	
	for(long long int i=0;i<Ny;i++){
		for(long long int j=0;j<Nx;j++){
			long long int adr=j+i*Nx;
			float pix_a=pix_p[adr];
			if(pix_a!=pix) src[adr]=0.0f;
		}
	}
}

void CWO::Pickup(CWO *a, float pix)
{
	cwoComplex *src=(cwoComplex*)GetBuffer();
	cwoComplex *pix_p=(cwoComplex*)a->GetBuffer();
	__PickupCplx(src, pix_p, pix);


/*	int type=GetFieldType();
	
	if(type==CWO_FLD_COMPLEX){
		cwoComplex *src=(cwoComplex*)GetBuffer();
		cwoComplex *pix_p=(cwoComplex*)a->GetBuffer();
		__PickupCplx(src, pix_p, pix);
	}
	else{
		float *src=(float*)GetBuffer();
		float *pix_p=(float*)a->GetBuffer();
		__PickupFloat(src, pix_p, pix);
	}
*/

}




void CWO::Expand(int Nx, int Ny)
{
//	Cplx();
	long long int dNx=Nx;
	long long int dNy=Ny;
	long long int sNx=GetNx();
	long long int sNy=GetNy();

	int size=dNx*dNy*sizeof(cwoComplex);
	cwoComplex *dst=(cwoComplex *)__Malloc(size);
	cwoComplex *src=(cwoComplex *)GetBuffer();
	__Memset(dst,0,size);

	if(dNx>=sNx && dNy>=sNy){
		__Expand(
			src,0,0,sNx,sNy,
			dst,(dNx-sNx)/2,(dNy-sNy)/2,dNx,dNy,
			CWO_C2C);
	}
	else{
		__Expand(
			src,(sNx-dNx)/2,(sNy-dNy)/2,sNx,sNy,
			dst,0,0,dNx,dNy,
			CWO_C2C);
	}

	SetSize(dNx,dNy);

	void *tmp_p=GetBuffer();
	__Free(&tmp_p);
	p_field=dst;

}
void CWO::ExpandTwice(CWO *src)
{
	long long int Nx=src->GetNx();
	long long int Ny=src->GetNy(); 
	long long int Nx2=Nx<<1;
	long long int Ny2=Ny<<1;
		
	__Expand(
		src->GetBuffer(),0,0,Nx,Ny,
		GetBuffer(),Nx2/2-Nx/2,Ny2/2-Ny/2,Nx2,Ny2,
		CWO_C2C);
	
}
void CWO::ExpandHalf(CWO *dst)
{
	long long int Nx=dst->GetNx();
	long long int Ny=dst->GetNy();
	long long int Nx2=Nx<<1;
	long long int Ny2=Ny<<1;

	__Expand(
		GetBuffer(),Nx2/2-Nx/2,Ny2/2-Ny/2,Nx2,Ny2,
		dst->GetBuffer(),0,0,Nx,Ny,
		CWO_C2C);
}
void CWO::__MaxMin(cwoComplex *a, float *max, float *min, int *max_x, int *max_y,int *min_x, int *min_y)
{
	cwoComplex *p=(cwoComplex*)GetBuffer();
	cwoMaxMin(&ctx,p,max,min,max_x,max_y,min_x,min_y);

/*	float tmax=CWO_RE(a[0]);
	float tmin=CWO_RE(a[0]);
	*max=tmax;
	*min=tmin;

	int Nx=GetNx();
	int Ny=GetNy();

	if(max_x!=NULL) *max_x=0;
	if(max_y!=NULL) *max_y=0;
	if(min_x!=NULL) *min_x=0;
	if(min_y!=NULL) *min_y=0;

	for(int y=0;y<Ny;y++){
		for(int x=0;x<Nx;x++){

			long long int adr=x+y*Nx;
			float r=CWO_RE(a[adr]);

			if(tmax<r){
				tmax=r;
				*max=tmax;
				if(max_x!=NULL) *max_x=x;
				if(max_y!=NULL) *max_y=y;
			}
			
			if(tmin>r){
				tmin=r;
				*min=tmin;
				if(min_x!=NULL) *min_x=x;
				if(min_y!=NULL) *min_y=y;
			}
		}
			
	}	

	//*max=tmax;
	//*min=tmin;*/
}

/*
int CWO::MaxMin(
	float *a,
	float *max, float *min, 
	int *max_x, int *max_y,
	int *min_x, int *min_y)
{
	

	return 0;
}*/

float CWO::Max()
{
	float max,min;
	__MaxMin((cwoComplex*)GetBuffer(),&max,&min);
	return max;
}
float CWO::Min()
{
	float max,min;
	__MaxMin((cwoComplex*)GetBuffer(),&max,&min);
	return min;
}

int CWO::MaxMin(
	float *max, float *min, 
	int *max_x, int *max_y,
	int *min_x, int *min_y)
{
	//if(GetFieldType()==CWO_FLD_COMPLEX) return CWO_ERROR;
	__MaxMin((cwoComplex*)GetBuffer(),max,min,max_x,max_y,min_x,min_y);
	return CWO_SUCCESS;
}

/*
void CWO::__Quant(float lim, float max, float min)
{
	//quantization 
	cwoComplex *p=(cwoComplex *)GetBuffer();
#ifdef _OPENMP
		omp_set_num_threads(GetThreads());
		#pragma omp parallel for schedule(static)	
#endif
	for(long long int i=0;i<GetNx()*GetNy()*GetNz();i++){
		CWO_RE(p[i])=lim*(CWO_RE(p[i])-min)/(max-min);
	}		
}*/

int CWO::ScaleReal(float lim)
{
	//-1 p_field includes not real value
	//0 success

	float max,min;

	__MaxMin((cwoComplex *)GetBuffer(),&max,&min);
		
/*	if(max!=min)
		__Quant(lim, max, min);
	else 
		__Quant(lim, max, 0);
*/
	if(max!=min)
		__ScaleReal(min, max, 0, lim);
	else
		__ScaleReal(0, max, 0, lim);

	return CWO_SUCCESS;
}
int CWO::ScaleReal(float i1, float i2, float o1, float o2)
{
/*	cwoComplex *p=(cwoComplex*)GetBuffer();
	cwoScaleReal(&ctx,p,p,i1,i2,o1,o2);

	return CWO_SUCCESS;*/
	return __ScaleReal(i1, i2, o1, o2);
}

int CWO::__ScaleReal(float i1, float i2, float o1, float o2)
{
	cwoComplex *p=(cwoComplex*)GetBuffer();
	cwoScaleReal(&ctx,p,p,i1,i2,o1,o2);

	return CWO_SUCCESS;
}

int CWO::ScaleCplx(float lim)
{
	cwoComplex *a=(cwoComplex *)GetBuffer();
	float max,min;

	cwoComplex *amp_fld=(cwoComplex *)__Malloc(GetMemSizeCplx());
	if(amp_fld==NULL) return CWO_ERROR;

	__Amp(a,amp_fld);
	__MaxMin(amp_fld,&max,&min);
	__Free((void**)&amp_fld);

	__Mul(a,lim/max,a);

/*	max=fabs(max);
	min=fabs(min);

	float amax;
	(max>min)?(amax=max):(amax=min); 
		
	__Mul(a,lim/amax,a);
*/
	return CWO_SUCCESS;
}

int CWO::ScaleCplx(float i1, float i2, float o1, float o2)
{
	/*cwoComplex *p=(cwoComplex*)GetBuffer();
	cwoScaleCplx(&ctx,p,p,i1,i2,o1,o2);

	return CWO_SUCCESS;*/
	return __ScaleCplx(i1, i2, o1, o2);
}
int CWO::__ScaleCplx(float i1, float i2, float o1, float o2)
{
	cwoComplex *p=(cwoComplex*)GetBuffer();
	cwoScaleCplx(&ctx,p,p,i1,i2,o1,o2);

	return CWO_SUCCESS;
}


float CWO::SNR(CWO &ref)
{
	if(GetNx()!=ref.GetNx() || GetNy()!=ref.GetNy()) return CWO_ERROR;
	int Nx=GetNx();
	int Ny=GetNy();
	cwoComplex *p_tar=(cwoComplex*)GetBuffer();
	cwoComplex *p_ref=(cwoComplex*)ref.GetBuffer();
		
	//scaling factor
	cwoComplex scale_a;
	float scale_b=0.0f;
	scale_a=0.0f;
	for(long long int i=0;i<Nx*Ny;i++){
		scale_a+=p_ref[i].Conj()*p_tar[i];
		scale_b+=p_ref[i].Intensity();
	}
	cwoComplex alpha=scale_a/scale_b;
	printf("alpha re=%e im=%e\n",CWO_RE(alpha),CWO_IM(alpha));

	float snr_a=0.0f;
	float snr_b=0.0f;
	cwoComplex tmp;
	for(long long int i=0;i<Nx*Ny;i++){
		snr_a+=p_tar[i].Intensity();
		tmp=p_tar[i]-p_ref[i]*alpha;
		snr_b+=tmp.Intensity();
	}

	return 10.0f*log10(snr_a/snr_b);

}

float CWO::MSE(CWO &ref)
{
	if(GetNx()!=ref.GetNx() || GetNy()!=ref.GetNy()) return -1.0f;
	cwoComplex *p_tar=(cwoComplex*)GetBuffer();
	cwoComplex *p_ref=(cwoComplex*)ref.GetBuffer();
	float mse=0.0f;

#ifdef _OPENMP
	omp_set_num_threads(GetThreads());
	#pragma omp parallel for reduction (+:mse)
#endif
	for(long long int i=0;i<GetNx()*GetNy();i++){
		float t=CWO_RE(p_tar[i]);
		float r=CWO_RE(p_ref[i]);
		mse+=(t-r)*(t-r);
	}

	return mse/(float)(GetNx()*GetNy());
}

float CWO::PSNR(CWO &ref)
{
	float mse=MSE(ref);
	if(mse==-1.0f) return -1.0f;

	return 10.0f*log10(255*255/mse);

}



/////////////////////////////
void CWO::PLS(int flag)
{
	switch(flag){
		case CWO_PLS_FRESNEL:
			__PLS_Fresnel();
			break;
		case CWO_PLS_FRESNEL_CGH:
			__PLS_CGH_Fresnel();
			break;
		case CWO_PLS_HUYGENS:
			__PLS_Huygens();
			break;
	}
	
}



///////////////////////////////////////////////
// test code
// 
///////////////////////////////////////////////

void CWO::test()
{
	printf("%s\n",__FUNCTION__);
}

void CWO::test2(CWO &a)
{

	a.test();
}


void CWO::__ScaledFFTCoeff(cwoComplex *p, float sx, float sy)
{
	long long int Nx=GetNx();
	long long int Ny=GetNy();
#ifdef _OPENMP
		omp_set_num_threads(GetThreads());
		#pragma omp parallel for schedule(static)	
#endif
	for(long long int i=0;i<Ny;i++){
		for(long long int j=0;j<Nx;j++){
			float dx=(j-Nx/2);
			float dy=(i-Ny/2);
			float ph=sx*dx*dx+sy*dy*dy;
			cwoComplex tmp;
			tmp.Re(cos(ph));
			tmp.Im(sin(ph));
			p[j+i*Nx]*=tmp;
		}
	}

}

void CWO::__ScaledFFTKernel(cwoComplex *p, float sx, float sy)
{
	long long int Nx=GetNx();
	long long int Ny=GetNy();
#ifdef _OPENMP
	omp_set_num_threads(GetThreads());
	#pragma omp parallel for schedule(static)	
#endif
	for(long long int i=0;i<Ny;i++){
		for(long long int j=0;j<Nx;j++){
			float dx=(j-Nx/2);
			float dy=(i-Ny/2);
			float ph=sx*dx*dx+sy*dy*dy;
			cwoComplex tmp;
			tmp.Re(cos(ph));
			tmp.Im(sin(ph));
			p[j+i*Nx]=tmp;
		}
	}
}

void CWO::__NUFFT_T1(cwoComplex *p_fld, cwoFloat2 *p_x, int R, int Msp)
{
	cwoNUFFT_T1(&ctx, p_fld, p_x, R, Msp);
}


void CWO::__NUFFT_T2(cwoComplex *p_fld, cwoFloat2 *p_x, int R, int Msp)
{
	cwoNUFFT_T2(&ctx, p_fld, p_x, R, Msp);
}




void CWO::NUFFT_T1(int R, int Msp)
{
	cwoComplex *p_fld=(cwoComplex*)GetBuffer();
	cwoFloat2 *p_x1=(cwoFloat2 *)GetBuffer(CWO_BUFFER_X1);//íçà”ÅIÇP

	__NUFFT_T1(p_fld,p_x1,R,Msp);
}

void CWO::NUFFT_T2(int R, int Msp)
{
	cwoComplex *p_fld=(cwoComplex*)GetBuffer();
	cwoFloat2 *p_x1=(cwoFloat2 *)GetBuffer(CWO_BUFFER_X1);//íçà”ÅIÇP

	__NUFFT_T2(p_fld,p_x1,R,Msp);
}


void CWO::NUFFT1(CWO *map, int R, int Msp)
{
	cwoComplex *p_fld=(cwoComplex*)GetBuffer();
	cwoFloat2 *p_x1=(cwoFloat2 *)map->GetBuffer();//íçà”ÅIÇP

	__NUFFT_T1(p_fld,p_x1,R,Msp);
}

void CWO::NUFFT2(CWO *map, int R, int Msp)
{
	cwoComplex *p_fld=(cwoComplex*)GetBuffer();
	cwoFloat2 *p_x1=(cwoFloat2 *)map->GetBuffer();//íçà”ÅIÇP

	__NUFFT_T2(p_fld,p_x1,R,Msp);
}

void CWO::SamplingMapScaleOnly(int Nx, int Ny, float R, float sgn)
{
	Create(Nx,Ny);
	cwoFloat2 *p=(cwoFloat2*)GetBuffer();

#ifdef _OPENMP
	omp_set_num_threads(GetThreads());
	#pragma omp parallel for
#endif
	for(long long int y=0;y<Ny;y++){
		for(long long int x=0;x<Nx;x++){
			float lx,ly;
			if(x>=Nx/4 && x<Nx/4*3 && y>=Ny/4 && y<Ny/4*3){
				lx=(x-Nx/2)*R + Nx/2;
				ly=(y-Ny/2)*R + Ny/2;
			}else{
				lx=(x-Nx/2) + Nx/2;
				ly=(y-Ny/2) + Ny/2;
			}
			long long int idx=x+y*Nx;	

			p[idx].x=sgn*lx*2*CWO_PI/Nx;
			p[idx].y=sgn*ly*2*CWO_PI/Ny;
		}
	}
}


void CWO::ConvertSamplingMap(int type)
{
	long long int Nx=GetNx();
	long long int Ny=GetNy();
	long long int Nx2=Nx<<1;
	long long int Ny2=Ny<<1;

	cwoFloat2 *new_map=(cwoFloat2*)__Malloc(Nx2*Ny2*sizeof(cwoFloat2));
	cwoFloat2 *old_map=(cwoFloat2*)GetBuffer();

	float sign=1.0f;
	if(type==CWO_NU_ANGULAR2 || type==CWO_NU_FRESNEL2)
		sign=-1.0f;

	SetSize(Nx2,Ny2);

#ifdef _OPENMP
	omp_set_num_threads(GetThreads());
	#pragma omp parallel for	
#endif
	for(long long int y=0;y<Ny2;y++){
		for(long long int x=0;x<Nx2;x++){
			float lx,ly;

			if(x>=Nx2/4 && x<Nx2/4+Nx && y>=Ny2/4 && y<Ny2/4+Ny){
				size_t old_adr=(x-Nx2/4)+(y-Ny2/4)*Nx;
				lx=old_map[old_adr].x+Nx2/2;
				ly=old_map[old_adr].y+Ny2/2;
			}
			else{
				lx=(x-Nx/2)+Nx2/2;
				ly=(y-Ny/2)+Ny2/2;
			}
						
			new_map[x+y*Nx2].x=sign*lx*2*CWO_PI/Nx2;
			new_map[x+y*Nx2].y=sign*ly*2*CWO_PI/Ny2;
		
		}
	}
	__Free((void**)&old_map);
	p_field=(void*)new_map;

}



void CWO::NUDFT()
{
	long long int Nx=GetNx();
	long long int Ny=GetNy();
	
	cwoFloat2 *p_x1=(cwoFloat2 *)GetBuffer(CWO_BUFFER_X1);//íçà”ÅIÇP

//for(int k=0;k<Nx;k++) printf("NUDFT x1 %e %e \n",p_x1[k].x,p_x1[k].y);

	cwoComplex *p_fld=(cwoComplex*)GetBuffer();
	cwoComplex *p_tmp=(cwoComplex*)new cwoComplex[Nx*Ny];
	__Memcpy(p_tmp,p_fld,Nx*Ny*sizeof(cwoComplex));

	#pragma omp parallel for schedule(static) num_threads(4)
	for(long long int m=0;m<Ny;m++){
		for(long long int n=0;n<Nx;n++){
			long long int adr_dst=n+m*Nx;
			p_fld[adr_dst]=0.0f;

			for(long long int i=0;i<Ny;i++){
				for(long long int j=0;j<Nx;j++){
					long long int adr_src=j+i*Nx;
					cwoComplex e;
					float ph=p_x1[adr_src].x*(n-Nx/2) + p_x1[adr_src].y*(m-Ny/2);
					CWO_RE(e)=cos(-ph);
					CWO_IM(e)=sin(-ph);
					p_tmp[adr_src]*=e;
					p_fld[adr_dst]+=p_tmp[adr_src];

				}
			}
		}
		//printf("loop %d\n",m);
	}


delete []p_tmp;

}

void CWO::__ResizeNearest(
	cwoComplex *p_new, int dNx, int dNy, 
	cwoComplex *p_old, int sNx, int sNy)
{
	cwoResizeNearest(&ctx, p_new, dNx, dNy, p_old, sNx,sNy);
}

void CWO::__ResizeLinear(
	cwoComplex *p_new, int dNx, int dNy, 
	cwoComplex *p_old, int sNx, int sNy)
{
	cwoResizeLinear(&ctx, p_new, dNx, dNy, p_old, sNx,sNy);
}
void CWO::__ResizeCubic(
	cwoComplex *p_new, int dNx, int dNy, 
	cwoComplex *p_old, int sNx, int sNy)
{
	cwoResizeCubic(&ctx, p_new, dNx, dNy, p_old, sNx,sNy);
}

void CWO::__ResizeLanczos(
	cwoComplex *p_new, int dNx, int dNy, 
	cwoComplex *p_old, int sNx, int sNy)
{
	cwoResizeLanczos(&ctx, p_new, dNx, dNy, p_old, sNx,sNy);
}


void CWO::AddNoiseWhite()
{
	cwoComplex *p=(cwoComplex*)GetBuffer();
	for(long long int i=0;i<GetNx()*GetNy();i++){
		CWO_RE(p[i])+=GetRandVal();
	}
}

void CWO::MulNoiseWhite()
{
	cwoComplex *p=(cwoComplex*)GetBuffer();
	for(long long int i=0;i<GetNx()*GetNy();i++){
		CWO_RE(p[i])*=GetRandVal();
	}
}


void CWO::AddNoiseGaussian(float mu, float sigma)
{
 
	cwoComplex *p=(cwoComplex*)GetBuffer();
	for(long long int i=0;i<GetNx()*GetNy();i++){
		float z=sqrt(-2.0*log(GetRandVal())) * sin(2.0*CWO_PI*GetRandVal());
		float r=mu+sigma*z;
		CWO_RE(p[i])+=r;
	}
}
void CWO::MulNoiseGaussian(float mu, float sigma)
{
	cwoComplex *p=(cwoComplex*)GetBuffer();
	for(long long int i=0;i<GetNx()*GetNy();i++){
		float z=sqrt( -2.0*log(GetRandVal())) * sin(2.0*CWO_PI*GetRandVal());
		float r=mu+sigma*z;
		CWO_RE(p[i])*=r;
	}
}


void CWO::Resize(int dNx, int dNy, int flag)
{
	int sNx=GetNx();
	int sNy=GetNy();

	if(sNx==dNx && sNy==dNy) return;

	cwoComplex *p_old=(cwoComplex*)GetBuffer();
	cwoComplex *p_new=(cwoComplex*)__Malloc(dNx*dNy*sizeof(cwoComplex));

	if(flag==CWO_INTER_NEAREST){
		__ResizeNearest(
			p_new, dNx, dNy,
			p_old, sNx, sNy);
	}
	else if(flag==CWO_INTER_LINEAR){
		__ResizeLinear(
			p_new, dNx, dNy,
			p_old, sNx, sNy);
	}
	else if(flag==CWO_INTER_CUBIC){
		__ResizeCubic(
			p_new, dNx, dNy,
			p_old, sNx, sNy);

	}
	else if(flag==CWO_INTER_LANCZOS){
		__ResizeLanczos(
			p_new, dNx, dNy,
			p_old, sNx, sNy);

	}

	__Free((void**)&p_old);
	p_field=p_new;
	SetSize(dNx,dNy);

}


void CWO::Rotate(float deg)
{




}

void CWO::AffineAngularSpectrum(float *mat_affine, float px, float py, int flag){

	cwoComplex *p_old=(cwoComplex*)GetBuffer();
	cwoComplex *p_new=(cwoComplex*)__Malloc(GetNx()*GetNx()*sizeof(cwoComplex));
	float wl=GetWaveLength();
	long long int Nx=GetNx();
	long long int Ny=GetNy();

	cwoAffineAngularSpectrum(&ctx,p_new,p_old,Nx,Ny,px,py,wl,mat_affine,flag);
	
	__Free(&p_field);
	p_field=p_new;
}


void CWO::ErrorDiffusion(CWO *a, int flag)
{
/*	int Nx=GetNx();
	int Ny=GetNy();
	a->Cplx();

	cwoComplex *p1=(cwoComplex*)GetBuffer();
	cwoComplex *p2=(cwoComplex*)a->GetBuffer();

	cwoComplex *err=(cwoComplex *)__Malloc((Nx+2)*2);

	for(int i=0;i<Ny;i++){
		for(int j=0;j<Nx;j++){
			size_t adr=j+i*Nx;
			cwoComplex e=p1[adr]-p2[adr];
			
			if(j != Nx-1)p1[(j+1)+i*Nx] += e*Polar((7.0f/16),0.0f); 
			if(j !=0 && i !=Ny-1) p1[(j-1)+(i+1)*Nx] += e*Polar((3.0f/16),0.0f);
			if(i !=Ny-1) p1[j+(i+1)*Nx] += e*Polar((5.0f/16),0.0f);
			if(j != Nx-1 && i!=Ny-1) p1[(j+1)+(i+1)*Nx] += e*Polar((1.0f/16),0.0f);
		}
	}

	__Free(&err);
	*/


	long long int Nx=GetNx();
	long long int Ny=GetNy();
	a->Cplx();

	cwoComplex *p1=(cwoComplex*)GetBuffer();
	cwoComplex *p2=(cwoComplex*)a->GetBuffer();
//
//	cwoComplex *out=(cwoComplex*)__Malloc(Nx*Ny*sizeof(cwoComplex));
	

	for(long long int i=0;i<Ny;i++){
		for(long long int j=0;j<Nx;j++){
			long long int adr=j+i*Nx;
			cwoComplex e=p1[adr]-p2[adr];
			
			if(j != Nx-1)p1[(j+1)+i*Nx]+= e*(7.0f/16); 
			if(j !=0 && i !=Ny-1) p1[(j-1)+(i+1)*Nx] +=e*(3.0f/16);
			if(i !=Ny-1) p1[j+(i+1)*Nx] +=e*(5.0f/16);
			if(j != Nx-1 && i!=Ny-1) p1[(j+1)+(i+1)*Nx]+=e*(1.0f/16);
		}
	}

/*	int mx=Nx+2;
	cwoComplex *err=(cwoComplex*)__Malloc(mx*2);
	__Memset(err,0,mx*2);

	for(int i=0;i<Ny;i++){
		for(int j=0;j<Nx;j++){
			int eadr=j+1;
			size_t adr=j+i*Nx;
					
			cwoComplex re;
			re=p1[adr]-p2[adr];
			
			err[eadr+1]+=re*7.0f/16.0f;
			err[eadr+mx-1]+=re*3.0f/16.0f;
			err[eadr+mx]+=re*5.0f/16.0f;
			err[eadr+mx+1]+=re;

			out[adr]=err[eadr];
		}

		for(int ii=0;ii<mx;ii++){
			err[ii]=err[ii+mx];
			err[ii+mx]=Polar(0,0);
		}

	}
*/
	//p_field=out;
	//__Free((void**)&p1);
	//__Free((void**)&err);
}

void CWO::RGB2YCbCr(CWO *rgb, CWO *ycbcr)
{
/*
	ycbcr[0]=rgb[0]*0.2989+rgb[1]*0.5866+rgb[2]*0.1145;
	ycbcr[1]=rgb[0]*-0.1687+rgb[1]*-0.3312+rgb[2]*0.5;
	ycbcr[2]=rgb[0]*0.5+rgb[1]*-0.4183+rgb[2]*-0.0816;
*/

/*
	ycbcr[0]=rgb[0]*0.299+rgb[1]*0.587+rgb[2]*0.114;
ycbcr[0]+=16.0;
	ycbcr[1]=rgb[0]*-0.169+rgb[1]*-0.331+rgb[2]*0.5;
ycbcr[1]+=128.0;
	ycbcr[2]=rgb[0]*0.5+rgb[1]*-0.419+rgb[2]*-0.081;
ycbcr[2]+=128.0;
*/

	/*	
	ycbcr[0]=rgb[0]*0.257+rgb[1]*0.504+rgb[2]*0.098;
ycbcr[0]+=16.0;
	ycbcr[1]=rgb[0]*-0.148+rgb[1]*-0.291+rgb[2]*0.439;
ycbcr[1]+=128.0;
	ycbcr[2]=rgb[0]*0.439+rgb[1]*-0.368+rgb[2]*-0.071;
ycbcr[2]+=128.0;
*/
/*
	ycbcr[0]=rgb[0]*0.299+rgb[1]*0.587+rgb[2]*0.114;
	ycbcr[1]=rgb[0]*-0.169+rgb[1]*-0.331+rgb[2]*0.500;
	ycbcr[2]=rgb[0]*0.500+rgb[1]*-0.419+rgb[2]*-0.081;
*/

	long long int Nx=rgb[0].GetNx();
	long long int	Ny=rgb[0].GetNy();
	int fld=rgb[0].GetFieldType();

	for(int i=0;i<3;i++) ycbcr[i].Create(Nx,Ny);
	
	cwoComplex *r=(cwoComplex*)rgb[0].GetBuffer();
	cwoComplex *g=(cwoComplex*)rgb[1].GetBuffer();
	cwoComplex *b=(cwoComplex*)rgb[2].GetBuffer();
	cwoComplex *y =(cwoComplex*)ycbcr[0].GetBuffer();
	cwoComplex *cb=(cwoComplex*)ycbcr[1].GetBuffer();
	cwoComplex *cr=(cwoComplex*)ycbcr[2].GetBuffer();

#ifdef _OPENMP
	omp_set_num_threads(GetThreads());
	#pragma omp parallel for schedule(static)	
#endif
/*	for(long long int i=0;i<Nx*Ny;i++){
		y[i]=r[i]*0.2989+g[i]*0.5866+b[i]*0.1145;
		cb[i]=r[i]*-0.1687+g[i]*-0.3312+b[i]*0.5;
		cr[i]=r[i]*0.5+g[i]*-0.4183+b[i]*-0.0816;
	}
*/	for(long long int i=0;i<Nx*Ny;i++){
		y[i]=r[i]*0.299f+g[i]*0.587f+b[i]*0.114f;
		cb[i]=r[i]*-0.14713f+g[i]*-0.28886f+b[i]*0.436f;
		cr[i]=r[i]*0.615f+g[i]*-0.51499f+b[i]*-0.10001f;
	}

}
void CWO::YCbCr2RGB(CWO *rgb, CWO *ycbcr)
{
/*
	rgb[0]=ycbcr[0]+ycbcr[2]*1.4022;
	rgb[1]=ycbcr[0]+ycbcr[1]*-0.3456+ycbcr[2]*-0.7145;
	rgb[2]=ycbcr[0]+ycbcr[1]*1.7710;
*/


/*
ycbcr[0]-=16.0;
ycbcr[0]*=1.164;
ycbcr[1]-=128.0;
ycbcr[2]-=128.0;
	rgb[0]=(ycbcr[0])+(ycbcr[2])*1.596;
	rgb[1]=(ycbcr[0])+(ycbcr[1])*-0.391+(ycbcr[2])*-0.813;
	rgb[2]=(ycbcr[0])+(ycbcr[1])*2.018;
*/
/*

	rgb[0]=ycbcr[0]+ycbcr[2]*1.402;
	rgb[1]=ycbcr[0]+ycbcr[1]*-0.344+ycbcr[2]*-0.714;
	rgb[2]=ycbcr[0]+ycbcr[1]*1.772;
*/




	long long int Nx=ycbcr[0].GetNx();
	long long int Ny=ycbcr[0].GetNy();
	int fld=ycbcr[0].GetFieldType();

	for(int i=0;i<3;i++) rgb[i].Create(Nx,Ny);

	
	cwoComplex *r=(cwoComplex*)rgb[0].GetBuffer();
	cwoComplex *g=(cwoComplex*)rgb[1].GetBuffer();
	cwoComplex *b=(cwoComplex*)rgb[2].GetBuffer();
	cwoComplex *y =(cwoComplex*)ycbcr[0].GetBuffer();
	cwoComplex *cb=(cwoComplex*)ycbcr[1].GetBuffer();
	cwoComplex *cr=(cwoComplex*)ycbcr[2].GetBuffer();

	#ifdef _OPENMP
		omp_set_num_threads(GetThreads());
		#pragma omp parallel for schedule(static)	
	#endif
	for(long long int i=0;i<Nx*Ny;i++){
		r[i]=(y[i])                +(cr[i])*1.13983f;
		g[i]=(y[i])+(cb[i])*-0.39465f+(cr[i])*-0.5806f;
		b[i]=(y[i])+(cb[i])*2.03211f;
	}

}



void CWO::MulMatrix(CWO *a, cwoComplex *b , CWO *c)
{
	int Nx=a[0].GetNx();
	int	Ny=a[0].GetNy();
	
	for(int i=0;i<3;i++) c[i].Create(Nx,Ny);
	
/*	c[0]=a[0]*b[0];
	c[0]+=a[1]*b[1];
	c[0]+=a[2]*b[2];
		
	c[1]=a[0]*b[3];
	c[1]+=a[1]*b[4];
	c[1]+=a[2]*b[5];

	c[2]=a[0]*b[6];
	c[2]+=a[1]*b[7];
	c[2]+=a[2]*b[8];
*/
	cwoComplex *p_c[3], *p_a[3];
	for(int i=0;i<3;i++){
		p_c[i]=(cwoComplex*)c[i].GetBuffer();
		p_a[i]=(cwoComplex*)a[i].GetBuffer();
	}

	for(int k=0;k<3;k++){
		if(!(CWO_RE(b[k])== 0.0f && CWO_IM(b[k])==0.0f)){
		#ifdef _OPENMP
			omp_set_num_threads(GetThreads());
			#pragma omp parallel for schedule(static)	
		#endif
			for(long long int i=0;i<Ny;i++){
				for(long long int j=0;j<Nx;j++){
					*(p_c[0]+j+i*Nx)+=*(p_a[k]+j+i*Nx)*b[k];
				}
			}
		}
	}
	for(int k=0;k<3;k++){
		if(!(CWO_RE(b[k+3])== 0.0f && CWO_IM(b[k+3])==0.0f)){
		#ifdef _OPENMP
			omp_set_num_threads(GetThreads());
			#pragma omp parallel for schedule(static)	
		#endif
			for(long long int i=0;i<Ny;i++){
				for(long long int j=0;j<Nx;j++){
					*(p_c[1]+j+i*Nx)+=*(p_a[k]+j+i*Nx)*b[3+k];
				}
			}
		}
	}
	for(int k=0;k<3;k++){
		if(!(CWO_RE(b[k+6])== 0.0f && CWO_IM(b[k+6])==0.0f)){
		#ifdef _OPENMP
			omp_set_num_threads(GetThreads());
			#pragma omp parallel for schedule(static)	
		#endif
			for(long long int i=0;i<Ny;i++){
				for(long long int j=0;j<Nx;j++){
					*(p_c[2]+j+i*Nx)+=*(p_a[k]+j+i*Nx)*b[6+k];
				}
			}
		}
	}



}	


void CWO::__InvFx2Fy2(cwoComplex *a)
{
	long long int Nx=GetNx();
	long long int Ny=GetNy();
	float px=GetSrcPx();
	float py=GetSrcPy();

	int x,y;
	float dx;
	float dy;
	float fx=1.0f/((float)Nx*px);
	float fy=1.0f/((float)Ny*py);

	for(y=0;y<Ny;y++){
		for(x=0;x<Nx;x++){
			long long int adr = (x+y*Nx);
							
			//éñè€ïœä∑
			if(x < Nx/2 && y < Ny/2){
				dx=(float)(x) * fx;
				dy=(float)(y) * fy;
			}
			else if(x >= Nx/2 && y < Ny/2){
				dx=(float)(x -Nx) * fx;
				dy=(float)(y) * fy;
			}
			else if(x < Nx/2 && y >= Ny/2){
				dx=(float)(x) * fx;
				dy=(float)(y - Ny) * fy;
			}
			else{
				dx=(float)(x - Nx) * fx;
				dy=(float)(y - Ny) * fy;
			}
			
			double fxfy;
			if(dx==0.0 && dy==0.0)
				fxfy=-1;
			else{
				fxfy=-1/(dx*dx+dy*dy);
				
			}
			
			if(fabs(fxfy)>fabs(1/(1000*dx*dx))) fxfy=-1;

		
			a[adr]*=fxfy;

		
		}
	}

}



float costbl[256]={1.000000f,0.999699f,0.998795f,0.997290f,0.995185f,0.992480f,0.989177f,0.985278f,0.980785f,0.975702f,0.970031f,0.963776f,0.956940f,0.949528f,0.941544f,0.932993f,0.923880f,0.914210f,0.903989f,0.893224f,0.881921f,0.870087f,0.857729f,0.844854f,0.831470f,0.817585f,0.803208f,0.788346f,0.773010f,0.757209f,0.740951f,0.724247f,0.707107f,0.689541f,0.671559f,0.653173f,0.634393f,0.615232f,0.595699f,0.575808f,0.555570f,0.534998f,0.514103f,0.492898f,0.471397f,0.449611f,0.427555f,0.405241f,0.382683f,0.359895f,0.336890f,0.313682f,0.290285f,0.266713f,0.242980f,0.219101f,0.195090f,0.170962f,0.146730f,0.122411f,0.098017f,0.073565f,0.049068f,0.024541f,-0.000000f,-0.024541f,-0.049068f,-0.073565f,-0.098017f,-0.122411f,-0.146731f,-0.170962f,-0.195090f,-0.219101f,-0.242980f,-0.266713f,-0.290285f,-0.313682f,-0.336890f,-0.359895f,-0.382683f,-0.405241f,-0.427555f,-0.449611f,-0.471397f,-0.492898f,-0.514103f,-0.534998f,-0.555570f,-0.575808f,-0.595699f,-0.615232f,-0.634393f,-0.653173f,-0.671559f,-0.689541f,-0.707107f,-0.724247f,-0.740951f,-0.757209f,-0.773010f,-0.788346f,-0.803208f,-0.817585f,-0.831470f,-0.844854f,-0.857729f,-0.870087f,-0.881921f,-0.893224f,-0.903989f,-0.914210f,-0.923880f,-0.932993f,-0.941544f,-0.949528f,-0.956940f,-0.963776f,-0.970031f,-0.975702f,-0.980785f,-0.985278f,-0.989177f,-0.992480f,-0.995185f,-0.997290f,-0.998795f,-0.999699f,-1.000000f,-0.999699f,-0.998795f,-0.997290f,-0.995185f,-0.992480f,-0.989177f,-0.985278f,-0.980785f,-0.975702f,-0.970031f,-0.963776f,-0.956940f,-0.949528f,-0.941544f,-0.932993f,-0.923880f,-0.914210f,-0.903989f,-0.893224f,-0.881921f,-0.870087f,-0.857729f,-0.844854f,-0.831470f,-0.817585f,-0.803207f,-0.788346f,-0.773010f,-0.757209f,-0.740951f,-0.724247f,-0.707107f,-0.689540f,-0.671559f,-0.653173f,-0.634393f,-0.615232f,-0.595699f,-0.575808f,-0.555570f,-0.534998f,-0.514103f,-0.492898f,-0.471397f,-0.449611f,-0.427555f,-0.405241f,-0.382683f,-0.359895f,-0.336890f,-0.313682f,-0.290285f,-0.266713f,-0.242980f,-0.219101f,-0.195090f,-0.170962f,-0.146730f,-0.122411f,-0.098017f,-0.073564f,-0.049068f,-0.024541f,0.000000f,0.024541f,0.049068f,0.073565f,0.098017f,0.122411f,0.146731f,0.170962f,0.195090f,0.219101f,0.242980f,0.266713f,0.290285f,0.313682f,0.336890f,0.359895f,0.382684f,0.405241f,0.427555f,0.449611f,0.471397f,0.492898f,0.514103f,0.534998f,0.555570f,0.575808f,0.595699f,0.615232f,0.634393f,0.653173f,0.671559f,0.689541f,0.707107f,0.724247f,0.740951f,0.757209f,0.773011f,0.788347f,0.803208f,0.817585f,0.831470f,0.844854f,0.857729f,0.870087f,0.881921f,0.893224f,0.903989f,0.914210f,0.923880f,0.932993f,0.941544f,0.949528f,0.956940f,0.963776f,0.970031f,0.975702f,0.980785f,0.985278f,0.989177f,0.992480f,0.995185f,0.997290f,0.998795f,0.999699f};
float sintbl[256]={0.000000f,0.024541f,0.049068f,0.073565f,0.098017f,0.122411f,0.146730f,0.170962f,0.195090f,0.219101f,0.242980f,0.266713f,0.290285f,0.313682f,0.336890f,0.359895f,0.382683f,0.405241f,0.427555f,0.449611f,0.471397f,0.492898f,0.514103f,0.534998f,0.555570f,0.575808f,0.595699f,0.615232f,0.634393f,0.653173f,0.671559f,0.689541f,0.707107f,0.724247f,0.740951f,0.757209f,0.773010f,0.788346f,0.803208f,0.817585f,0.831470f,0.844854f,0.857729f,0.870087f,0.881921f,0.893224f,0.903989f,0.914210f,0.923880f,0.932993f,0.941544f,0.949528f,0.956940f,0.963776f,0.970031f,0.975702f,0.980785f,0.985278f,0.989177f,0.992480f,0.995185f,0.997290f,0.998795f,0.999699f,1.000000f,0.999699f,0.998795f,0.997290f,0.995185f,0.992480f,0.989177f,0.985278f,0.980785f,0.975702f,0.970031f,0.963776f,0.956940f,0.949528f,0.941544f,0.932993f,0.923880f,0.914210f,0.903989f,0.893224f,0.881921f,0.870087f,0.857729f,0.844854f,0.831470f,0.817585f,0.803208f,0.788346f,0.773010f,0.757209f,0.740951f,0.724247f,0.707107f,0.689541f,0.671559f,0.653173f,0.634393f,0.615232f,0.595699f,0.575808f,0.555570f,0.534998f,0.514103f,0.492898f,0.471397f,0.449611f,0.427555f,0.405241f,0.382683f,0.359895f,0.336890f,0.313682f,0.290285f,0.266713f,0.242980f,0.219101f,0.195090f,0.170962f,0.146730f,0.122411f,0.098017f,0.073564f,0.049068f,0.024541f,-0.000000f,-0.024541f,-0.049068f,-0.073565f,-0.098017f,-0.122411f,-0.146731f,-0.170962f,-0.195090f,-0.219101f,-0.242980f,-0.266713f,-0.290285f,-0.313682f,-0.336890f,-0.359895f,-0.382684f,-0.405241f,-0.427555f,-0.449611f,-0.471397f,-0.492898f,-0.514103f,-0.534998f,-0.555570f,-0.575808f,-0.595699f,-0.615232f,-0.634393f,-0.653173f,-0.671559f,-0.689541f,-0.707107f,-0.724247f,-0.740951f,-0.757209f,-0.773011f,-0.788346f,-0.803208f,-0.817585f,-0.831470f,-0.844854f,-0.857729f,-0.870087f,-0.881921f,-0.893224f,-0.903989f,-0.914210f,-0.923880f,-0.932993f,-0.941544f,-0.949528f,-0.956940f,-0.963776f,-0.970031f,-0.975702f,-0.980785f,-0.985278f,-0.989177f,-0.992480f,-0.995185f,-0.997290f,-0.998795f,-0.999699f,-1.000000f,-0.999699f,-0.998795f,-0.997290f,-0.995185f,-0.992480f,-0.989177f,-0.985278f,-0.980785f,-0.975702f,-0.970031f,-0.963776f,-0.956940f,-0.949528f,-0.941544f,-0.932993f,-0.923880f,-0.914210f,-0.903989f,-0.893224f,-0.881921f,-0.870087f,-0.857729f,-0.844853f,-0.831470f,-0.817585f,-0.803207f,-0.788346f,-0.773010f,-0.757209f,-0.740951f,-0.724247f,-0.707107f,-0.689540f,-0.671559f,-0.653173f,-0.634393f,-0.615231f,-0.595699f,-0.575808f,-0.555570f,-0.534997f,-0.514103f,-0.492898f,-0.471397f,-0.449611f,-0.427555f,-0.405241f,-0.382683f,-0.359895f,-0.336890f,-0.313682f,-0.290285f,-0.266713f,-0.242980f,-0.219101f,-0.195090f,-0.170962f,-0.146730f,-0.122411f,-0.098017f,-0.073564f,-0.049068f,-0.024541f};
inline float modf( float X, float Y )
{
    return ( X - int( X / Y ) * Y );
}

void CWO::PrepareTblFresnelApproxInt(int nx, float z0, float pz, int nz)
{
	if(fre_s_tbl!=NULL) delete []fre_s_tbl;
	if(fre_c_tbl!=NULL) delete []fre_c_tbl;
	
	fre_s_tbl=new float[nx*nz];
	fre_c_tbl=new float[nx*nz];

	float lambda=GetWaveLength();
	float p=GetPx();

	for(int j=0;j<nz;j++){

		float z=z0+pz*(float)j;

		float coeff=sqrt(2.0f/(lambda*z))*p;
		float x;	

		//FresnelS
		float boundary=sqrt(2.0f);
		for(int i=0;i<nz;i++){
			float s;
			x=(float)i*coeff;
			if(x>boundary){
				s=0.5f-(1.0f-0.049f*exp(-2.0f*(x-boundary)))/(CWO_PI*x)*
					cos(CWO_PI*x*x/2.0f);
			}
			else{
				s=x*sin(0.5567f*exp(-1.0f*(1.5545f*x-1.9941f)*(1.5545f*x-1.9941f)));

			}

			fre_s_tbl[i+j*nx]=s;
		}

		//FresnelC
		for(int i=0;i<nz;i++){
			float c;
			x=(float)i*coeff;
			if(x>1.0f){
				c=0.5f+(1.0f-0.121f*exp(-2.0f*(x-1.0f)))/(CWO_PI*x)*
					sin(CWO_PI*x*x/2.0f);
			}else{
				c=x*cos(0.6855f*x*x);
			}
			fre_c_tbl[i+j*nx]=c;
		}
	}

}

float CWO::fresnel_s_tbl(float x, float z, float z0, float pz)
{

	float s;
	float sign=1.0f;

	if(x<0.0f){
		sign=-1.0f;
		x=x*-1.0f;
	}
	
	int idx=(float)x;
	s=fre_s_tbl[idx];

	return s*sign;
}

float CWO::fresnel_c_tbl(float x, float z, float z0, float pz)
{
	float c;

	float sign=1.0f;
	if(x<0.0f){
		sign=-1.0f;
		x=x*-1.0f;
	}

	int idx=(int)x;
	c=fre_c_tbl[idx];
	return c*sign;
}

void CWO::FresnelApproxIntTbl(
    float z,
    int x1, int y1, int sx1, int sy1)
{
    SetFieldType(CWO_FLD_COMPLEX);

    cwoComplex *p=(cwoComplex*)GetBuffer();

    int Nx=GetNx();
    int Ny=GetNy();
    int Lx=sx1>>1;
    int Ly=sy1>>1;
    float px=GetPx();
    float py=GetPy();
    float lambda=GetWaveLength();
    float k=2.0f*CWO_PI/lambda;
    float coeff=sqrt(2.0f/(lambda*z));
    float lz=(lambda*z/2.0f);

	#pragma omp parallel for schedule(static) num_threads(8)
    for(int i=0;i<Ny;i++){
        for(int j=0;j<Nx;j++){

            float ax1=coeff*((Nx/2)-Lx-(j-x1))*px;
            float ay1=coeff*((Ny/2)-Ly-(i-y1))*py;
            float ax2=coeff*((Nx/2)+Lx-(j-x1))*px;
            float ay2=coeff*((Ny/2)+Ly-(i-y1))*py;

            cwoComplex c1,c2,c3;
            CWO_RE(c1)=(fresnel_c_tbl(ax2,0,0,0)-fresnel_c_tbl(ax1,0,0,0));
            CWO_IM(c1)=+(fresnel_s_tbl(ax2,0,0,0)-fresnel_s_tbl(ax1,0,0,0));

	        CWO_RE(c2)=(fresnel_c_tbl(ay2,0,0,0)-fresnel_c_tbl(ay1,0,0,0));
            CWO_IM(c2)=+(fresnel_s_tbl(ay2,0,0,0)-fresnel_s_tbl(ay1,0,0,0));

            CWO_RE(c3)=CWO_RE(c1)*CWO_RE(c2)-CWO_IM(c1)*CWO_IM(c2);
            CWO_IM(c3)=CWO_RE(c1)*CWO_IM(c2)+CWO_IM(c1)*CWO_RE(c2);

			//coefficient
			CWO_RE(c2)=cos(k*z-CWO_PI/2.0f)/(lambda*z);
            CWO_IM(c2)=sin(k*z-CWO_PI/2.0f)/(lambda*z);

            CWO_RE(c1)=CWO_RE(c3)*CWO_RE(c2)-CWO_IM(c3)*CWO_IM(c2);
            CWO_IM(c1)=CWO_RE(c3)*CWO_IM(c2)+CWO_IM(c3)*CWO_RE(c2);
	
			int adr=(j)+(i)*Nx;
       
			CWO_RE(p[adr])+=CWO_RE(c1);
			CWO_IM(p[adr])+=CWO_IM(c1);

		}
	}
}



float CWO::fresnel_s(float x)
{
	float boundary=1.4142135623730950488016887242097f;//sqrt(2.0);
	float s;
	float sign=1.0f;

	if(x<0.0f){
		sign=+1.0f;
		x=x*-1.0f;
	}
	else{
		sign=-1.0f;
		
	}

	if(x>boundary){
		s=0.5f-(1.0f-0.049f*exp(-2.0f*(x-boundary)))/(CWO_PI*x)*
			cos(CWO_PI*x*x/2.0f);
	}
	else{
		s=x*sin(0.5567f*exp(-1.0f*(1.5545f*x-1.9941f)*(1.5545f*x-1.9941f)));
	}

	return s*sign;
}

float CWO::fresnel_c(float x)
{
	float c;

	float sign=1.0f;
	if(x<0.0f){
		sign=+1.0f;
		x=x*-1.0f;
	}else{
		sign=-1.0f;
	}

	if(x>1.0f){
		c=0.5f+(1.0f-0.121f*exp(-2.0f*(x-1.0f)))/(CWO_PI*x)*
		sin(CWO_PI*x*x/2.0f);
	}else{
		c=x*cos(0.6855f*x*x);
	}

	return c*sign;
}


void CWO::FresnelInt(
    float z, int x1, int y1, int x2, int y2)
{
    //printf("%s\n",__FUNCTION__);

    //SetFieldType(CWO_FLD_COMPLEX);
	Cplx();
    cwoComplex *p=(cwoComplex*)GetBuffer();

    long long int Nx=GetNx();
    long long int Ny=GetNy();
  //  int Lx=sx1>>1;
   // int Ly=sy1>>1;
    float px=GetPx();
    float py=GetPy();
    float lambda=GetWaveLength();
    float k=2.0f*CWO_PI/lambda;
    float coeff=sqrt(2.0f/(lambda*z));
    float lz=(lambda*z/2.0f);
	
	#pragma omp parallel for schedule(static) num_threads(8)
    for(long long int i=0;i<Ny;i++){
        for(long long int j=0;j<Nx;j++){

          /*  float ax1=coeff*((Nx/2)-Lx-(j-x1))*px;
            float ay1=coeff*((Ny/2)-Ly-(i-y1))*py;
            float ax2=coeff*((Nx/2)+Lx-(j-x1))*px;
            float ay2=coeff*((Ny/2)+Ly-(i-y1))*py;
			*/
			float ax1=coeff*((j-x1))*px;
            float ay1=coeff*((i-y1))*py;
            float ax2=coeff*((j-x2))*px;
            float ay2=coeff*((i-y2))*py;


		/*	float ax1=(Nx/2)-Lx-(j-x1);
            float ay1=(Ny/2)-Ly-(i-y1);
            float ax2=(Nx/2)+Lx-(j-x1);
            float ay2=(Ny/2)+Ly-(i-y1);*/

            cwoComplex c1,c2,c3;
            CWO_RE(c1)=(fresnel_c(ax2)-fresnel_c(ax1));
            CWO_IM(c1)=(fresnel_s(ax2)-fresnel_s(ax1));

	        CWO_RE(c2)=(fresnel_c(ay2)-fresnel_c(ay1));
            CWO_IM(c2)=(fresnel_s(ay2)-fresnel_s(ay1));

            CWO_RE(c3)=CWO_RE(c1)*CWO_RE(c2)-CWO_IM(c1)*CWO_IM(c2);
            CWO_IM(c3)=CWO_RE(c1)*CWO_IM(c2)+CWO_IM(c1)*CWO_RE(c2);

			//coeff
            CWO_RE(c2)=cos(k*z-CWO_PI/2.0f)/(lambda*z);
            CWO_IM(c2)=sin(k*z-CWO_PI/2.0f)/(lambda*z);

            CWO_RE(c1)=CWO_RE(c3)*CWO_RE(c2)-CWO_IM(c3)*CWO_IM(c2);
            CWO_IM(c1)=CWO_RE(c3)*CWO_IM(c2)+CWO_IM(c3)*CWO_RE(c2);

			long long int adr=(j)+(i)*Nx;
       
			CWO_RE(p[adr])+=CWO_RE(c1);
            CWO_IM(p[adr])+=CWO_IM(c1);
        }

			
     }



}


//ë‰å`åˆéÆÇ≈åvéZ
double daikei(
  int flag, //0Å@cos 1 sin
  double k, //îgêî
  double pitch, //ÉTÉìÉvÉäÉìÉOä‘äu
  double d, //ï™äÑêî
  double z,
  cwoVect Q,
  cwoVect P,
  cwoVect N,
  cwoVect T
)
{

	#define VER_COS(a,b,c) (1.0/(a*a+b*b)*cos(c))
	#define VER_SIN(a,b,c) (1.0/(a*a+b*b)*sin(c))

	cwoVect sub=Q-P;
	double start=sub.Dot(T);
	double end=start+2.0*300.0*pitch;

	int a1=(int)(start/pitch);
	int a2=(int)(end/pitch);

//	printf("a1 %d a2 %d \n",a1,a2);

	double h=(end-start)/d;

	double qpn=sub.Dot(N);

	double ph=0;

	double ret=0.0;
	if(flag==0)
	{
	/*	ret=0.5*(VER_COS(start,qpn,0.5*k*start*start/z)+VER_COS(end,qpn,0.5*k*start*start/z));
		for(int i=1;i<d-1;i++)
		{
			double a=start+h*(double)i;
			ret+=VER_COS(a,qpn,0.5*k*a*a/z);
		}

		ret=ret*h;
	*/
	
		for(int i=a1;i<a2;i++){
			float l=(float)i*pitch;
			float coff=l*l+qpn*qpn;
			ret+=cos(0.5*k*l*l/z)/coff;
		}




	/*	double p1=start;
		double p2=end;
		double p1_3=pow(p1,3.0);
		double p2_3=pow(p2,3.0);
		double p1_5=pow(p1,5.0);
		double p2_5=pow(p2,5.0);
		double a=qpn;
		double a_2=a*a;
		double a_4=a_2*a_2;
		double a_6=a_4*a_2;

		ret=(24.0*(p2_5-p1_5)-40.0*a_2*(p2_3-p2_3)+120.0*a_4*(p2-p1))*z*z;
		ret=ret-3.0*a_4*k*k*(p2_5-p1_5);
		ret=ret/(120.0*a_6*z*z);
*/

	}
	else
	{

/*		ret=0.5*(VER_SIN(start,qpn,0.5*k*start*start/z)+VER_SIN(end,qpn,0.5*k*start*start/z));
		for(int i=1;i<d-1;i++)
		{
			double a=start+h*(double)i;
			ret+=VER_SIN(a,qpn,0.5*k*a*a/z);
		}

		ret=ret*h;
*/		for(int i=a1;i<a2;i++){
			float l=(float)i*pitch;
			float coff=l*l+qpn*qpn;
			ret+=sin(0.5*k*l*l/z)/coff;
		}

	}


	//ret=-ret/(2.0*GWO_PI);

	return ret;
}
void CWO::FresnelPolyAperture()
{

	cwoVect n[4],t[4],s[4];

	float z=0.5;
	double k=2.0f*CWO_PI/GetWaveLength();
	cwoComplex *ap=(cwoComplex*)GetBuffer();

	float p=GetPx();
	int NX=GetNx();
	int NY=GetNy();

	n[0].Set(0,-1,0);
	n[1].Set(1,0,0);
	n[2].Set(0,1,0);
	n[3].Set(-1,0,0);
	
	t[0].Set(1,0,0);
	t[1].Set(0,1,0);
	t[2].Set(-1,0,0);
	t[3].Set(0,-1,0);

#define L 300.0

	s[0].Set(-L*p,-L*p,0.0);
	s[1].Set(L*p,-L*p,0.0);
	s[2].Set(L*p,L*p,0.0);
	s[3].Set(-L*p,L*p,0.0);

//	
///*	vec_set(s[0],600*p,600*p,0.0);
//	vec_set(s[1],600*p+2*L*p,600*p,0.0);
//	vec_set(s[2],600*p+2*L*p,600*p+2*L*p,0.0);
//	vec_set(s[3],600*p,600*p+2*L*p,0.0);
//*/
//

//
//	for(int i=0;i<512;i++)
//	{	
//		vec vp;
//		vp.x=(double)(i-NX/2)*p;
//		vp.y=(double)(0-NY/2)*p;
//
//		TRACE("%f\n",daikei(0,k,p,100,z,s[0],vp,n[0],t[0]));
//
//	}
//
//
	for(int i=0;i<4;i++){		
#pragma omp parallel for schedule(static) num_threads(8)
		for(int y=0;y<NY;y++){
			for(int x=0; x<NX; x++){
		
				double Lj=2.0*L*p;
				
				cwoVect vp;
				vp.x=(double)(x-NX/2)*p;
				vp.y=(double)(y-NY/2)*p;
				
				cwoVect pq=s[i]-vp;
				double p1,p2;
				p1=pq.Dot(t[i]);
				p2=Lj+p1;

				double a;
				a=pq.Dot(n[i]);
				//a=a*a;

			//	double c1=1.0/(6.0*pow(a,4.0)*z);
			//	double c2=pow(p1,3.0)-pow(p2,3.0);
			//	double c3=p1-p2;
				
				cwoComplex x1,x2,x3;
				CWO_RE(x1)=daikei(0,k,p,100,z,s[i],vp,n[i],t[i]);//c1*(2.0*c2-6.0*pow(a,2.0)*c3)*z;
				CWO_IM(x1)=daikei(1,k,p,100,z,s[i],vp,n[i],t[i]);//-c1*pow(a,2.0)*k*c2;

			//	CWO_RE(x1)=c1*(2.0*c2-6.0*pow(a,2.0)*c3)*z;
			//	CWO_IM(x1)=-c1*pow(a,2.0)*k*c2;

				//
//				//êœï™ÇÃäO
				CWO_RE(x2)=a*cos(k/(2.0*z)*a*a);
				CWO_IM(x2)=a*sin(k/(2.0*z)*a*a);
		
				CWO_RE(ap[x+y*NX])+=CWO_RE(x1)*CWO_RE(x2)-CWO_IM(x1)*CWO_IM(x2);
				CWO_IM(ap[x+y*NX])+=CWO_RE(x1)*CWO_IM(x2)+CWO_IM(x1)*CWO_RE(x2);
			

				CWO_RE(ap[x+y*NX])=-CWO_RE(ap[x+y*NX])/(2.0*CWO_PI);
			//	CWO_IM(ap[x+y*NX])=-CWO_IM(ap[x+y*NX])/(2.0*CWO_PI);

			//	CWO_RE(ap[x+y*NX])+=CWO_RE(x1)/(NX*NY);
			//	CWO_IM(ap[x+y*NX])+=CWO_IM(x1)/(NX*NY);
////
//				
			}
		//printf("%d %d\n",i,y);

		}
	}



}
/**********************************
Class for look-up table
***********************************/
cwoTbl::cwoTbl()
{
	tbl_sin=NULL;
	tbl_cos=NULL;
	tbl_wrp=NULL;
	__Nz=0;
}
cwoTbl::~cwoTbl()
{
	delete []tbl_sin;
	delete []tbl_cos;
	delete []tbl_wrp;
}

void cwoTbl::SetNz(int Nz)
{
	__Nz=Nz;
}
int cwoTbl::GetNz()
{
	return __Nz;
}

void cwoTbl::MakeSin(int N)
{
	delete tbl_sin;
	tbl_sin=(float*)new float[N];
	
	for(int i=0;i<N;i++)	tbl_sin[i]=sin(2.0*CWO_PI*i/N);
}
void cwoTbl::MakeCos(int N)
{
	delete tbl_cos;
	tbl_sin=(float*)new float[N];
	
	for(int i=0;i<N;i++)	tbl_sin[i]=sin(2.0*CWO_PI*i/N);	
}

void cwoTbl::MakeWRPTbl(
	float z, int Nz, float pz, float wl, float px, float py)
{

	double wn=2.0*CWO_PI/wl;
	delete []tbl_wrp;
	tbl_wrp=(CWO*)new CWO[Nz];
	
	SetNz(Nz);

	for(int i=0;i<Nz;i++){
		double dz=z+pz*i;

		int rdsx=(int)(fabs(dz)*(wl/(2.0*px))/px);
		int rdsy=(int)(fabs(dz)*(wl/(2.0*py))/py);
		int rds=(rdsx>rdsx) ? (rdsx):(rdsy);

		rds=(int)((float)rds/1.4);

		tbl_wrp[i].Create(rds*2,rds*2);
		tbl_wrp[i].Clear();
		
		for(long long int y=0;y<rds*2;y++){		
			for(long long int x=0;x<rds*2;x++){			
				double dx=(x-rds)*px;
				double dy=(y-rds)*py;
				double ph=sqrt(dx*dx+dy*dy+dz*dz);
				cwoComplex tmp;
				tmp.Re(cos(wn*ph));
				tmp.Im(sin(wn*ph));
				tbl_wrp[i].SetPixel(x,y,tmp);
			}
		}
/*
		tbl_wrp[i].Re();
		tbl_wrp[i].Scale(255);
		char fname[256];
		sprintf(fname,"e:\\test%03d.jpg",i);
		tbl_wrp[i].Save(fname);
		*/
	}

}

void cwoTbl::ClipWRPTbl()
{
	int Nz=GetNz();
	
	for(int i=0;i<Nz;i++){
		CWO *p=GetWRPTbl(i);
		int Nx=p->GetNx();
		int Ny=p->GetNy();
		CWO tmp(Nx,Ny/2);
		for(long long int n=0;n<Ny/2;n++){
			for(long long int m=0;m<Nx;m++){
				cwoComplex tp;
				p->GetPixel(m,n,tp);
				tmp.SetPixel(m,n,tp);
			}
		}

		(*p)=tmp;
	}

}

CWO* cwoTbl::GetWRPTbl(int idx)
{
	return &tbl_wrp[idx];
}



