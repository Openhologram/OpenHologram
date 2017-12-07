// Copyright (C) Tomoyoshi Shimobaba 2011-


#include "gwoPLS.h"

gwoPLS::gwoPLS()
{
	p_pnt=NULL;
	SetPointNum(0);
}

gwoPLS::~gwoPLS()
{
	cwoObjPoint *p=GetPointBuffer();
	__Free((void**)&p);
}

cwoObjPoint* gwoPLS::GetPointBuffer()
{
	return p_pnt;
}

void gwoPLS::Send(cwoPLS &a)
{
	int Nx=a.GetNx();
	int Ny=a.GetNy();
	int N=a.GetPointNum();
	
	//for PLS
	if(GetPointBuffer()==NULL || GetPointNum()!=N){
		cwoObjPoint *p=GetPointBuffer();
		__Free((void**)&p);	
		//p_field=__Malloc(Nx*Ny*sizeof(cwoComplex));
		SetPointNum(N);
		p_pnt=(cwoObjPoint*)__Malloc(N*sizeof(cwoObjPoint));
	}

	//for complex amp field or CGH
	if(GetBuffer()==NULL || GetNx()!=Nx || GetNy()!=Ny){
		void *p=GetBuffer();
		__Free(&p);	
		p_field=__Malloc(Nx*Ny*sizeof(cwoComplex));
		__Memset(GetBuffer(),0,Nx*Ny*sizeof(cwoComplex));
	}

	cwoObjPoint *host=a.GetPointBuffer();
	cwoObjPoint *dev=GetPointBuffer();

	ctx=a.ctx;//important!!!!

	gwoSend((gwoCtx*)&ctx, host, dev, N*sizeof(cwoObjPoint));


}


void gwoPLS::__PLS_Fresnel(float ph)
{

	SetFieldType(CWO_FLD_COMPLEX);
	cwoComplex *field=(cwoComplex*)GetBuffer();
	gwoObjPoint *obj=(gwoObjPoint*)GetPointBuffer();
	gwoPLSFresnel((gwoCtx*)&ctx, obj , (gwoComplex*)field, ph);

}

void gwoPLS::__PLS_CGH_Fresnel(float ph)
{

	SetFieldType(CWO_FLD_INTENSITY);
	float *field=(float*)GetBuffer();
	gwoObjPoint *obj=(gwoObjPoint*)GetPointBuffer();
	gwoPLSCGHFresnel((gwoCtx*)&ctx, obj , field, ph);

}
