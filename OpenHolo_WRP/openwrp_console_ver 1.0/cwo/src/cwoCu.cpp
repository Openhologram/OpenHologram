// Copyright (C) Tomoyoshi Shimobaba 2011-


#include <stdlib.h>
#include <time.h>
#include <math.h>

#include "cwocu.h"
#include <cuda_runtime.h>

cwoCu::cwoCu()
{
//	printf("%s\n",__FUNCTION__);
//	Init();

}
cwoCu::~cwoCu()
{
//	printf("%s\n",__FUNCTION__);
	if(p_field!=NULL){
		cudaFreeHost(p_field);
		//gwoDevFree(p_field);
		p_field=NULL;
	}
/*	if(p_diff_a!=NULL){
		cudaFreeHost(p_diff_a);
	//	gwoDevFree(p_diff_a);
		p_diff_a=NULL;
	}
	if(p_diff_b!=NULL){
		cudaFreeHost(p_diff_b);
		//gwoDevFree(p_diff_b);
		p_diff_b=NULL;
	}*/


}
/*
void cwoCu::SetDev(int dev)
{
	int cur_dev;
	cudaGetDevice(&cur_dev); //current device
	if(dev!=cur_dev) cudaSetDevice(dev); //Select device

}

void cwoCu::Create(int Nx, int Ny)
{
	
	//if(p_field==NULL || GetNx()!=Nx || GetNy()!=Ny){
		//SetFieldType(CWO_FLD_COMPLEX);
		__Free(p_field);
		SetSize(Nx,Ny);
		p_field=__Malloc(GetNx()*GetNy()*sizeof(cwoComplex));

		printf("%s %X \n",__FUNCTION__,(int)p_field);
	//}
}

void cwoCu::Delete(int dev)
{
	SetDev(dev);
	__Free(p_field);
}
*/




void* cwoCu::__Malloc(size_t size)
{
	//printf("%s\n",__FUNCTION__);
	void *p=NULL;
	cudaHostAlloc((void**)&p,size,cudaHostAllocDefault); 
	return p;
}

void cwoCu::__Free(void **a)
{
//	printf("%s\n",__FUNCTION__);

	if(*a!=NULL){
		cudaFreeHost(*a);
		*a=NULL;
	}

}
