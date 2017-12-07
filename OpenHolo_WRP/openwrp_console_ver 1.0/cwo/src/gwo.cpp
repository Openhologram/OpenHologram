// Copyright (C) Tomoyoshi Shimobaba 2011-



#include <stdio.h>

#include "gwo.h"
#include "gwo_lib.h"


#include "cwo.h"
#include "cwoCu.h"

//#include <cutil.h>
//#include <cuda.h> 
#include <cuda_runtime.h>
//#include <curand.h>



GWO::GWO()
{	
	p_field=NULL;
	p_rnd = NULL;

//	p_diff_a=NULL;
//p_diff_b=NULL;

	//threads.x1=16;
	//threads.x2=16;
	//SetThreads(16,16);

	SetFieldType(CWO_FLD_COMPLEX);
	SetSize(0,0,1);
	SetWaveLength(633.0e-9);
	SetPitch(10.0e-6,10.0e-6,10.0e-6);
	SetDstPitch(10.0e-6,10.0e-6,10.0e-6);
	SetOffset(0,0);
	SetDstOffset(0,0);
	SetPropDist(0.1);
	SetCalcType(-1);

	//gtx.stream=NULL;
	gtx.SetStream(NULL);
	SetStreamMode(0);
	//Init();
}

GWO::~GWO()
{
	//do not use any virtual functions!
	if(p_field!=NULL){
		gwoDevFree(p_field);
		p_field=NULL;
	}

	if (p_rnd != NULL){
		gwoDevFree(p_rnd);
	}

	SetStreamMode(0);
}

GWO::GWO(GWO &tmp)
{
	//copy constructor	
	//Init();
	InitParams();
	InitBuffers();

	ctx=tmp.ctx;
	
	int size=GetMemSizeCplx();
	p_field=(cwoComplex*)__Malloc(size);

	__Memcpy(GetBuffer(),tmp.GetBuffer(),size);	

	//DestroyStream();
	gtx.SetStream(NULL);
	SetStreamMode(0);

}

GWO::GWO(int Nx, int Ny, int Nz)
{
	InitParams();
	InitBuffers();

	SetSize(Nx, Ny, Nz);
	p_field = __Malloc(Nx*Ny*sizeof(cwoComplex));

	gtx.SetStream(NULL);
	SetStreamMode(0);
}



void GWO::SetDev(int dev)
{
	int cur_dev;
	cudaGetDevice(&cur_dev); //current device
	if(dev!=cur_dev) cudaSetDevice(dev); //Select device
}

void GWO::SetThreads(int Nx, int Ny)
{

	gwoSetThreads(Nx,Ny);
}

int GWO::GetThreadsX()
{
	return gwoGetThreadsX();
}
int GWO::GetThreadsY()
{
	return gwoGetThreadsY();
}
/*
void GWO::Create(int dev, int Nx, int Ny)
{
	printf("%s\n",__FUNCTION__);

	p_field=NULL;
	
	SetSize(Nx,Ny);
		
	if(p_field==NULL || GetNx()!=Nx || GetNy()!=Ny){
		__Free(&p_field);
		p_field=NULL;
		p_field=__Malloc(GetNx()*GetNy()*sizeof(cwoComplex));
	}
	
}*/

void GWO::Delete()
{
	if(p_field!=NULL){
		gwoDevFree(p_field);
		p_field=NULL;
	}
/*	if(p_diff_a!=NULL){
		gwoDevFree(p_diff_a);
		p_diff_a=NULL;
	}
	if(p_diff_b!=NULL){
		gwoDevFree(p_diff_b);
		p_diff_b=NULL;
	}
	*/
}

/*

void GWO::Send(CWO &a)
{
	int size;
	int type=a.GetFieldType();
	int Nx=a.GetNx();
	int Ny=a.GetNy();
	int Nz=a.GetNz();
	

	if(p_field==NULL || GetNx()!=Nx || GetNy()!=Ny || GetNz()!=Nz){
		__Free(&p_field);	
		p_field=__Malloc(Nx*Ny*Nz*sizeof(cwoComplex));
	}

	void *host=a.GetBuffer();
	void *dev=GetBuffer();


	ctx=a.ctx;//important!!!!

	if(type==CWO_FLD_COMPLEX)
		size=Nx*Ny*Nz*sizeof(cwoComplex);
	else
		size=Nx*Ny*Nz*sizeof(float);

	gwoSend( &ctx, host, dev, size);


}
*/


/*
void GWO::Send(CWO &a)
{
	int size;
	int type=a.GetFieldType();
	int Nx=a.GetNx();
	int Ny=a.GetNy();
	int Nz=a.GetNz();
	

	if(p_field==NULL || GetNx()!=Nx || GetNy()!=Ny || GetNz()!=Nz){
		__Free(&p_field);	
		p_field=__Malloc(Nx*Ny*Nz*sizeof(cwoComplex));
	}

	void *host=a.GetBuffer();
	void *dev=GetBuffer();


	ctx=a.ctx;//important!!!!

	if(type==CWO_FLD_COMPLEX)
		size=Nx*Ny*Nz*sizeof(cwoComplex);
	else
		size=Nx*Ny*Nz*sizeof(float);

	gwoSend( &ctx, host, dev, size);


}*/


void GWO::Send(CWO &a)
{
	int size;
	int IsSizeChange=((GetNx()!=a.GetNx()) | (GetNy()!=a.GetNy()) | (GetNz()!=a.GetNz()));
		
	if(p_field==NULL || IsSizeChange){
		__Free(&p_field);	
		p_field=__Malloc(a.GetMemSizeCplx());
	}
	
//	int gpu_nthreads_x=ctx.nthread_x;
//	int gpu_nthreads_y=ctx.nthread_y;
	ctx=a.ctx;//important!!!!
//	ctx.nthread_x=gpu_nthreads_x;
//	ctx.nthread_y=gpu_nthreads_y;

	size=a.GetMemSizeCplx();
	
	void *src=a.GetBuffer();
	void *dst=GetBuffer();
	//gwoSend(&ctx, src, dst, size, &cstream);
	//printf("send %x\n",cstream);
	gwoSend(&ctx, &gtx, src, dst, size);


}


void GWO::Recv(CWO &a)
{
	int size;
	int Nx=GetNx();
	int Ny=GetNy();
	int Nz=GetNz();

	size=GetMemSizeCplx();//Nx*Ny*Nz*sizeof(cwoComplex);

	if(a.GetBuffer()==NULL || a.GetNx()!=Nx || a.GetNy()!=Ny || a.GetNz()!=Nz){
		a.__Free(&a.p_field);
		a.p_field=a.__Malloc(Nx*Ny*Nz*sizeof(cwoComplex));
//		int cpu_nthreads_x=a.ctx.nthread_x;
	//	int cpu_nthreads_y=a.ctx.nthread_y;
		a.ctx=ctx;
//		a.ctx.nthread_x=cpu_nthreads_x;
	//	a.ctx.nthread_y=cpu_nthreads_y;
	}

//	a.SetFieldType(/*GetFieldType()*/type);

	void *host=a.GetBuffer();
	void *dev=GetBuffer();
	
	//gwoRecv(&ctx,dev,host,size,&cstream);
	//printf("recv %x\n",cstream);
	gwoRecv(&ctx, &gtx, dev, host, size);


}


void GWO::CreateStream()
{
	//cudaStreamCreate(&gtx.stream);
	cwoStream stream;
	cudaStreamCreate(&stream);
//	cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
	gtx.SetStream(stream);
}

cwoStream GWO::GetStream()
{
	return gtx.stream;
}
void GWO::DestroyStream()
{
	if(gtx.stream!=NULL) cudaStreamDestroy(gtx.GetStream());
	gtx.stream=NULL;
}

void GWO::SyncStream()
{
	if(gtx.stream!=NULL) cudaStreamSynchronize(gtx.GetStream());
}

void GWO::SetStreamMode(int mode)
{
	stream_mode=mode;
	if(mode==1){
		CreateStream();
	//	gtx.stream=(void*)&cstream;
	}
	else{
		DestroyStream();
		gtx.SetStream(NULL);
	/*	cstream=NULL;
		ctx.cstream=(void*)&cstream;;*/
	}

}

int GWO::GetStreamMode()
{
	return stream_mode;
}


int GWO::Load(char* fname, int c)
{
	CWO *cwo=(CWO*)new CWO;
	cwo->Load(fname,c);
	GWO::Send((*cwo));
	delete cwo;
	return CWO_SUCCESS;
}

int GWO::Load(char* fname_amp, char *fname_pha, int c)
{
	CWO *cwo=(CWO*)new CWO;
	cwo->Load(fname_amp,fname_pha,c);
	GWO::Send((*cwo));
	delete cwo;
	return CWO_SUCCESS;
}


int GWO::Save(char *fname, int bmp_8_24)
{
	//return 0: success
	//return -1: error (field type is not real number etc...)

	CWO *c=(CWO*)new CWO;
	(*c).Create(GetNx(),GetNy());
	GWO::Recv(*c);

	if(CheckExt(fname,"cwo")){
		//save as cwo file

		FILE *fp=fopen(fname,"wb");
		if(fp==NULL) return -1;
		
		fwrite(&ctx,sizeof(c->GetCtx()),1,fp);
		cwoComplex *p=c->GetBuffer();
		fwrite(p, sizeof(cwoComplex), c->GetNx()*c->GetNy()*c->GetNz(),fp); 

		fclose(fp);
	
	}
	else{
		
		if(CheckExt(fname,"bmp") && bmp_8_24==8){
			//save as Bitmap file etc..
			int Nx=c->GetNx();
			int Ny=c->GetNy();
			unsigned char *save_img=(unsigned char *)new char[Nx*Ny];
			for(int i=0;i<Nx*Ny;i++){
				cwoComplex *p=c->GetBuffer();
				save_img[i]=(unsigned char)CWO_RE(p[i]);
			}
			cwoSaveBmp(fname,save_img,Nx,Ny,8,-1);
			delete []save_img;
		}
		else{
			cwoSaveCimgMono(&ctx,fname,c->GetBuffer());
		}

	}
	
	delete c;

	return 0;

	
}


int GWO::Save(char* fname, CWO *r, CWO *g, CWO *b)
{
	
	if(r==0 && g==0 && b==0){
		CWO *cwo=(CWO*)new CWO;
		GWO::Recv(*cwo);
		cwo->Save(fname);
		delete cwo;
	}
	else{
	/*	CWO *cr = NULL, *cg = NULL, *cb = NULL;
		CWO tmp;
		int Nx=0, Ny=0;
		
		if(r!=0){
			cr=new CWO;
			r->Recv(*cr);
			Nx = cr->GetNx();
			Ny = cr->GetNy();
		}
		if(g!=0){
			cg=new CWO;
			g->Recv(*cg);
			Nx = cg->GetNx();
			Ny = cg->GetNy();
		}
		if(b!=0){
			cb=new CWO;
			b->Recv(*cg);
			Nx = cb->GetNx();
			Ny = cb->GetNy();
		}
		printf("%s %d %d\n",__FUNCTION__, Nx, Ny);
		tmp.SetSize(Nx, Ny);
		tmp.Save(fname, cr, cg, cb);

		delete cr;
		delete cg;
		delete cb;*/
	
	}

	return CWO_SUCCESS;
}

int GWO::SaveAsImage(char* fname, float i1, float i2, float o1, float o2, int flag)
{

/*	GWO *tmp=NULL;
	tmp=(GWO*)new GWO;
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
	*/
	return CWO_SUCCESS;

}
int GWO::SaveAsImage(char* fname, int flag, CWO *rL, CWO *g, CWO *b)
{
	if (GetBuffer() == NULL) return CWO_ERROR;


	GWO *tmp=NULL;
	tmp=(GWO*)new GWO;
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



int GWO::SaveAsImage(cwoComplex *p, int Nx, int Ny, char* fname, int flag, int bmp_8_24)
{

	GWO *tmp=NULL;
	tmp=(GWO*)new GWO;
	if(tmp==NULL) return CWO_ERROR;

	(*tmp).Create(Nx,Ny);
	(*tmp).__Copy(p,0,0,Nx,Ny,(*tmp).GetBuffer(),0,0,Nx,Ny,Nx,Ny);
	
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
	tmp->Save(fname, bmp_8_24);

	delete tmp;

	return CWO_SUCCESS;
}




void* GWO::__Malloc(size_t size)
{
	void *p=NULL;
	cudaMalloc(&p,size);
	return p;
}


void GWO::__Free(void **a)
{	
	if(*a!=NULL) cudaFree(*a);
	*a = NULL;
}

void GWO::__Memcpy(void *dst, void *src, size_t size)
{

	cudaMemcpy(dst,src,size,cudaMemcpyDeviceToDevice);
}

void GWO::__Memset(void *p, int c, size_t size)
{
	cudaMemset(p,c,size);
}


void GWO::Fill(cwoComplex pix)
{
	gwoFill(&ctx, &gtx,GetBuffer(),pix);
}

void GWO::__Expand(
	void *src, int sx, int sy, int srcNx, int srcNy, 
	void *dst, int dx, int dy, int dstNx, int dstNy,
	int type)
{

	//Nz
	gwoExpand(&ctx, &gtx,src,srcNx,srcNy,dst,dstNx,dstNy,type);
	SetSize(dstNx,dstNy);
}


void GWO::__ShiftedFresnelAperture(cwoComplex *a)
{
	gwoShiftedFresnelAperture(&ctx, &gtx, a);//8ms
}
void GWO::__ShiftedFresnelProp(cwoComplex *a)
{
	gwoShiftedFresnelProp(&ctx, &gtx, a);//1ms
}
void GWO::__ShiftedFresnelCoeff(cwoComplex *a)
{
	gwoShiftedFresnelCoeff(&ctx, &gtx, a);
}

void GWO::__ARSSFresnelAperture(cwoComplex *a)
{
	gwoARSSFresnelAperture(&ctx, &gtx,a);
}
void GWO::__ARSSFresnelProp(cwoComplex *a)
{
	gwoARSSFresnelProp(&ctx, &gtx, a);
}
void GWO::__ARSSFresnelCoeff(cwoComplex *a)
{
	gwoARSSFresnelCoeff(&ctx, &gtx,a);
}



void GWO::__FresnelConvProp(cwoComplex *a)
{
	gwoFresnelConvProp(&ctx, &gtx, a);
}

void GWO::__FresnelConvCoeff(cwoComplex *a, float const_val)
{
	gwoFresnelConvCoeff(&ctx, &gtx, a);
}

void GWO::__AngularProp(cwoComplex *a, int flag)
{
	//gwoAngularProp(&ctx, a);


	switch(flag){
		case CWO_PROP_CENTER:
			gwoAngularProp(&ctx, &gtx,a);		
			break;
		case CWO_PROP_FFT_SHIFT:
			gwoAngularPropFS(&ctx, &gtx,a);		
			break;
		case CWO_PROP_MUL_CENTER:
			gwoAngularPropMul(&ctx, &gtx,a);	
			break;
		case CWO_PROP_MUL_FFT_SHIFT:
			gwoAngularPropMulFS(&ctx, &gtx,a);	
			break;
		default:
			gwoAngularProp(&ctx, &gtx,a);		
	}
}
/*
void GWO::__ShiftedAngularProp(cwoComplex *a)
{
	gwoShiftedAngularProp( &ctx, a);
}*/
void GWO::__HuygensProp(cwoComplex *a)
{
	gwoHuygensProp( &ctx, &gtx, (cwoComplex*)a);

}
void GWO::__FresnelFourierProp(cwoComplex *a)
{
	gwoFresnelFourierProp( &ctx, &gtx, (cwoComplex*)a);
}
void GWO::__FresnelFourierCoeff(cwoComplex *a)
{
	gwoFresnelFourierCoeff( &ctx, &gtx, (cwoComplex*)a);
}


void GWO::__FresnelDblAperture(cwoComplex *a, float z1)
{
	gwoFresnelDblAperture( &ctx, &gtx,  (cwoComplex*)a, z1);
}
void GWO::__FresnelDblFourierDomain(cwoComplex *a, float z1, float z2, cwoInt4 *zp)
{
	gwoFresnelDblFourierDomain( &ctx, &gtx, (cwoComplex*)a, z1, z2, zp);
}
void GWO::__FresnelDblCoeff(cwoComplex *a, float z1, float z2)
{
	gwoFresnelDblCoeff( &ctx, &gtx,(cwoComplex*)a, z1, z2);

	//printf("%s\n", __FUNCTION__);

}



void GWO::__FFT(void *src, void *dst, int type)
{
	gwoFFT( &ctx, &gtx, src, dst, type);
}
void GWO::__IFFT(void *src, void *dst)
{
	gwoIFFT( &ctx, &gtx, src, dst);
}
void GWO::__FFTShift(void *src)
{
	gwoFFTShift( &ctx, &gtx, src,src);
}


void GWO::__NUFFT_T1(cwoComplex *p_fld, cwoFloat2 *p_x, int R, int Msp)
{

	gwoNUFFT_T1(&ctx, &gtx, p_fld, p_x, R, Msp);
}


void GWO::__NUFFT_T2(cwoComplex *p_fld, cwoFloat2 *p_x, int R, int Msp)
{
	gwoNUFFT_T2(&ctx, &gtx, p_fld, p_x, R, Msp);
}




void GWO::__Add(cwoComplex *a, cwoComplex b, cwoComplex *c)
{
	cwoComplex gb;
	CWO_RE(gb)=CWO_RE(b);
	CWO_IM(gb)=CWO_IM(b);
	gwoAddCplx( &ctx, &gtx, (cwoComplex *)a, gb, (cwoComplex *)c);
}
void GWO::__Add(cwoComplex *a, cwoComplex *b, cwoComplex *c)
{
	gwoAddCplxArry( &ctx, &gtx, (cwoComplex *)a, (cwoComplex *)b, (cwoComplex *)c);
}
void GWO::__Sub(cwoComplex *a, cwoComplex b, cwoComplex *c)
{
	cwoComplex gb;
	CWO_RE(gb)=CWO_RE(b);
	CWO_IM(gb)=CWO_IM(b);
	gwoSubCplx( &ctx, &gtx, (cwoComplex *)a, gb, (cwoComplex *)c);
}
void GWO::__Sub(cwoComplex *a, cwoComplex *b, cwoComplex *c)
{
	gwoSubCplxArry( &ctx, &gtx, (cwoComplex *)a, (cwoComplex *)b, (cwoComplex *)c);
}
void GWO::__Mul(cwoComplex *a, cwoComplex b, cwoComplex *c)
{
	cwoComplex gb;
	CWO_RE(gb)=CWO_RE(b);
	CWO_IM(gb)=CWO_IM(b);
	gwoMulCplx( &ctx, &gtx, (cwoComplex *)a, gb, (cwoComplex *)c);
}
void GWO::__Mul(cwoComplex *a, cwoComplex *b, cwoComplex *c)
{
	gwoMulCplxArry( &ctx, &gtx, (cwoComplex *)a, (cwoComplex *)b, (cwoComplex *)c);
}
void GWO::__Div(cwoComplex *a, cwoComplex b, cwoComplex *c)
{
	cwoComplex gb;
	CWO_RE(gb)=CWO_RE(b);
	CWO_IM(gb)=CWO_IM(b);
	gwoDivCplx( &ctx, &gtx, (cwoComplex *)a, gb, (cwoComplex *)c);
}
void GWO:: __Div(cwoComplex *a, cwoComplex *b, cwoComplex *c)
{
	gwoDivCplxArry( &ctx, &gtx, (cwoComplex *)a, (cwoComplex *)b, (cwoComplex *)c);
}


void GWO::SqrtReal()
{
	cwoComplex *p=(cwoComplex *)GetBuffer();
	gwoSqrtReal(&ctx, &gtx, p);
}
void GWO::SqrtCplx()
{
	cwoComplex *p=(cwoComplex *)GetBuffer();
	gwoSqrtCplx(&ctx, &gtx, p);
}


void GWO::__AddSphericalWave(cwoComplex *p, float x, float y, float z, float px, float py, float a)
{
	gwoAddSphericalWave( &ctx, &gtx, (cwoComplex*)p, x, y, z, px, py, a);
}

void GWO::__MulSphericalWave(cwoComplex *p, float x, float y, float z, float px, float py, float a)
{
	gwoMulSphericalWave( &ctx, &gtx, (cwoComplex*)p, x, y, z, px, py, a);
}
void GWO::__AddApproxSphWave(cwoComplex *p, float x, float y, float z, float zx, float zy, float px, float py, float a)
{
	gwoAddApproxSphWave(&ctx, &gtx,p,x,y,z,zx,zy,px,py,a);
}
void GWO::__MulApproxSphWave(cwoComplex *p, float x, float y, float z, float zx, float zy, float px, float py, float a)
{
	gwoMulApproxSphWave(&ctx, &gtx,p,x,y,z,zx, zy,px,py,a);
}


void GWO::__Real2Complex(float *src, cwoComplex *dst)
{
	gwoReal2Complex( &ctx, &gtx,src,(cwoComplex*)dst);
}

void GWO::__Phase2Complex(float *src, cwoComplex *dst)
{
	gwoPhase2Complex( &ctx, &gtx,src,(cwoComplex*)dst);
	
}

void GWO::__Arg2Cplx(cwoComplex *src, cwoComplex *dst, float scale, float offset)
{
	gwoArg2Cplx(&ctx, &gtx,src,dst,scale,offset);
}


void GWO::__Re(cwoComplex*a , cwoComplex *b)
{
	gwoRe(&ctx, &gtx,a,b);
}
void GWO::__Im(cwoComplex*a , cwoComplex *b)
{
	gwoIm(&ctx, &gtx,a,b);
}

void GWO::__Intensity(cwoComplex *a, cwoComplex *b)
{
	gwoIntensity( &ctx, &gtx, a, b);
}
void GWO::__Amp(cwoComplex *a, cwoComplex *b)
{
	gwoAmp(&ctx, &gtx,a,b);
}

void GWO::__Phase(cwoComplex *a, cwoComplex *b, float offset)
{
	gwoPhase(&ctx, &gtx,a,b,offset);
}
void GWO::__Arg(cwoComplex *a, cwoComplex *b, float scale, float offset)
{
	gwoArg(&ctx, &gtx,a, b, scale, offset);
}
void GWO::__Polar(float *amp, float *ph, cwoComplex *c)
{
	gwoPolar( &ctx, &gtx, amp, ph, (cwoComplex*)c);
}
void GWO::__ReIm(cwoComplex *re, cwoComplex *im, cwoComplex *c)
{
	gwoReIm( &ctx, &gtx, re, im, (cwoComplex*)c);
}

void GWO::SetRandSeed(long long s){
	rnd_seed = s;
	//gwoRandSeed(&ctx, &gtx, s);
}

void GWO::RandReal(float max, float min)
{
	
	//if (IsSizeChanged()){
		
	p_rnd = (float*)__Malloc(GetNx()*GetNy()*sizeof(float));
	//	prev_ctx = ctx;
	//}
	gwoSetRandReal(&ctx, &gtx, p_rnd, GetBuffer(),GetRandSeed(), max, min);
	
	SetRandSeed(GetRandSeed()+1);
	__Free((void**)&p_rnd);

}

void GWO::__RandPhase(cwoComplex *a, float max, float min)
{
	
	p_rnd = (float*)__Malloc(GetNx()*GetNy()*sizeof(float));
	//	prev_ctx = ctx;
	//}
	gwoSetRandPhase(&ctx, &gtx, p_rnd, GetBuffer(), GetRandSeed(), max, min);
	SetRandSeed(GetRandSeed()+1);
	__Free((void**)&p_rnd);
}

void GWO::__MulRandPhase(cwoComplex *a, float max, float min)
{
	//if (IsSizeChanged()){
	p_rnd = (float*)__Malloc(GetNx()*GetNy()*sizeof(float));
	//	prev_ctx = ctx;
	//}
	gwoMulRandPhase(&ctx, &gtx, p_rnd, GetBuffer(), GetRandSeed(), max, min);
	//__Free((void**)&p_rnd);
	SetRandSeed(GetRandSeed()+1);
	__Free((void**)&p_rnd);
	
}




void GWO::FFTShift()
{
	gwoFFTShift( &ctx, &gtx,GetBuffer(),GetBuffer());
}



void GWO::Gamma(float gamma)
{
	gwoGamma( &ctx, &gtx,(cwoComplex*)GetBuffer(), gamma);
}

void GWO::Threshold(float max, float min)
{
	gwoThreshold( &ctx, &gtx,/*(float*)*/GetBuffer(), max, min);
}

void GWO::__PickupFloat(float *src, float *pix_p, float pix)
{

	gwoPickupFloat( &ctx, &gtx, src, pix_p, pix);

}
void GWO::__PickupCplx(cwoComplex *src, cwoComplex *pix_p, float pix)
{

	gwoPickupCplx( &ctx, &gtx, src, pix_p, pix);

}

float GWO::Average()
{
	return gwoAverage( &ctx, &gtx, GetBuffer());

}
float GWO::Variance()
{
	float ave=gwoAverage( &ctx, &gtx, GetBuffer());
	return gwoVariance( &ctx, &gtx, GetBuffer(), ave);
}

void GWO::__MaxMin(cwoComplex *a, float *max, float *min, int *max_x, int *max_y,int *min_x, int *min_y)
{

	gwoMaxMin(&ctx, &gtx,a,max,min);
}

/*
int GWO::MaxMin(
	float *a,
	float *max, float *min, 
	int *max_x, int *max_y,
	int *min_x, int *min_y)
{
	//-1 p_field includes not real value
	//0 success

	gwoMaxMin( &ctx,a,max,min);

	return 0;

}


int GWO::MaxMin(
	float *max, float *min, 
	int *max_x, int *max_y,
	int *min_x, int *min_y)
{
	
	return MaxMin((float*)GetBuffer(),max,min,max_x,max_y,min_x,min_y);
}
*/

cwoComplex GWO::TotalSum()
{
	return gwoTotalSum(&ctx, &gtx, GetBuffer());
}

/*
void GWO::__Quant(float lim, float max, float min)
{
	//quantization 
	cwoComplex *a=(cwoComplex *)GetBuffer();
	gwoQuant( &ctx, &gtx, a, lim, max, min);
}*/

int GWO::__ScaleReal(float i1, float i2, float o1, float o2)
{
	cwoComplex *p=(cwoComplex*)GetBuffer();
	gwoScaleReal(&ctx, &gtx,p,p,i1,i2,o1,o2);

	return CWO_SUCCESS;
}
int GWO::__ScaleCplx(float i1, float i2, float o1, float o2)
{
	cwoComplex *p=(cwoComplex*)GetBuffer();
	gwoScaleCplx(&ctx, &gtx,p,p,i1,i2,o1,o2);

	return CWO_SUCCESS;
}

void GWO::__RectFillInside(cwoComplex *p, int x, int y, int Sx, int Sy, cwoComplex a)
{
	gwoRectFillInside(&ctx, &gtx, p, x, y, Sx, Sy, a);
}

void GWO::__RectFillOutside(cwoComplex *p, int x, int y, int Sx, int Sy, cwoComplex a)
{

	gwoRectFillOutside(&ctx, &gtx, p, x, y, Sx, Sy, a);
}

void GWO::__FloatToChar(char *dst, float *src, int N)
{
	gwoFloatToChar(&ctx, &gtx, dst, src);

}
void GWO::__CharToFloat(float *dst, char *src, int N)
{
	gwoCharToFloat(&ctx, &gtx, dst, src);

}

void GWO::__Copy(
			cwoComplex *src, int x1, int y1, int sNx, int sNy,
			cwoComplex *dst, int x2, int y2, int dNx, int dNy, 
			int Sx, int Sy)
{

	gwoCopy(&ctx, &gtx, 
		src, x1, y1, sNx, sNy,
		dst, x2, y2, dNx, dNy, 
		Sx, Sy);
}

void GWO::__ResizeNearest(
	cwoComplex *p_new, int dNx, int dNy, cwoComplex *p_old, int sNx, int sNy)
{
	gwoResizeNearest(&ctx, &gtx, p_new, dNx, dNy, p_old, sNx,sNy);
}

void GWO::__ResizeLinear(
	cwoComplex *p_new, int dNx, int dNy, cwoComplex *p_old, int sNx, int sNy)
{
	gwoResizeLinear(&ctx, &gtx, p_new, dNx, dNy, p_old, sNx,sNy);
}

void GWO::__ResizeCubic(
	cwoComplex *p_new, int dNx, int dNy, cwoComplex *p_old, int sNx, int sNy)
{
	gwoResizeCubic(&ctx, &gtx, p_new, dNx, dNy, p_old, sNx,sNy);
}

void GWO::__ResizeLanczos(
	cwoComplex *p_new, int dNx, int dNy, cwoComplex *p_old, int sNx, int sNy)
{
	gwoResizeLanczos(&ctx, &gtx, p_new, dNx, dNy, p_old, sNx,sNy);
}


void GWO::ErrorDiffusion(CWO *output, int flag)
{
	cwoComplex *p_i = GetBuffer();
	cwoComplex *p_o = output->GetBuffer();
	gwoErrorDiffusion(&ctx, &gtx, p_i,p_o);
	
}

void GWO::ErrorDiffusionSegmented(CWO *output, int flag)
{

	cwoComplex *p_i = GetBuffer();
	cwoComplex *p_o = output->GetBuffer();
	gwoErrorDiffusion(&ctx, &gtx, p_i, p_o);

}


float GWO::MSE(CWO &ref)
{
	if(GetNx()!=ref.GetNx() || GetNy()!=ref.GetNy()) return -1.0f;
	cwoComplex *p_tar=GetBuffer();
	cwoComplex *p_ref=ref.GetBuffer();
	float mse=0.0f;

	return gwoMSE(&ctx, &gtx,p_tar,p_ref);
}



////////////////////////
//Test code
////////////////////////

void GWO::test()
{
	printf("%s\n",__FUNCTION__);
}

void GWO::__ArbitFresnelDirect(
	cwoComplex *p1, cwoComplex *p2, 
	cwoFloat2 *p_x1, cwoFloat2 *p_x2, 
	float *p_d1, float *p_d2)
{
	gwoArbitFresnelDirect(&ctx, &gtx,p1, p2, p_x1, p_x2, p_d1, p_d2);

}

void GWO::__ArbitFresnelCoeff(
	cwoComplex *p, cwoFloat2 *p_x2, float *p_d2)
{

	gwoArbitFresnelCoeff(&ctx, &gtx, p, p_x2, p_d2);

}

