// Copyright (C) Tomoyoshi Shimobaba 2011-

#ifdef _OPENMP
#include<omp.h>
#endif

#include"cwoPLS.h"


cwoPLS::cwoPLS()
{
	p_pnt=NULL;

	wrp_tbl.N=0;
	wrp_tbl.radius=NULL;
	wrp_tbl.tbl=NULL;
	
	//nib=NULL;
	wbi=NULL;

	SetNz(0);
	p_tbl = NULL;
	

	SetPointNum(0);
}

cwoPLS::~cwoPLS()
{
	cwoObjPoint*p=GetPointBuffer();
	//__Free((void**)&wrp_tbl);
	__Free((void**)&p);
	//__Free((void**)&nib);
	__Free((void**)&wbi);

	
	if(wrp_tbl.radius!=NULL) delete wrp_tbl.radius;
	for(int i=0;i<wrp_tbl.N;i++)
		if(wrp_tbl.tbl[i]!=NULL)  delete wrp_tbl.tbl[i];

	if(wrp_tbl.tbl!=NULL) delete wrp_tbl.tbl;

	if (p_tbl != NULL) delete[]p_tbl;

}


int cwoPLS::Load(char *fname)
{
	FILE *fp=fopen(fname,"rb");
	if(fp==NULL) return CWO_ERROR;

	int num;
	fread(&num,sizeof(int),1,fp);
	
	SetPointNum(num);

	cwoObjPoint* tmp_p=GetPointBuffer();
	__Free((void**)&tmp_p);
	p_pnt=(cwoObjPoint*)__Malloc(GetPointNum()*sizeof(cwoObjPoint));

	if(CheckExt(fname,"3d")){
		for(int i=0;i<num;i++){
			int x,y,z;
			fread(&x,sizeof(int),1,fp);
			fread(&y,sizeof(int),1,fp);
			fread(&z,sizeof(int),1,fp);
		
			p_pnt[i].x=x;
			p_pnt[i].y=y;
			p_pnt[i].z=z;
			p_pnt[i].a=1.0;

		}
	}
	else if(CheckExt(fname,"3df")){
		for(int i=0;i<num;i++){
			float x,y,z;
			fread(&x,sizeof(float),1,fp);
			fread(&y,sizeof(float),1,fp);
			fread(&z,sizeof(float),1,fp);
		
			p_pnt[i].x=x;
			p_pnt[i].y=y;
			p_pnt[i].z=z;
			p_pnt[i].a=1.0;

		}
	}


	fclose(fp);
	return num;

}



cwoObjPoint* cwoPLS::GetPointBuffer()
{
	return p_pnt;
}

void cwoPLS::SetPointNum(int num)
{
	ctx.PLS_num=num;

}

int cwoPLS::GetPointNum()
{
	return ctx.PLS_num;
}

void cwoPLS::ScalePoint(float lim)
{
	cwoObjPoint *p=GetPointBuffer();
	float mx=0,my=0,mz=0;
	int N=GetPointNum();

/*	for(int i=0;i<N;i++){
		float x=p[i].x;
		float y=p[i].y;
		float z=p[i].z;

		if(fabs(mx)<fabs((float)x)) mx=fabs(x);
		if(fabs(my)<fabs((float)y)) my=fabs(y);
		if(fabs(mz)<fabs((float)z)) mz=fabs(z);
	}

	float max;
	max=(mx>my)?(mx):(my);
	max=(max>mz)?(max):(mz);
*/
	cwoFloat3 xyz=MaxAbsXYZ();

	float max;
	max = (xyz.x>xyz.y) ? (xyz.x) : (xyz.y);
	max = (max>xyz.z) ? (max) : (xyz.z);

	for(int i=0;i<N;i++){
		p[i].x=p[i].x/max *lim;
		p[i].y=p[i].y/max *lim;
		p[i].z=p[i].z/max *lim;
	}
}

void cwoPLS::ScalePoint(float lim_x, float lim_y, float lim_z)
{
	cwoObjPoint *p=GetPointBuffer();
	//float mx=0,my=0,mz=0;
	int N=GetPointNum();

/*	for(int i=0;i<N;i++){
		float x=p[i].x;
		float y=p[i].y;
		float z=p[i].z;

		if(fabs(mx)<fabs((float)x)) mx=fabs(x);
		if(fabs(my)<fabs((float)y)) my=fabs(y);
		if(fabs(mz)<fabs((float)z)) mz=fabs(z);
	}
*/
	cwoFloat3 min_xyz = MinXYZ();
	cwoFloat3 max_xyz = MaxXYZ();

	float dx = fabs(max_xyz.x - min_xyz.x);
	float dy = fabs(max_xyz.y - min_xyz.y);
	float dz = fabs(max_xyz.z - min_xyz.z);

	for(int i=0;i<N;i++){
		p[i].x=p[i].x *lim_x / dx;
		p[i].y = p[i].y *lim_y / dy;
		p[i].z = p[i].z *lim_z / dz;
	}
}

void cwoPLS::ShiftPoint(float dx, float dy, float dz)
{
	cwoObjPoint *p=GetPointBuffer();
	int N=GetPointNum();

	for(int i=0;i<N;i++){
		p[i].x=p[i].x+dx;
		p[i].y=p[i].y+dy;
		p[i].z=p[i].z+dz;
	}
}


cwoFloat3 cwoPLS::MaxXYZ()
{
	float mx = 0, my = 0, mz = 0;
	int N = GetPointNum();
	cwoObjPoint *p = GetPointBuffer();

	for (int i = 0; i<N; i++){
		float x = p[i].x;
		float y = p[i].y;
		float z = p[i].z;

		if (mx<x) mx = x;
		if (my<y) my = y;
		if (mz<z) mz = z;
	}

	cwoFloat3 ret;
	ret.x = mx; ret.y = my; ret.z = mz;
	return ret;
}
cwoFloat3 cwoPLS::MinXYZ()
{
	float mx = 0, my = 0, mz = 0;
	int N = GetPointNum();
	cwoObjPoint *p = GetPointBuffer();

	for (int i = 0; i<N; i++){
		float x = p[i].x;
		float y = p[i].y;
		float z = p[i].z;

		if (mx>x) mx = x;
		if (my>y) my = y;
		if (mz>z) mz = z;
	}

	cwoFloat3 ret;
	ret.x = mx; ret.y = my; ret.z = mz;
	return ret;
}


cwoFloat3 cwoPLS::MaxAbsXYZ()
{
	float mx = 0, my = 0, mz = 0;
	int N = GetPointNum();
	cwoObjPoint *p = GetPointBuffer();

	for (int i = 0; i<N; i++){
		float x = p[i].x;
		float y = p[i].y;
		float z = p[i].z;

		if (fabs(mx)<fabs((float)x)) mx = fabs(x);
		if (fabs(my)<fabs((float)y)) my = fabs(y);
		if (fabs(mz)<fabs((float)z)) mz = fabs(z);
	}

	cwoFloat3 ret;
	ret.x = mx; ret.y = my; ret.z = mz;
	return ret;
}


cwoFloat3 cwoPLS::Centroid()
{
	cwoFloat3 cent;
	cent.x=0.0f;
	cent.y=0.0f;
	cent.z=0.0f;
	int Np=GetPointNum();
	cwoObjPoint *p=GetPointBuffer();

	for(int i=0;i<Np;i++){
		cent.x+=p[i].x;
		cent.y+=p[i].y;
		cent.z+=p[i].z;
	}
	cent.x/=Np;
	cent.y/=Np;
	cent.z/=Np;

	return cent;
}

void cwoPLS::SetNz(int Nz){
	NzTbl = Nz;
}
int cwoPLS::GetNz(){
	return NzTbl;
}

void cwoPLS::PreTbl(float d1, float d2, int Nz)
{
	float p = GetPx();
	float wl = GetWaveLength();
	float dz = (d2 - d1) / Nz;
	
	///printf("%d %d\n", Nx, Ny);
	SetNz(Nz);
	p_tbl = (CWO*)new CWO[Nz];
		
	float z = d1;
	for (int i = 0; i < Nz; i++){
		int Nx = GetNx();
		int Ny = GetNy();
		int free_area = GetAliasingFreeApproxSphWave(z, p, wl);
		Nx = (free_area > Nx) ? (Nx) : (free_area);
		Ny = (free_area > Ny) ? (Ny) : (free_area);
		p_tbl[i].Create(Nx,Ny);
		p_tbl[i].AddApproxSphWave(0.0f, 0.0f, z);
		CWO mask = p_tbl[i];
		mask.Clear();
		mask.Circ(Nx/2, Ny/2, free_area/2, cwoCplxCart(1, 0));
		p_tbl[i] *= mask;
		
		z += dz;
	}
}
CWO* cwoPLS::GetPtrTbl(int idx)
{
	return p_tbl+idx;
}


void cwoPLS::Huygens(float z, float ph)
{
	//ph : unit in radian
	int N=GetPointNum();
	int Nx=GetNx();
	int Ny=GetNy();
	int Nx_h=Nx>>1;
	int Ny_h=Ny>>1;
	float spx=GetPx();
	float spy=GetPy();
	float sox=GetOx();//point offset
	float soy=GetOy();
//	float soz=GetOz();
	float dpx=GetDstPx();//destination offset
	float dpy=GetDstPy();
	float dox=GetDstOx();//destination offset
	float doy=GetDstOy();
	
	double wave_num=GetWaveNum();
	cwoObjPoint *p=GetPointBuffer();

	SetFieldType(CWO_FLD_COMPLEX);
	cwoComplex *p_dst=(cwoComplex*)GetBuffer();	

#ifdef _OPENMP
	omp_set_num_threads(GetThreads());
	#pragma omp parallel for
#endif
	for(int i=0;i<Ny;i++){
		for(int j=0;j<Nx;j++){
			float re=0.0f;
			float im=0.0f;
			int adr=j+i*Nx;
			for(int k=0; k<N; k++){
				double dx=(float)(j-Nx_h)*dpx+dox - (p[k].x+sox);
				double dy=(float)(i-Ny_h)*dpy+doy - (p[k].y+soy);
				double dz=p[k].z+/*soz*/z;

			//	cwoComplex a = p[k].a;
						
				double r=sqrt(dx*dx+dy*dy+dz*dz);
				cwoComplex e;
				re += cos(wave_num*r+(double)ph) /r;
				im += sin(wave_num*r+(double)ph) /r;
									
			}
			
			CWO_RE(p_dst[adr])=re;
			CWO_IM(p_dst[adr])=im;
		}
	}

}

void cwoPLS::Fresnel(float z, float ph)
{
	//ph : unit in radian
	int N=GetPointNum();
	int Nx=GetNx();
	int Ny=GetNy();
	int Nx_h=Nx>>1;
	int Ny_h=Ny>>1;
	float spx=GetPx();
	float spy=GetPy();
	float sox=GetOx();//point offset
	float soy=GetOy();
//	float soz=GetOz();
	float dpx=GetDstPx();//destination offset
	float dpy=GetDstPy();
	float dox=GetDstOx();//destination offset
	float doy=GetDstOy();
	
	float wave_num=GetWaveNum();
	cwoObjPoint *p=GetPointBuffer();

	cwoComplex *p_dst=(cwoComplex*)GetBuffer();	

#ifdef _OPENMP
	omp_set_num_threads(GetThreads());
	#pragma omp parallel for
#endif
	for(int i=0;i<Ny;i++){
		for(int j=0;j<Nx;j++){
			float re=0.0f;
			float im=0.0f;
			int adr=j+i*Nx;
			for(int k=0; k<N; k++){
				float dx=(float)(j-Nx_h)*dpx+dox - (p[k].x+sox);
				float dy=(float)(i-Ny_h)*dpy+doy - (p[k].y+soy);
				float dz=p[k].z+/*soz*/z;
			//	float a=p[k].a;
						
				float r=(dx*dx+dy*dy)/(2.0f*dz);
			
				float ph_tmp = wave_num*r + ph;
				re += cos(ph_tmp);
				im += sin(ph_tmp);
									
			}
			
			CWO_RE(p_dst[adr])=re;
			CWO_IM(p_dst[adr])=im;
		}
		//printf("i %d\n", i);
	}

}



void cwoPLS::FresnelTbl(float z, float ph)
{
	//ph : unit in radian
	int N = GetPointNum();
	int Nx = GetNx();
	int Ny = GetNy();
	int Nx_h = Nx >> 1;
	int Ny_h = Ny >> 1;
	float spx = GetPx();
	float spy = GetPy();
	float sox = GetOx();//point offset
	float soy = GetOy();
	//	float soz=GetOz();
	float dpx = GetDstPx();//destination offset
	float dpy = GetDstPy();
	float dox = GetDstOx();//destination offset
	float doy = GetDstOy();

	float wn = GetWaveNum();
	cwoObjPoint *p = GetPointBuffer();

	cwoComplex *p_dst = (cwoComplex*)GetBuffer();

	cwoFloat3 maxXYZ = MaxXYZ();
	cwoFloat3 minXYZ = MinXYZ();
	float z_max = maxXYZ.z;
	float z_min = minXYZ.z;
	int Nz = GetNz();

#ifdef _OPENMP
	omp_set_num_threads(GetThreads());
#pragma omp parallel for
#endif
	for (int k = 0; k < N; k ++){
		
		int idx_z = (int)((p[k].z - z_min) / fabs(z_max - z_min)* Nz);
		int m = (int)(p[k].x / dpx);
		int n = (int)(p[k].y / dpy);
	
		//printf("z %d p_z %f z_max %f z_min %f \n", idx_z, p[k].z, z_max, z_min);
		CWO *tbl = GetPtrTbl(idx_z);
		AddPixel(Nx / 2 + m, Ny / 2 + n, *tbl);
	}

}

void cwoPLS::CGHFresnel(float z, float ph)
{
	//ph : unit in radian
	int N=GetPointNum();
	int Nx=GetNx();
	int Ny=GetNy();
	int Nx_h=Nx>>1;
	int Ny_h=Ny>>1;
	float spx=GetPx();
	float spy=GetPy();
	float sox=GetOx();//point offset
	float soy=GetOy();
//	float soz=GetOz();
	float dpx=GetDstPx();//hologram offset
	float dpy=GetDstPy();
	float dox=GetDstOx();//hologram offset
	float doy=GetDstOy();
	
	float wave_num=GetWaveNum();
	cwoObjPoint *p=GetPointBuffer();

	SetFieldType(CWO_FLD_INTENSITY);
	float *cgh=(float*)GetBuffer();	

#ifdef _OPENMP
	omp_set_num_threads(GetThreads());
	#pragma omp parallel for	
#endif
	for(int i=0;i<Ny;i++){
		for(int j=0;j<Nx;j++){
			float I=0.0f;
			int adr=j+i*Nx;
			for(int k=0; k<N; k++){
				float dx=(float)(j-Nx_h)*dpx+dox - (p[k].x+sox);
				float dy=(float)(i-Ny_h)*dpy+doy - (p[k].y+soy);
				float dz=p[k].z+/*soz*/z;
				//float a=p[k].a;
						
				float r=(float)((dx*dx+dy*dy)/(2.0f*dz));
			
				I += cos(wave_num*r+ph);
									
			}
			
			cgh[adr]=I;
		}
	}

}

void cwoPLS::WRP(float z)
{

	float wn=GetWaveNum();
	float wl=GetWaveLength();
	
	int Nx=GetNx();
	int Ny=GetNy();

	int Nx_h=Nx>>1;
	int Ny_h=Ny>>1;

	float spx=GetPx(); //
	float spy=GetPy();

	float sox=GetOx();//point offset
	float soy=GetOy();
	float soz=GetOz();
	
	float dpx=GetDstPx();//wrp pitch
	float dpy=GetDstPy();
	
	float dox=GetDstOx();//wrp offset
	float doy=GetDstOy();

	cwoObjPoint *p=GetPointBuffer();

#ifdef _OPENMP
	omp_set_num_threads(GetThreads());
	#pragma omp parallel for
#endif
	for(int k=0;k<GetPointNum();k++){
		float dz=z;
		//float dz=z+p[k].z;
		//float tw=(int)(fabs(dz)*tan(asin(wl/(2.0*dpx)))/dpx);
		float tw=(int)(fabs(dz)*(wl/(2.0*dpx))/dpx);
		//int w=(int)((float)tw/1.4);
		int w=(int)tw;
	//	printf("w %d\n",w);
		
		for(int wy=-w;wy<w;wy++){
			for(int wx=-w;wx<w;wx++){//WRP coordinate

				double dx = wx*dpx;
				double dy = wy*dpy;
				double dz=p[k].z+z;
			//	float a=p[k].a;

				
				double sign=(dz>0.0) ? (1.0) : (-1.0);
				float r =dz+(dx*dx+dy*dy)/(2.0f*dz);
				//double r=sign*sqrt(dx*dx+dy*dy+dz*dz);

				int tx=(int)(p[k].x/dpx)+Nx_h;
				int ty=(int)(p[k].y/dpy)+Ny_h;

				cwoComplex tmp;
				//CWO_RE(tmp) = wn*r;//a*cosf(wn*r);
				//CWO_IM(tmp) = a*wn*r;//a*sinf(wn*r);
				CWO_RE(tmp) = cosf(wn*r);
				CWO_IM(tmp) = sinf(wn*r);

				if(tx+wx>=0 && tx+wx<Nx && ty+wy>=0 && ty+wy<Ny)
					AddPixel(wx+tx, wy+ty, tmp);
			
			}
		}
	}
	
}

void cwoPLS::MakeWrpTable(float d1, float d2, float dz)
{
		
	float wl=GetWaveLength();
	float wn=GetWaveNum();

	float dpx=GetDstPx();//wrp pitch
	float dpy=GetDstPy();
	
	float dox=GetDstOx();//wrp offset
	float doy=GetDstOy();

	int N_wrp=(int)((d2-d1)/dz)+1; //number of wr planes along depth direction
	wrp_tbl.N=N_wrp;
	wrp_tbl.radius=(int*)new int[N_wrp];
	wrp_tbl.tbl=(cwoComplex**)new cwoComplex*[N_wrp];

	for(int i=0;i<N_wrp;i++){

		float z=d1+i*dz;
		int w=(int)(fabs(z)*tan(asin(wl/(2.0*dpx)))/dpx);

		wrp_tbl.radius[i]=w;
		wrp_tbl.tbl[i]=(cwoComplex*)new cwoComplex[w*w];

		printf("i=%d w=%d\n",i,w);

		for(int wy=-w;wy<w;wy++){
			for(int wx=-w;wx<w;wx++){

				float dx = wx*dpx;
				float dy = wy*dpy;
				float r =z+(dx*dx+dy*dy)/(2.0f*dz);
	
				cwoComplex tmp;
				CWO_RE(tmp) = cosf(wn*r);
				CWO_IM(tmp) = sinf(wn*r);

				AddPixel(wx+w, wy+w, tmp);
			}
		}
	}

}


void cwoPLS::PreWrp(int Dx, int Dy, float dz)
{
	int Nx=GetNx();
	int Ny=GetNy();

	float wl=GetWaveLength();
	float dpx=GetDstPx();//wrp pitch

	cwoObjPoint *p_tmp=GetPointBuffer();
	
	int nbx=Nx/Dx; //number of blocks along to x-axis
	int nby=Ny/Dy; //number of blocks along to y-axis

	wbi=(cwoWrpBlkInfo *)__Malloc(nbx*nby*sizeof(cwoWrpBlkInfo)); //nunber of points in block
	__Memset(wbi,0,nbx*nby*sizeof(cwoWrpBlkInfo));

/*
p_tmp[0].x=10e-6 * +(+512-32);
p_tmp[0].y=0;
p_tmp[0].z=5e-3;
ctx.PLS_num=1;

p_tmp[1].x=10e-6 * +(+512-32);
p_tmp[1].y=10e-6 *-80;
p_tmp[1].z=5e-3;
ctx.PLS_num=2;
*/
	for(int n=0;n<GetPointNum();n++){
		float x=p_tmp[n].x;
		float y=p_tmp[n].y;
		float z=p_tmp[n].z;

		int bx=(int)((x/dpx+Nx/2)/Dx);		
		int by=(int)((y/dpx+Ny/2)/Dy);

		//float r=(z+dz)*tan(asin(wl/(2.0f*dpx)))/1.41421356f /dpx;//radius in pixel unit
		float r=(z + dz) * wl/(2.0f*dpx)/1.41421356f / dpx;//radius in pixel unit
		r=fabs(r);
		
		int t=(int)(r/Dx);	

		int x1=bx-t, x2=bx+t;
		int y1=by-t, y2=by+t;

		if(x1<0) x1=0;
		if(x2>nbx-1) x2=nbx-1;
		if(y1<0) y1=0;
		if(y2>nby-1) y2=nby-1;
	
		for(int i=y1;i<=y2;i++){
			for(int j=x1;j<=x2;j++){
				int adr = j+nbx*i; 
				//nib[adr]++;
				wbi[adr].N++;
			}
		}
	}

	int p_cnt=0;
	for(int i=0;i<nby;i++){
		for(int j=0;j<nbx;j++){
			int adr=j+i*nbx;
			//p_cnt+=nib[adr];
			wbi[adr].idx=p_cnt;	
			p_cnt+=wbi[adr].N;
			

		}
	}
/*	int prev_idx=0;
	for(int i=1;i<nbx*nby;i++){
		//	nib[i]+=prev_nib;
	//	prev_nib=nib[i-1];
		//wbi[i].idx=prev_idx;
		//prev_idx+=wbi[i].N;
	}
*/	



	//cwoObjPoint *p_pnt_new=(cwoObjPoint*)new cwoObjPoint[p_cnt];
	cwoObjPoint *p_pnt_new=(cwoObjPoint*)__Malloc(p_cnt*sizeof(cwoObjPoint));

	int *tmp_cnt=new int[nbx*nby];
	__Memset((void*)tmp_cnt,0,nbx*nby*sizeof(int));

	for(int n=0;n<GetPointNum();n++){
		float x=p_tmp[n].x;
		float y=p_tmp[n].y;
		float z=p_tmp[n].z;
//		float a=p_tmp[n].a;

		int bx=(int)((x/dpx+Nx/2)/Dx);		
		int by = (int)((y / dpx + Ny / 2) / Dy);

		//float r=(z + dz) * tan(asin(wl/(2.0f*dpx)))/1.41421356f / dpx;//radius in pixel unit
		float r=(z + dz) * wl/(2.0f*dpx)/1.41421356f / dpx;//radius in pixel unit
		r=fabs(r);

		int t=(int)(r/Dx);	

		int x1=bx-t, x2=bx+t;
		int y1=by-t, y2=by+t;

		if(x1<0) x1=0;
		if(x2>nbx-1) x2=nbx-1;
		if(y1<0) y1=0;
		if(y2>nby-1) y2=nby-1;
	

		for(int i=y1;i<=y2;i++) {
			for(int j=x1;j<=x2;j++) {
				int adr=j+i*nbx;
				int c=tmp_cnt[adr];
				//int idx=nib[adr]+c;
				int idx=wbi[adr].idx+c;
				
				p_pnt_new[idx].x=x;
				p_pnt_new[idx].y=y;
				p_pnt_new[idx].z=z;
			//	p_pnt_new[idx].a=a;

				tmp_cnt[adr]++;
			}
		}
	}

	SetPointNum(p_cnt);
		
	__Free((void**)&p_pnt);
	p_pnt=p_pnt_new;

	delete []tmp_cnt;
	//delete []nib;

}

void cwoPLS::WrpBlock(int Dx, int Dy, float z)
{
	int Nx=GetNx();
	int Ny=GetNy();

	float dpx=GetDstPx();
	float dpy=GetDstPy();

	float wn=GetWaveNum();

	cwoComplex *p_wrp=(cwoComplex *)GetBuffer();
	cwoObjPoint *p_obj=GetPointBuffer();
	cwoWrpBlkInfo *p_wbi=wbi;

	int nbx=Nx/Dx; //number of blocks along to x-axis
	int nby=Ny/Dy; //number of blocks along to y-axis
	
	for(int i=0;i<nby;i++){
		for(int j=0;j<nbx;j++){
			int wbi_adr=j+i*nbx;
			int N=p_wbi[wbi_adr].N;
			
			for(int k=0;k<N;k++) {	
				
			/*	int idx=p_wbi[wbi_adr].idx;
				printf("idx=%d\n",idx);
				printf("%e ", p_obj[idx+k].x);
				printf("%e ", p_obj[idx+k].y);
				printf("%e ", p_obj[idx+k].z);
				printf("\n");
*/
				for(int m=0;m<Dy;m++){
					for(int n=0;n<Dx;n++){										
						int idx=p_wbi[wbi_adr].idx;
						
						int xx=j*Dx+n;
						int yy=i*Dy+m;

						int tDx=(p_obj[idx+k].x<0)?(-Dx) : (+Dx);
						int tDy=(p_obj[idx+k].y<=0)?(+Dy) : (-Dy);
						float dx = (xx-Nx/2-tDx/2)*dpx - p_obj[idx+k].x;
						float dy = (yy-Ny/2-tDy/2)*dpy - p_obj[idx+k].y;
						//float dx = (xx-Nx/2-Dx/2)*dpx - p_obj[idx+k].x;
						//float dy = (yy-Ny/2-Dy/2)*dpy - p_obj[idx+k].y;
						float dz = z+p_obj[idx+k].z;
						
						float r = dz+(dx*dx + dy*dy)/(2.0f*dz);
					//	float r = sqrtf(dx*dx + dy*dy + dz*dz);
					//	float z = 1/r;

						p_wrp[xx+yy*Nx]+=Polar(cos(wn*r),sin(wn*r));

					}

				}
			}
		}
	}
	



}