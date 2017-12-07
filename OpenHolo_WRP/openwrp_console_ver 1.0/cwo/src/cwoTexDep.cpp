// Copyright (C) Tomoyoshi Shimobaba 2011-

#include"cwoTexDep.h"


void cwoTexDep::LoadTex(char *fname, int c)
{
	c_tex.Load(fname,c);

	int Nx=c_tex.GetNx();
	int Ny=c_tex.GetNy();
	Create(Nx,Ny);
	//Clear();
	SetFieldType(CWO_FLD_COMPLEX);
}

void cwoTexDep::LoadDep(char *fname, int c)
{
	c_dep.Load(fname,c);

	int Nx=c_dep.GetNx();
	int Ny=c_dep.GetNy();
	Create(Nx,Ny);
	//Clear();
	SetFieldType(CWO_FLD_COMPLEX);
}


void cwoTexDep::WRP(float z)
{
	float px=GetPx();
	float py=GetPy();
	float wn=GetWaveNum();
	float wl=GetWaveLength();
	int Nx=GetNx();
	int Ny=GetNy();

	#pragma omp parallel for schedule(static) num_threads(8)
	for(int ty=0;ty<Ny;ty++){		
		for(int tx=0;tx<Nx;tx++){ //texture coordinate
			
			float t;

			c_tex.GetPixel(tx,ty,t);
			int tex=t;

			if(tex!=0){

				float dz=z;
				float tw=(int)(abs(dz)*tan(asin(wl/(2.0*px)))/px);
				//int w=(int)((float)tw/1.4);
				int w=(int)tw;

				for(int wy=-w;wy<w;wy++){
					for(int wx=-w;wx<w;wx++){//WRP coordinate

						float dx = wx*px;
						float dy = wy*py;
										
						float r =z+(dx*dx+dy*dy)/(2.0*dz);

						cwoComplex tmp;
						CWO_RE(tmp) = tex*cosf(wn*r);
						CWO_IM(tmp) = tex*sinf(wn*r);

						if(tx+wx>=0 && tx+wx<Nx && ty+wy>=0 && ty+wy<Ny)
							AddPixel(wx+tx, wy+ty, tmp);
			
					}
				}
			}
		}
	}
}


void cwoTexDep::WRP(float z, float delta_z, cwoInt2 range)
{
	
	float px=GetPx();
	float py=GetPy();
	float wn=GetWaveNum();
	float wl=GetWaveLength();
	int Nx=GetNx();
	int Ny=GetNy();

	#pragma omp parallel for schedule(static) num_threads(8)
	for(int ty=0;ty<Ny;ty++){
		
		for(int tx=0;tx<Nx;tx++){ //texture coordinate
			
			float t;

			c_tex.GetPixel(tx,ty,t);
			int tex=t;

			if(tex!=0){

				int dep=t;

				if(c_dep.GetBuffer() != NULL) 
					c_dep.GetPixel(tx,ty,t);
				else
					dep=0;


				if(dep >= range.x1 && dep < range.x2){
							
					float dz=z+(range.x2-dep)*delta_z;

					float tw=(int)(abs(dz)*tan(asin(wl/(2.0*px)))/px);
					int w=(int)((float)tw/1.4);

					for(int wy=-w;wy</*w*/0;wy++){ //half zone plate		
						for(int wx=-w;wx<w;wx++){//WRP coordinate

							float dx = wx*px;
							float dy = wy*py;
										
							float r =(dx*dx+dy*dy)/(2.0*dz);

							cwoComplex tmp;
							CWO_RE(tmp) = tex*cosf(wn*r);
							CWO_IM(tmp) = tex*sinf(wn*r);

							if(tx+wx>=0 && tx+wx<Nx && ty+wy>=0 && ty+wy<Ny)
								AddPixel(wx+tx, wy+ty, tmp);
			
						}
					}
				}
			}
		}

	}

}