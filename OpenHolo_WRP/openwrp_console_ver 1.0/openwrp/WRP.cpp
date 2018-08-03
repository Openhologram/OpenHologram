#ifdef _OPENMP
#include<omp.h>
#endif

#include "WRP.h"

int OPHWRP::readxyz(string filename)   //read point cloud
{

	int dimen = 3;

	ifstream infilee(filename);
	string temp, temp2mat;
	vector <string> temsplit;

	if (!infilee)
	{
		cerr << "can not read the file";
		return -1;
	}

	while (getline(infilee, temp))
	{
		istringstream LineBand(temp);
		while (LineBand >> temp2mat)
		{
			//			printf("temp2mat=%e\n", stof(temp2mat.data()));
			vec_xyz.push_back(stod(temp2mat.data()));

		}
	}

	int num = vec_xyz.size() / dimen;
	obj_num = num;

	cwoObjPoint *op = (cwoObjPoint*)malloc(num*sizeof(cwoObjPoint));


	for (int i = 0; i < num; i++){
		double x, y, z;
		x = vec_xyz.at(dimen*i);
		y = vec_xyz.at(dimen*i + 1);
		z = vec_xyz.at(dimen*i + 2);

		op[i].x = x;
		op[i].y = y;
		op[i].z = z;

	}

	obj = op;

	return 0;

}
int OPHWRP::Setparameter(float m_wavelength, float m_pixelpitch, float m_resolution)
{
	SetWaveLength(m_wavelength);
	SetPitch(m_pixelpitch);
	SetDstPitch(m_pixelpitch);
	Create(m_resolution, m_resolution, 1);
	return 0;

}
int OPHWRP::Fresnelpropagation(float z)
{
	Diffract(z, CWO_FRESNEL_CONV);
	return 0;
}
int OPHWRP::SavetoImage(char* filename)
{
	SaveAsImage(filename, CWO_SAVE_AS_ARG);
	return 0;
}
void OPHWRP::SingleWRP(float z)  //generate WRP plane
{

	float wn = GetWaveNum();
	float wl = GetWaveLength();

	int Nx = GetNx();
	int Ny = GetNy();

	float spx = GetPx(); //
	float spy = GetPy();

	float sox = GetOx();//point offset
	float soy = GetOy();
	float soz = GetOz();

	float dpx = GetDstPx();//wrp pitch
	float dpy = GetDstPy();

	float dox = GetDstOx();//wrp offset
	float doy = GetDstOy();

	int Nx_h = Nx >> 1;
	int Ny_h = Ny >> 1;

	cwoObjPoint *pc = obj;

	int num = obj_num;

#ifdef _OPENMP
	omp_set_num_threads(GetThreads());
#pragma omp parallel for
#endif

	for (int k = 0; k < num; k++){
		float dz = z - pc[k].z;
		float tw = (int)fabs(wl*dz / dpx / dpx / 2 + 0.5) * 2 - 1;
		int w = (int)tw;

		int tx = (int)(pc[k].x / dpx) + Nx_h;
		int ty = (int)(pc[k].y / dpy) + Ny_h;

		printf("num=%d, tx=%d, ty=%d, w=%d\n", k, tx, ty, w);

		for (int wy = -w; wy < w; wy++){
			for (int wx = -w; wx<w; wx++){//WRP coordinate

				double dx = wx*dpx;
				double dy = wy*dpy;
				double dz = z - pc[k].z;
				

				double sign = (dz>0.0) ? (1.0) : (-1.0);
				double r = sign*sqrt(dx*dx + dy*dy + dz*dz);


				cwoComplex tmp;
				CWO_RE(tmp) = cosf(wn*r) / (r + 0.05);
				CWO_IM(tmp) = sinf(wn*r) / (r + 0.05);

				if (tx + wx >= 0 && tx + wx < Nx && ty + wy >= 0 && ty + wy < Ny)
					AddPixel(wx + tx, wy + ty, tmp);

			}
		}
	}

}
