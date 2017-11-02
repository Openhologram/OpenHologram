#include "stdafx.h"
#include "WRP.h"

std::vector<std::string> split(const std::string& str_arg, const char token)
{
	std::vector<std::string> elements;
	size_t last = 0;
	size_t index = str_arg.find_first_of(token, last);
	while (index != std::string::npos)
	{
		elements.push_back(str_arg.substr(last, index - last));
		last = index + 1;
		index = str_arg.find_first_of(token, last);
	}
	if (index - last > 0)
	{
		elements.push_back(str_arg.substr(last, index - last));
	}
	return std::move(elements);
}

void OHWRP::LoadTex(char *fname, int c)
{
	c_tex.Load(fname, c);

	int Nx = c_tex.GetNx();
	int Ny = c_tex.GetNy();
	Create(Nx, Ny);
	//Clear();
	SetFieldType(CWO_FLD_COMPLEX);
}
/*
float* readtxt3d(string filename,const char taken, int dim)
{
	ifstream infile(filename);

	if (!infile)
	{
		cerr << "can not read the file";
		return -1;
	}

	string temp, temp2mat;
	vector <string> temsplit;
	vector <float> result;
	
	while (getline(infile, temp))
	{
		istringstream LineBand(temp);
		while (LineBand >> temp2mat)
		{
			temsplit = split(temp2mat, '\,');
			int ts = 0;
			ts=(temsplit.size());
			for (int i = 0; i < ts; i++)
			{
				result.push_back(atof(temsplit[i].data()));
			}
		}
	}




	return 0;

}*/
void OHWRP::SingleWRP(float z)
{
	float px = GetPx();
	float py = GetPy();
	float wn = GetWaveNum();
	float wl = GetWaveLength();
	int Nx = GetNx();
	int Ny = GetNy();

#pragma omp parallel for schedule(static) num_threads(8)
	for (int ty = 0; ty < Ny; ty++){
		for (int tx = 0; tx < Nx; tx++){ //texture coordinate

			float t;

			c_tex.GetPixel(tx, ty, t);
			int tex = t;

			if (tex != 0){

				float dz = z;
				float tw = (int)(abs(dz)*tan(asin(wl / (2.0*px))) / px);
				//int w=(int)((float)tw/1.4);
				int w = (int)tw;

				for (int wy = -w; wy < w; wy++){
					for (int wx = -w; wx < w; wx++){//WRP coordinate

						float dx = wx*px;
						float dy = wy*py;

						float r = z + (dx*dx + dy*dy) / (2.0*dz);
						
						cwoComplex tmp;
						CWO_RE(tmp) = tex*cosf(wn*r);
						CWO_IM(tmp) = tex*sinf(wn*r);

						if (tx + wx >= 0 && tx + wx < Nx && ty + wy >= 0 && ty + wy < Ny)
							AddPixel(wx + tx, wy + ty, tmp);

					}
				}
			}
		}
	}
}


void OHWRP::SingleWRP(float z, float delta_z, cwoInt2 range)
{
	float px = GetPx();
	float py = GetPy();
	float wn = GetWaveNum();
	float wl = GetWaveLength();
	int Nx = GetNx();
	int Ny = GetNy();

#pragma omp parallel for schedule(static) num_threads(8)
	for (int ty = 0; ty < Ny; ty++){

		for (int tx = 0; tx < Nx; tx++){ //texture coordinate

			float t;

			c_tex.GetPixel(tx, ty, t);
			int tex = t;

			if (tex != 0){

				int dep = t;

				if (c_dep.GetBuffer() != NULL)
					c_dep.GetPixel(tx, ty, t);
				else
					dep = 0;


				if (dep >= range.x && dep < range.y){

					float dz = z + (range.y - dep)*delta_z;

					float tw = (int)(abs(dz)*tan(asin(wl / (2.0*px))) / px);
					int w = (int)((float)tw / 1.4);

					for (int wy = -w; wy </*w*/0; wy++){ //half zone plate		
						for (int wx = -w; wx < w; wx++){//WRP coordinate

							float dx = wx*px;
							float dy = wy*py;

							float r = (dx*dx + dy*dy) / (2.0*dz);

							cwoComplex tmp;
							CWO_RE(tmp) = tex*cosf(wn*r);
							CWO_IM(tmp) = tex*sinf(wn*r);

							if (tx + wx >= 0 && tx + wx < Nx && ty + wy >= 0 && ty + wy < Ny)
								AddPixel(wx + tx, wy + ty, tmp);

						}
					}
				}
			}
		}

	}

}

/*
void OHWRP::LoadDep(char *fname, int c)
{
c_dep.Load(fname, c);

int Nx = c_dep.GetNx();
int Ny = c_dep.GetNy();
Create(Nx, Ny);
//Clear();
SetFieldType(CWO_FLD_COMPLEX);
}
*/