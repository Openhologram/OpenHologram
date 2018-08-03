// Copyright (C) Tomoyoshi Shimobaba 2011-

#ifndef _CWOPLS_H
#define _CWOPLS_H

#include "cwo.h"
#include "math.h"

typedef struct __cwoWrpTbl{
	int N;
	cwoComplex **tbl;
	int *radius;
}cwoWrpTbl;

typedef struct __cwoWrpBlkInfo{
	int N;		//number of objet points in block
	int idx;	//index for object point buffer
}cwoWrpBlkInfo;

//cwoPLS
//Complex amplitude field and CGH calculations from Point Light Source
//cwo Point Light Source
class cwoPLS : public CWO{
	cwoWrpTbl wrp_tbl;
	//int *nib;
	cwoWrpBlkInfo *wbi;

	int NzTbl;
	CWO *p_tbl;

public:
	cwoPLS();
	~cwoPLS();

	int Load(char *fname);
	//int Load(char *tex_name, char *dep_name);
	
	cwoObjPoint* GetPointBuffer();
	int GetPointNum();
	void SetPointNum(int num);
	void ScalePoint(float lim);
	void ScalePoint(float lim_x, float lim_y, float lim_z);
	void ShiftPoint(float dx, float dy, float dz);
	cwoFloat3 MaxXYZ();
	cwoFloat3 MinXYZ();
	cwoFloat3 MaxAbsXYZ();
	cwoFloat3 Centroid();

	void SetNz(int Nz);
	int GetNz();
	void PreTbl(float d1, float d2, int Nz);
	CWO *GetPtrTbl(int idx);

	void Huygens(float z, float ph=0.0f);
	void Fresnel(float z, float ph=0.0f);
	void FresnelTbl(float z, float ph = 0.0f);
	void CGHFresnel(float z, float ph=0.0f);

	//
	void MakeWrpTable(float d1, float d2, float dz);
	void WRP(float z);

	void PreWrp(int Dx, int Dy, float dz);
	void WrpBlock(int Dx, int Dy, float dz);

};





#endif
