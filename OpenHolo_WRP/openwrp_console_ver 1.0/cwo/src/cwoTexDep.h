// Copyright (C) Tomoyoshi Shimobaba 2011-

#ifndef _CWOPLS_H
#define _CWOPLS_H

#include "cwo.h"
#include "math.h"


class cwoTexDep : public CWO
{
private:
	CWO c_tex;
	CWO c_dep;
public:
	void LoadTex(char *fname, int c=CWO_GREY);
	void LoadDep(char *fname, int c=CWO_GREY);

	void WRP(float z);
	void WRP(float z, float delta_z, cwoInt2 range);

};

#endif
