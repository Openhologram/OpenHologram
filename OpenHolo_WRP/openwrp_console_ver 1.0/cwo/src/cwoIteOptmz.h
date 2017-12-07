// Copyright (C) Tomoyoshi Shimobaba 2011-

#ifndef _CWO_ITE_OPTMZ_H
#define _CWO_ITE_OPTMZ_H

#include "cwo.h"

class cwoIteOptmz
{

public:
	int GS(
		CWO *src, CWO *src_rnd, CWO *dst,
		float z, 
		float ox, float oy, 
		float spx, float spy, 
		float dpx, float dpy,
		int ite_num, float wl=633e-9, 
		int diff_type=CWO_SHIFTED_FRESNEL);
	
	//void PIE	
	//void EPIE
	//
};

#endif
