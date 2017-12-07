// Copyright (C) Tomoyoshi Shimobaba 2011-


#ifndef _GWOPLS_H
#define _GWOPLS_H

#include "cwo.h"
#include "cwoPLS.h"
#include "gwo.h"
#include "gwo_lib.h"
#include "math.h"

//gwoPLS
//Complex amplitude field and CGH calculations from Point Light Source
//cwo Point Light Source
class gwoPLS : public GWO{

public:
	gwoPLS();
	~gwoPLS();
	cwoObjPoint* GetPointBuffer();

	void Send(cwoPLS &a);

	void __PLS_Fresnel(float ph=0.0f);
	void __PLS_CGH_Fresnel(float ph=0.0f);

};



#endif
