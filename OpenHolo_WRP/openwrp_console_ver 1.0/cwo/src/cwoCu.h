// Copyright (C) Tomoyoshi Shimobaba 2011-

#ifndef _CWOCU_H
#define _CWOCU_H

#include <stdio.h>
#include <math.h>

#include "cwo.h"

class cwoCu : public CWO{

public:
	//void Create(int dev, int Nx, int Ny);
	cwoCu();
	~cwoCu();

/*	void SetDev(int dev);
	void Create(int Nx, int Ny);
	void Delete(int dev);
	void* __Malloc(int size);
	void __Free(void **a);
	*/

	void* __Malloc(size_t size); 
	void __Free(void **a);
	
	
};

#endif


