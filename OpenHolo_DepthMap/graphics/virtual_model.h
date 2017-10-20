#ifndef __virtual_model_h
#define __virtual_model_h

#include "graphics/vec.h"
#include "graphics/geom.h"

namespace graphics{

class VirtualModel {

public:
	virtual vec3 ComputeCenterOfModel() const;
	virtual box3 GetBoundingBox() const;

};

};

#endif

