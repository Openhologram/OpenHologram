#include "graphics/virtual_model.h"

namespace graphics {

vec3 VirtualModel::ComputeCenterOfModel() const
{
	return vec3(0);

}

box3 VirtualModel::GetBoundingBox() const
{
	return box3();

}

};