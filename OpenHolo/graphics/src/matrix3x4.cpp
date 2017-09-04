#include "graphics/matrix3x4.h"

namespace graphics {


const matrix3x4 matrix3x4::_identity(1, 0, 0, 0,
	                             0, 1, 0, 0,
								 0, 0, 1, 0);




matrix3x4  matrix3x4::operator *  (real b) const
{
	matrix3x4 ret;
	for (int i = 0 ; i < 12 ;++i)
		(&ret.a00)[i] = (&a00)[i] * b;

	return ret;
}

matrix3x4&  matrix3x4::operator *= (real b)
{
	for (int i = 0 ; i < 12 ;++i)
		(&a00)[i] = (&a00)[i] * b;
	return *this;
}


}