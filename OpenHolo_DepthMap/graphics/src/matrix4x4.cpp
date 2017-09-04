
#include "graphics/matrix4x4.h"
#include <graphics/EulerAngles.h>

namespace graphics {


const matrix4x4 matrix4x4::_identity(1, 0, 0, 0,
	                             0, 1, 0, 0,
								 0, 0, 1, 0,
								 0, 0, 0, 1);



void matrix4x4::transpose()
{
	real tmp;
	tmp = a01;
	a01 = a10;
	a10 = tmp;

	tmp = a02;
	a02 = a20;
	a20 = tmp;

	tmp = a03;
	a03 = a30;
	a30 = tmp;

	tmp = a12;
	a12 = a21;
	a21 = tmp;

	tmp = a13;
	a13 = a31;
	a31 = tmp;

	tmp = a23;
	a23 = a32;
	a32 = tmp;
}

matrix4x4  matrix4x4::operator *  ( const matrix4x4& b) const
{
	matrix4x4 ret;
	ret.a00 = a00*b.a00 + a01*b.a10 + a02*b.a20 + a03*b.a30;
	ret.a01 = a00*b.a01 + a01*b.a11 + a02*b.a21 + a03*b.a31;
	ret.a02 = a00*b.a02 + a01*b.a12 + a02*b.a22 + a03*b.a32;
	ret.a03 = a00*b.a03 + a01*b.a13 + a02*b.a23 + a03*b.a33;

	ret.a10 = a10*b.a00 + a11*b.a10 + a12*b.a20 + a13*b.a30;
	ret.a11 = a10*b.a01 + a11*b.a11 + a12*b.a21 + a13*b.a31;
	ret.a12 = a10*b.a02 + a11*b.a12 + a12*b.a22 + a13*b.a32;
	ret.a13 = a10*b.a03 + a11*b.a13 + a12*b.a23 + a13*b.a33;

	ret.a20 = a20*b.a00 + a21*b.a10 + a22*b.a20 + a23*b.a30;
	ret.a21 = a20*b.a01 + a21*b.a11 + a22*b.a21 + a23*b.a31;
	ret.a22 = a20*b.a02 + a21*b.a12 + a22*b.a22 + a23*b.a32;
	ret.a23 = a20*b.a03 + a21*b.a13 + a22*b.a23 + a23*b.a33;

	ret.a30 = a30*b.a00 + a31*b.a10 + a32*b.a20 + a33*b.a30;
	ret.a31 = a30*b.a01 + a31*b.a11 + a32*b.a21 + a33*b.a31;
	ret.a32 = a30*b.a02 + a31*b.a12 + a32*b.a22 + a33*b.a32;
	ret.a33 = a30*b.a03 + a31*b.a13 + a32*b.a23 + a33*b.a33;
	return ret;
}

matrix4x4& matrix4x4::operator *= ( const matrix4x4& b)
{
	matrix4x4 ret;
	ret = *this * b;
	*this = ret;
	return *this;
}

matrix4x4  matrix4x4::operator +  ( const matrix4x4& b) const
{
	matrix4x4 ret;
	for (int i = 0 ; i < 16 ;++i)			(&ret.a00)[i] = (&a00)[i] + (&b.a00)[i];

	return ret;
}

matrix4x4& matrix4x4::operator += ( const matrix4x4& b)
{
	for (int i = 0 ; i < 16 ;++i)		(&a00)[i] = (&a00)[i] + (&b.a00)[i];
	return *this;
}

matrix4x4  matrix4x4::operator -  ( const matrix4x4& b) const
{
	matrix4x4 ret;
	for (int i = 0 ; i < 16 ;++i)		(&ret.a00)[i] = (&a00)[i] - (&b.a00)[i];

	return ret;
}

matrix4x4& matrix4x4::operator -= ( const matrix4x4& b)
{
	for (int i = 0 ; i < 16 ;++i)		(&a00)[i] = (&a00)[i] - (&b.a00)[i];
	return *this;
}


matrix4x4  matrix4x4::operator *  (real b) const
{
	matrix4x4 ret;
	for (int i = 0 ; i < 16 ;++i)		(&ret.a00)[i] = (&a00)[i] * b;

	return ret;
}

matrix4x4&  matrix4x4::operator *= (real b)
{
	for (int i = 0 ; i < 16 ;++i)		(&a00)[i] = (&a00)[i] * b;
	return *this;
}


 matrix4x4 matrix4x4::inverse(real det) const
 {
	 matrix4x4 inv;
	 matrix4x4& m = * const_cast<matrix4x4*>(this);

	 inv[0][0]=

	 m[1][1]*m[2][2]*m[3][3] 

	 -m[1][1]*m[2][3]*m[3][2] 

	 -m[2][1]*m[1][2]*m[3][3] 

	 +m[2][1]*m[1][3]*m[3][2]

	 +m[3][1]*m[1][2]*m[2][3] 

	 -m[3][1]*m[1][3]*m[2][2];

	 inv[0][1]=

		 -m[0][1]*m[2][2]*m[3][3] 

	 +m[0][1]*m[2][3]*m[3][2] 

	 +m[2][1]*m[0][2]*m[3][3] 

	 -m[2][1]*m[0][3]*m[3][2]

	 -m[3][1]*m[0][2]*m[2][3] 

	 +m[3][1]*m[0][3]*m[2][2];

	 inv[0][2]=

		 m[0][1]*m[1][2]*m[3][3] 

	 -m[0][1]*m[1][3]*m[3][2] 

	 -m[1][1]*m[0][2]*m[3][3] 

	 +m[1][1]*m[0][3]*m[3][2]

	 +m[3][1]*m[0][2]*m[1][3] 

	 -m[3][1]*m[0][3]*m[1][2];

	 inv[0][3]=

		 -m[0][1]*m[1][2]*m[2][3] 

	 +m[0][1]*m[1][3]*m[2][2] 

	 +m[1][1]*m[0][2]*m[2][3] 

	 -m[1][1]*m[0][3]*m[2][2]

	 -m[2][1]*m[0][2]*m[1][3] 

	 +m[2][1]*m[0][3]*m[1][2];

	 inv[1][0]=

		 -m[1][0]*m[2][2]*m[3][3] 

	 +m[1][0]*m[2][3]*m[3][2] 

	 +m[2][0]*m[1][2]*m[3][3] 

	 -m[2][0]*m[1][3]*m[3][2]

	 -m[3][0]*m[1][2]*m[2][3] 

	 +m[3][0]*m[1][3]*m[2][2];

	 inv[1][1]=

		 m[0][0]*m[2][2]*m[3][3] 

	 -m[0][0]*m[2][3]*m[3][2] 

	 -m[2][0]*m[0][2]*m[3][3] 

	 +m[2][0]*m[0][3]*m[3][2]

	 +m[3][0]*m[0][2]*m[2][3] 

	 -m[3][0]*m[0][3]*m[2][2];

	 inv[1][2]=

		 -m[0][0]*m[1][2]*m[3][3] 

	 +m[0][0]*m[1][3]*m[3][2] 

	 +m[1][0]*m[0][2]*m[3][3] 

	 -m[1][0]*m[0][3]*m[3][2]

	 -m[3][0]*m[0][2]*m[1][3] 

	 +m[3][0]*m[0][3]*m[1][2];

	 inv[1][3]=

		 m[0][0]*m[1][2]*m[2][3] 

	 -m[0][0]*m[1][3]*m[2][2] 

	 -m[1][0]*m[0][2]*m[2][3] 

	 +m[1][0]*m[0][3]*m[2][2]

	 +m[2][0]*m[0][2]*m[1][3] 

	 -m[2][0]*m[0][3]*m[1][2];

	 inv[2][0]=

		 m[1][0]*m[2][1]*m[3][3] 

	 -m[1][0]*m[2][3]*m[3][1] 

	 -m[2][0]*m[1][1]*m[3][3] 

	 +m[2][0]*m[1][3]*m[3][1]

	 +m[3][0]*m[1][1]*m[2][3] 

	 -m[3][0]*m[1][3]*m[2][1];

	 inv[2][1]=

		 -m[0][0]*m[2][1]*m[3][3] 

	 +m[0][0]*m[2][3]*m[3][1] 

	 +m[2][0]*m[0][1]*m[3][3] 

	 -m[2][0]*m[0][3]*m[3][1]

	 -m[3][0]*m[0][1]*m[2][3] 

	 +m[3][0]*m[0][3]*m[2][1];

	 inv[2][2]=

		 m[0][0]*m[1][1]*m[3][3] 

	 -m[0][0]*m[1][3]*m[3][1] 

	 -m[1][0]*m[0][1]*m[3][3] 

	 +m[1][0]*m[0][3]*m[3][1]

	 +m[3][0]*m[0][1]*m[1][3] 

	 -m[3][0]*m[0][3]*m[1][1];

	 inv[2][3]=

		 -m[0][0]*m[1][1]*m[2][3] 

	 +m[0][0]*m[1][3]*m[2][1] 

	 +m[1][0]*m[0][1]*m[2][3] 

	 -m[1][0]*m[0][3]*m[2][1]

	 -m[2][0]*m[0][1]*m[1][3] 

	 +m[2][0]*m[0][3]*m[1][1];

	 inv[3][0]=

		 -m[1][0]*m[2][1]*m[3][2] 

	 +m[1][0]*m[2][2]*m[3][1] 

	 +m[2][0]*m[1][1]*m[3][2] 

	 -m[2][0]*m[1][2]*m[3][1]

	 -m[3][0]*m[1][1]*m[2][2] 

	 +m[3][0]*m[1][2]*m[2][1];

	 inv[3][1]=

		 m[0][0]*m[2][1]*m[3][2] 

	 -m[0][0]*m[2][2]*m[3][1] 

	 -m[2][0]*m[0][1]*m[3][2] 

	 +m[2][0]*m[0][2]*m[3][1]

	 +m[3][0]*m[0][1]*m[2][2] 

	 -m[3][0]*m[0][2]*m[2][1];

	 inv[3][2]=

		 -m[0][0]*m[1][1]*m[3][2] 

	 +m[0][0]*m[1][2]*m[3][1] 

	 +m[1][0]*m[0][1]*m[3][2] 

	 -m[1][0]*m[0][2]*m[3][1]

	 -m[3][0]*m[0][1]*m[1][2] 

	 +m[3][0]*m[0][2]*m[1][1];

	 inv[3][3]=

		 m[0][0]*m[1][1]*m[2][2] 

	 -m[0][0]*m[1][2]*m[2][1] 

	 -m[1][0]*m[0][1]*m[2][2] 

	 +m[1][0]*m[0][2]*m[2][1]

	 +m[2][0]*m[0][1]*m[1][2]

	 -m[2][0]*m[0][2]*m[1][1];

	 inv /= det;
	 return inv;
 }


 void matrix4x4::get_euler_angle(real& x, real& y, real&z)
 {
	 matrix4x4 mat = *this;
	 vec3 xa = mat.x_axis();
	 vec3 ya = mat.y_axis();
	 vec3 za = mat.z_axis();

	 xa.unit();
	 ya.unit();
	 za.unit();
	 mat.set_x_axis(xa);
	 mat.set_y_axis(ya);
	 mat.set_z_axis(za);

	 HMatrix mat_type;
	 for (int a = 0; a < 3; a++)
		 for (int b = 0; b < 3; b++)
			 mat_type[a][b] = mat[a][b];
	 EulerAngles outAngs = Eul_FromHMatrix(mat_type, EulOrdXYZs);
	 x = outAngs.x;
	 y = outAngs.y;
	 z = outAngs.z;
 }
}