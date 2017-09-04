
#include "graphics/matrix3x3.h"
#include "graphics/matrix3x4.h"

namespace graphics {


const matrix3x3 matrix3x3::_identity(1, 0, 0, 
	                             0, 1, 0,
								 0, 0, 1);


void matrix3x3::transpose()
{
	real tmp;
	tmp = a01;
	a01 = a10;
	a10 = tmp;

	tmp = a02;
	a02 = a20;
	a20 = tmp;

	tmp = a12;
	a12 = a21;
	a21 = tmp;
}
matrix3x3 operator * (real a, const matrix3x3& b)
{
	matrix3x3 ret;
	for (int i = 0 ; i < 3 ;++i)
		for (int j = 0 ; j < 3 ; ++j)
			ret[i][j] = a * b[i][j];
	return ret;
}

void matrix3x3::rotationX(real ang)
{
	real c = (real) cos(ang);   
	real s = (real) sin(ang);  
	makeIdentity();
	a11 = c;
	a21 = s;
	a12 = -s;
	a22 = c;
}

void matrix3x3::rotationY(real ang)
{
	real c = (real) cos(ang);   
	real s = (real) sin(ang);  
	makeIdentity();
	a00 = c;   
	a02 = s;   
	a20 = -s;   
	a22 = c;  
}
void matrix3x3::rotationZ(real ang)
{
	real c = (real) cos(ang);   
	real s = (real) sin(ang);  
	makeIdentity();
	a00 = c;   
	a01 = -s;   
	a10 = s;   
	a11 = c; 
}

void matrix3x3::rotation(real a, real x, real y, real z)
{
	vec3 axis(x, y, z);
	axis.unit();

	real s = (real)sin( a );   
	real c = (real)cos( a );   
	x = axis[0], y = axis[1], z = axis[2];   

	makeIdentity();

	a00 = x*x*(1-c)+c;   
	a01 = x*y*(1-c)-(z*s);   
	a02 = x*z*(1-c)+(y*s);   
 
	a10 = y*x*(1-c)+(z*s);   
	a11 = y*y*(1-c)+c;   
	a12 = y*z*(1-c)-(x*s);   

	a20 = z*x*(1-c)-(y*s);   
	a21 = z*y*(1-c)+(x*s);   
	a22 = z*z*(1-c)+c;   
}


void matrix3x3::scaling(real v)
{
	makeIdentity();
	a00 = v;   
	a11 = v;   
	a22 = v;   
}
void matrix3x3::scaling(real v1, real v2, real v3)
{
	makeIdentity();
	a00 = v1;   
	a11 = v2;   
	a22 = v3;   
}


void matrix3x3::scale(real s)
{
	matrix3x3 m;
	m.scaling(s);
	*this *= m;
}

void matrix3x3::scale(real xs, real ys, real zs)
{
	matrix3x3 m;
	m.scaling(xs, ys, zs);
	*this *= m;
}

matrix3x3  matrix3x3::operator *  ( const matrix3x3& b) const
{
	matrix3x3 ret;
	ret.a00 = a00*b.a00 + a01*b.a10 + a02*b.a20;
	ret.a01 = a00*b.a01 + a01*b.a11 + a02*b.a21;
	ret.a02 = a00*b.a02 + a01*b.a12 + a02*b.a22;

	ret.a10 = a10*b.a00 + a11*b.a10 + a12*b.a20 ;
	ret.a11 = a10*b.a01 + a11*b.a11 + a12*b.a21;
	ret.a12 = a10*b.a02 + a11*b.a12 + a12*b.a22 ;

	ret.a20 = a20*b.a00 + a21*b.a10 + a22*b.a20 ;
	ret.a21 = a20*b.a01 + a21*b.a11 + a22*b.a21 ;
	ret.a22 = a20*b.a02 + a21*b.a12 + a22*b.a22;


	return ret;
}
matrix3x4  matrix3x3::operator *  ( const matrix3x4& b) const
{
	matrix3x4 ret;
	ret.a00 = a00*b.a00 + a01*b.a10 + a02*b.a20;
	ret.a01 = a00*b.a01 + a01*b.a11 + a02*b.a21;
	ret.a02 = a00*b.a02 + a01*b.a12 + a02*b.a22;
	ret.a03 = a00*b.a03 + a01*b.a13 + a02*b.a23;

	ret.a10 = a10*b.a00 + a11*b.a10 + a12*b.a20 ;
	ret.a11 = a10*b.a01 + a11*b.a11 + a12*b.a21;
	ret.a12 = a10*b.a02 + a11*b.a12 + a12*b.a22 ;
	ret.a13 = a10*b.a03 + a11*b.a13 + a12*b.a23 ;

	ret.a20 = a20*b.a00 + a21*b.a10 + a22*b.a20 ;
	ret.a21 = a20*b.a01 + a21*b.a11 + a22*b.a21 ;
	ret.a22 = a20*b.a02 + a21*b.a12 + a22*b.a22;
	ret.a23 = a20*b.a03 + a21*b.a13 + a22*b.a23;


	return ret;
}
matrix3x3& matrix3x3::operator *= ( const matrix3x3& b)
{
	matrix3x3 ret;
	ret = *this * b;
	*this = ret;
	return *this;
}

matrix3x3  matrix3x3::operator +  ( const matrix3x3& b) const
{
	matrix3x3 ret;
	for (int i = 0 ; i < 9 ;++i)
			(&ret.a00)[i] = (&a00)[i] + (&b.a00)[i];

	return ret;
}

matrix3x3& matrix3x3::operator += ( const matrix3x3& b)
{
	for (int i = 0 ; i < 9 ;++i)
		(&a00)[i] = (&a00)[i] + (&b.a00)[i];
	return *this;
}

matrix3x3  matrix3x3::operator -  ( const matrix3x3& b) const
{
	matrix3x3 ret;
	for (int i = 0 ; i < 9 ;++i)
		(&ret.a00)[i] = (&a00)[i] - (&b.a00)[i];

	return ret;
}

matrix3x3& matrix3x3::operator -= ( const matrix3x3& b)
{
	for (int i = 0 ; i < 9 ;++i)
		(&a00)[i] = (&a00)[i] - (&b.a00)[i];
	return *this;
}


matrix3x3  matrix3x3::operator *  (real b) const
{
	matrix3x3 ret;
	for (int i = 0 ; i < 9 ;++i)
		(&ret.a00)[i] = (&a00)[i] * b;

	return ret;
}

matrix3x3&  matrix3x3::operator *= (real b)
{
	for (int i = 0 ; i < 9 ;++i)
		(&a00)[i] = (&a00)[i] * b;
	return *this;
}


matrix3x3 matrix3x3::inverse(real det) const
{
	 matrix3x3 in;
	 const matrix3x3& a = *this;

	 in[0][0]=(a[1][1]*a[2][2]-a[2][1]*a[1][2])/det;
	 in[1][0]=-(a[1][0]*a[2][2]-a[1][2]*a[2][0])/det;
	 in[2][0]=(a[1][0]*a[2][1]-a[2][0]*a[1][1])/det;
	 in[0][1]=-(a[0][1]*a[2][2]-a[0][2]*a[2][1])/det;
	 in[1][1]=(a[0][0]*a[2][2]-a[0][2]*a[2][0])/det;
	 in[2][1]=-(a[0][0]*a[2][1]-a[2][0]*a[0][1])/det;
	 in[0][2]=(a[0][1]*a[1][2]-a[0][2]*a[1][1])/det;
	 in[1][2]=-(a[0][0]*a[1][2]-a[1][0]*a[0][2])/det;
	 in[2][2]=(a[0][0]*a[1][1]-a[1][0]*a[0][1])/det;
	 return in;
}

}