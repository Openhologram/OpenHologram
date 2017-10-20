#ifndef __matrix3x3_h
#define __matrix3x3_h
#include "graphics/real.h"
#include "graphics/sys.h"
#include "graphics/vec.h"
#include <vector>

namespace graphics {
class matrix3x4;

class matrix3x3 {
public:

      real
      a00, a01, a02,
      a10, a11, a12,
      a20, a21, a22;

      real* operator[](int i) { return &a00 + i * 3; }
      const real* operator[](int i) const { return &a00 + i * 3; }
      const real* array() const { return &a00; } // for passing to OpenGL

      matrix3x3() {}

      matrix3x3(real a, real b, real c,
              real d, real e, real f,
              real g, real h, real i) {
        a00 = a;
        a01 = b;
        a02 = c;
        a10 = d;
        a11 = e;
        a12 = f;
        a20 = g;
        a21 = h;
        a22 = i;
      }
      matrix3x3(const std::vector<real> & val) {
		  set(val);
      }
      void set(real a, real b, real c,
               real d, real e, real f,
               real g, real h, real i)
      {
        a00 = a;
        a01 = b;
        a02 = c;
        a10 = d;
        a11 = e;
        a12 = f;
        a20 = g;
        a21 = h;
        a22 = i;
      }

      void set(const std::vector<real>& val)
      {
        a00 = val[0];
        a01 = val[1];
        a02 = val[2];
        a10 = val[3];
        a11 = val[4];
        a12 = val[5];
        a20 = val[6];
        a21 = val[7];
        a22 = val[8];
      }

	  void print() 
	  {
		  for (int i = 0 ; i < 3 ;++i) {
			  for (int j = 0 ; j < 3 ; ++j)  {
				  LOG("%f  ", (*this)[i][j]);
			  }
			  LOG("\n");
		  }
	  }
	  void transpose();
      matrix3x3  operator *  ( const matrix3x3& ) const;
	  matrix3x4  operator *  ( const matrix3x4& ) const;
      matrix3x3& operator *= ( const matrix3x3& );

      matrix3x3  operator +  ( const matrix3x3& ) const;
      matrix3x3& operator += ( const matrix3x3& );
      matrix3x3  operator -  ( const matrix3x3& ) const;
      matrix3x3& operator -= ( const matrix3x3& );
      matrix3x3  operator *  (real) const;
      matrix3x3& operator *= (real);
      matrix3x3  operator /  (real d) const { return (*this) * (1 / d); }
      matrix3x3& operator /= (real d) { return (*this) *= (1 / d); }

	  real& operator ()(int a, int b)
		{
			return (*this)[a][b];
		}
	  const real& operator () (int a, int b) const
		{
			return (*this)[a][b];
		}

	  void setZero() {
		  for (int i = 0 ; i < 9 ;++i) { (&a00)[i] = 0; }
	  }
      /*! Returns the transformation of v by this matrix. */
      vec3  operator *  ( const vec3& v) const
      {
        return vec3(
                 a00 * v[0] + a01 * v[1] + a02 * v[2],
                 a10 * v[0] + a11 * v[1] + a12 * v[2],
                 a20 * v[0] + a21 * v[1] + a22 * v[2]);
      }
      /*! Same as this*v */
      vec3 transform(const vec3& v) const { return (*this) * v; }

      bool  operator != ( const matrix3x3& b) const { return memcmp(&a00, &b.a00, 9 * sizeof(real)) != 0; }
      bool  operator == ( const matrix3x3& b) const { return !memcmp(&a00, &b.a00, 9 * sizeof(real)); }

      real determinant() const
      {
        return
          a20 * (a01 * a12 - a02 * a11) +
          a21 * (a02 * a10 - a00 * a12) +
          a22 * (a00 * a11 - a01 * a10);
      }
      matrix3x3 inverse(real det) const;
      matrix3x3 inverse() const { return inverse(determinant()); }
	  void set_col(int i, const vec3& a) {
		  for (int j = 0 ; j < 3 ; ++j)
			  (*this)(j, i) = a[j];
	  }
      static const matrix3x3 _identity;
      static const matrix3x3& identity() { return _identity; }

	  matrix3x3 get_transpose() const { matrix3x3 tmp = *this; tmp.transpose(); return tmp; }

	  void set_diag(const vec3& val) 
	  {
		  makeIdentity();
		  a00 = val[0];
		  a11 = val[1];
		  a22 = val[2];
	  }
      /*! Replace the contents with the identity. */
      void makeIdentity() { *this = _identity; }
	  vec3 col(int i) const { vec3 ret((*this)(0,i),(*this)(1,i),(*this)(2,i)); return ret; }
	  vec3 col(int i) { vec3 ret((*this)(0,i),(*this)(1,i),(*this)(2,i)); return ret; }
	  vec3 row(int i) const { vec3 ret((*this)(i,0),(*this)(i,1),(*this)(i,2)); return ret; }
	  vec3 row(int i)  { vec3 ret((*this)(i,0),(*this)(i,1),(*this)(i,2)); return ret; }
      void scaling(real);
      void scaling(real, real, real);
      void scaling(const vec3& v) { scaling(v[0], v[1], v[2]); }
      void rotationX(real);
      void rotationY(real);
      void rotationZ(real);
      void rotation(real a) { rotationZ(a); }
      void rotation(real a, real x, real y, real z);
      void rotation(real a, const vec3& v) { rotation(a, v[0], v[1], v[2]); }

      // destructive modifications:
      //void transpose();
      void scale(real);
      void scale(real, real, real = 1);
      void scale(const vec3& v) { scale(v[0], v[1], v[2]); }
      void rotateX(real);
      void rotateY(real);
      void rotateZ(real);
      void rotate(real a) { rotateZ(a); }
      void rotate(real a, real x, real y, real z);
      void rotate(real a, const vec3& v) { rotate(a, v[0], v[1], v[2]); }
      void skew(real a);
};

matrix3x3 operator * (real a, const matrix3x3& b);
}

#endif