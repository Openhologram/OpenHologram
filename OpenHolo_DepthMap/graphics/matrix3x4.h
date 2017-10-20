#ifndef __matrix3x4_h
#define __matrix3x4_h
#include "graphics/real.h"
#include "graphics/sys.h"
#include "graphics/vec.h"
#include <math.h>
#include <float.h>
#include <vector>
#include "graphics/matrix3x3.h"

namespace graphics {

    class matrix3x4
    {
      inline int isNaN(real lValue) const { 
#ifdef _WIN32
        return _isnan(lValue) != 0;
#else
		return isnan(lValue) != 0;
#endif
      }
    public:
      // warning: this relies on C++ packing these together in order.

      // Matrix is stored in transposed order to allow it to be passed
      // to OpenGL unchanged:
      real
      a00, a01, a02, a03,
      a10, a11, a12, a13,
      a20, a21, a22, a23;

      real* operator[](int i) { return &a00 + i * 4; }
      const real* operator[](int i) const { return &a00 + i * 4; }
      const real* array() const { return &a00; } 
	  real& operator ()(int a, int b)
		{
			return (*this)[a][b];
		}
	  const real& operator () (int a, int b) const
		{
			return (*this)[a][b];
		}

	  void setZero() {
		  for (int i = 0 ; i < 12 ;++i) { (&a00)[i] = 0; }
	  }
	  void print() 
	  {
		  for (int i = 0 ; i < 4 ;++i) {
			  for (int j = 0 ; j < 4 ; ++j)  {
				  LOG("%f  ", (*this)[i][j]);
			  }
			  LOG("\n");
		  }

	  }
	 
	  void set (const matrix3x3& aa) {
		  for (int i = 0 ; i < 3 ;++i) {
			  for (int j = 0 ; j < 3 ; ++j) {
				  (*this)(i,j) = aa(i, j);
				}
		  }
	  }

      matrix3x4() { makeIdentity(); }
      matrix3x4(const real array[16]) { memcpy(&a00, array, 16 * sizeof(real)); }

      matrix3x4(real a, real b, real c, real d,
              real e, real f, real g, real h,
              real i, real j, real k, real l) {
        a00 = a;
        a01 = b;
        a02 = c;
        a03 = d;
        a10 = e;
        a11 = f;
        a12 = g;
        a13 = h;
        a20 = i;
        a21 = j;
        a22 = k;
        a23 = l;
      }

      void set(real a, real b, real c, real d,
               real e, real f, real g, real h,
               real i, real j, real k, real l)
      {
        a00 = a;
        a01 = b;
        a02 = c;
        a03 = d;
        a10 = e;
        a11 = f;
        a12 = g;
        a13 = h;
        a20 = i;
        a21 = j;
        a22 = k;
        a23 = l;
      }


      matrix3x4  operator *  (real v) const;
      matrix3x4& operator *= (real v);
      matrix3x4  operator /  (real d) const { return (*this) * (1 / d); }
      matrix3x4& operator /= (real d) { return (*this) *= (1 / d); }

	  void transpose();
      vec3  operator *  (const vec4& v) const
      {
        return vec3(
                 a00 * v[0] + a01 * v[1] + a02 * v[2] + a03 * v[3],
                 a10 * v[0] + a11 * v[1] + a12 * v[2] + a13 * v[3],
                 a20 * v[0] + a21 * v[1] + a22 * v[2] + a23 * v[3]
                 );
      }

      /*! Same as this*v. */
      //vec4 transform(const vec4& v) const { return (*this) * v; }

      /*!
         Same as the xyz of transform(v,1). This will transform a point
         in space but \e only if this is not a perspective matrix, meaning
         the last row is 0,0,0,1.
       */
      vec3 transform(const vec4& v) const
      {
        return vec3(
                 a00 * v[0] + a01 * v[1] + a02 * v[2] + a03 * v[3],
                 a10 * v[0] + a11 * v[1] + a12 * v[2] + a13 * v[3],
                 a20 * v[0] + a21 * v[1] + a22 * v[2] + a23 * v[3]
                 );
      }

      /*!
         Same as the xyz of transform(v,0). This will transform a vector
         in space but \e only if this is not a perspective matrix, meaning
         the last row is 0,0,0,1.
       */
      vec3 vtransform(const vec3& v) const
      {
        return vec3(
                 a00 * v[0] + a01 * v[1] + a02 * v[2],
                 a10 * v[0] + a11 * v[1] + a12 * v[2],
                 a20 * v[0] + a21 * v[1] + a22 * v[2]
                 );
      }

      /*!
         Same as transpose().transform(v,0). If this is the inverse of
         a transform matrix, this will transform normals.
       */
      vec3 ntransform(const vec3& v) const
      {
        return vec3(
                 a00 * v[0] + a10 * v[1] + a20 * v[2],
                 a01 * v[0] + a11 * v[1] + a21 * v[2],
                 a02 * v[0] + a12 * v[1] + a22 * v[2]
                 );
      }

      /*!
         Same as this*vec4(v[0],v[1],v[2],w). Useful for doing transforms
         when w is stored in a different location than the xyz.
       */
      vec3 transform(const vec3& v, real w) const
      {
        return vec3(
                 a00 * v[0] + a01 * v[1] + a02 * v[2] + a03 * w,
                 a10 * v[0] + a11 * v[1] + a12 * v[2] + a13 * w,
                 a20 * v[0] + a21 * v[1] + a22 * v[2] + a23 * w
                 );
      }

	  vec3 col(int i) const { vec3 ret((*this)(0,i),(*this)(1,i),(*this)(2,i)); return ret; }
	  vec3 col(int i) { vec3 ret((*this)(0,i),(*this)(1,i),(*this)(2,i)); return ret; }
	  vec4 row(int i) const { vec4 ret((*this)(i,0),(*this)(i,1),(*this)(i,2), (*this)(i,3)); return ret; }
	  vec4 row(int i)  { vec4 ret((*this)(i,0),(*this)(i,1),(*this)(i,2), (*this)(i,3)); return ret; }
      bool operator != ( const matrix3x4& b) const { return memcmp(&a00, &b.a00, 16 * sizeof(real)) != 0; }
      bool operator == ( const matrix3x4& b) const { return !memcmp(&a00, &b.a00, 16 * sizeof(real)); }


	  void set_col(int i, const vec3& a) {
		  for (int j = 0 ; j < 3 ; ++j)
			  (*this)(j, i) = a[j];
	  }
	  void set_col(int i, const std::vector<real> a) {
		  for (int j = 0 ; j < 3 ; ++j)
			  (*this)(j, i) = a[j];
	  }
      static const matrix3x4 _identity;
      static const matrix3x4& identity() { return _identity; }

      /*! Replace the contents with the identity. */
      void makeIdentity() { *this = _identity; }

      /*! Return whether all of the components are valid numbers. */
      bool isValid() const 
      {
        if (isNaN(a00) || isNaN(a10) || isNaN(a20) || 
            isNaN(a01) || isNaN(a11) || isNaN(a21) || 
            isNaN(a02) || isNaN(a12) || isNaN(a22) || 
            isNaN(a03) || isNaN(a13) || isNaN(a23))
          return false;
        return true;
      }
    };


}


#endif