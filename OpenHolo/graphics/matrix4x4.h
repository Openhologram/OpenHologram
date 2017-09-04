#ifndef __matrix4x4_h
#define __matrix4x4_h
#include "graphics/real.h"
#include "graphics/sys.h"
#include "graphics/vec.h"
#include <math.h>
#include <float.h>
#include <vector>

namespace graphics {

    class matrix4x4
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
      a20, a21, a22, a23,
      a30, a31, a32, a33;

      // warning: for back compatibility the [][] operator is transposed
      // to be [col][row] order!
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
		  for (int i = 0 ; i < 16 ;++i) { (&a00)[i] = 0; }
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

      matrix4x4() { makeIdentity(); }
      matrix4x4(const real array[16]) { memcpy(&a00, array, 16 * sizeof(real)); }

      matrix4x4(real a, real b, real c, real d,
              real e, real f, real g, real h,
              real i, real j, real k, real l,
              real m, real n, real o, real p) {
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
        a30 = m;
        a31 = n;
        a32 = o;
        a33 = p;
      }

      void set(real a, real b, real c, real d,
               real e, real f, real g, real h,
               real i, real j, real k, real l,
               real m, real n, real o, real p)
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
        a30 = m;
        a31 = n;
        a32 = o;
        a33 = p;
      }

      matrix4x4  operator *  ( const matrix4x4& ) const;
      matrix4x4& operator *= ( const matrix4x4& );

      matrix4x4  operator +  ( const matrix4x4& ) const;
      matrix4x4& operator += ( const matrix4x4& );
      matrix4x4  operator -  ( const matrix4x4& ) const;
      matrix4x4& operator -= ( const matrix4x4& );
      matrix4x4  operator *  (real) const;
      matrix4x4& operator *= (real);
      matrix4x4  operator /  (real d) const { return (*this) * (1 / d); }
      matrix4x4& operator /= (real d) { return (*this) *= (1 / d); }

	  void transpose();
      vec4  operator *  (const vec4& v) const
      {
        return vec4(
                 a00 * v[0] + a01 * v[1] + a02 * v[2] + a03 * v[3],
                 a10 * v[0] + a11 * v[1] + a12 * v[2] + a13 * v[3],
                 a20 * v[0] + a21 * v[1] + a22 * v[2] + a23 * v[3],
                 a30 * v[0] + a31 * v[1] + a32 * v[2] + a33 * v[3]
                 );
      }

      /*! Same as this*v. */
      //vec4 transform(const vec4& v) const { return (*this) * v; }

      /*!
         Same as the xyz of transform(v,1). This will transform a point
         in space but \e only if this is not a perspective matrix, meaning
         the last row is 0,0,0,1.
       */
      vec4 transform(const vec4& v) const
      {
        return vec4(
                 a00 * v[0] + a01 * v[1] + a02 * v[2] + a03,
                 a10 * v[0] + a11 * v[1] + a12 * v[2] + a13,
                 a20 * v[0] + a21 * v[1] + a22 * v[2] + a23,
				 1
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
      vec4 transform(const vec3& v, real w) const
      {
        return vec4(
                 a00 * v[0] + a01 * v[1] + a02 * v[2] + a03 * w,
                 a10 * v[0] + a11 * v[1] + a12 * v[2] + a13 * w,
                 a20 * v[0] + a21 * v[1] + a22 * v[2] + a23 * w,
                 a30 * v[0] + a31 * v[1] + a32 * v[2] + a33 * w
                 );
      }

      bool operator != ( const matrix4x4& b) const { return memcmp(&a00, &b.a00, 16 * sizeof(real)) != 0; }
      bool operator == ( const matrix4x4& b) const { return !memcmp(&a00, &b.a00, 16 * sizeof(real)); }

      real determinant(void) const
      {
        return
          a01 * a23 * a32 * a10 - a01 * a22 * a33 * a10 - a23 * a31 * a02 * a10 + a22 * a31 * a03 * a10
          - a00 * a23 * a32 * a11 + a00 * a22 * a33 * a11 + a23 * a30 * a02 * a11 - a22 * a30 * a03 * a11
          - a01 * a23 * a30 * a12 + a00 * a23 * a31 * a12 + a01 * a22 * a30 * a13 - a00 * a22 * a31 * a13
          - a33 * a02 * a11 * a20 + a32 * a03 * a11 * a20 + a01 * a33 * a12 * a20 - a31 * a03 * a12 * a20
          - a01 * a32 * a13 * a20 + a31 * a02 * a13 * a20 + a33 * a02 * a10 * a21 - a32 * a03 * a10 * a21
          - a00 * a33 * a12 * a21 + a30 * a03 * a12 * a21 + a00 * a32 * a13 * a21 - a30 * a02 * a13 * a21;
      }
      matrix4x4 inverse(real det) const;
      matrix4x4 inverse() const { return inverse(determinant()); }

      static const matrix4x4 _identity;
      static const matrix4x4& identity() { return _identity; }

      /*! Replace the contents with the identity. */
      void makeIdentity() { *this = _identity; }

      /*! Return whether all of the components are valid numbers. */
      bool isValid() const 
      {
        if (isNaN(a00) || isNaN(a10) || isNaN(a20) || isNaN(a30) ||
            isNaN(a01) || isNaN(a11) || isNaN(a21) || isNaN(a31) ||
            isNaN(a02) || isNaN(a12) || isNaN(a22) || isNaN(a32) ||
            isNaN(a03) || isNaN(a13) || isNaN(a23) || isNaN(a33))
          return false;
        return true;
      }
	  vec3 translation() { return vec3(a03, a13, a23); }
	  vec3 x_axis() { return vec3(a00, a10, a20); }
	  vec3 y_axis() { return vec3(a01, a11, a21); }
	  vec3 z_axis() { return vec3(a02, a12, a22); }
	  void set_x_axis(const vec3& v) { a00 = v[0]; a10 = v[1]; a20 = v[2]; }
	  void set_y_axis(const vec3& v) { a01 = v[0]; a11 = v[1]; a21 = v[2]; }
	  void set_z_axis(const vec3& v) { a02 = v[0]; a12 = v[1]; a22 = v[2]; }
	  void set_translation(const vec3& v) { a03 = v[0]; a13 = v[1]; a23 = v[2]; }
	  vec3 scale() { return vec3(sqrtf(a00*a00 + a10*a10 + a20*a20), sqrtf(a01*a01 + a11*a11 + a21*a21), sqrtf(a02*a02 + a12*a12 + a22*a22)); }


	  void get_euler_angle(real& x, real& y, real&z);
    };
	void transpose();

    //! convert a DD::Image::matrix4x4 to a std::vector<double>
    //! transposes from column-major to row-major
    inline std::vector<real> matrix4x4ToVector(const matrix4x4& matrix)
    {
      std::vector<real> ret(16);
      for (int i = 0; i < 16;++i) {
        ret[i] = matrix[i % 4][i / 4];
      }
      return ret;
    }
    
    //! convert a std::vector<double> to a DD::Image::matrix4x4
    //! transposes from row-major to column-major
    inline matrix4x4 VectorTomatrix4x4(const std::vector<real>& matrix)
    {

      matrix4x4 ret;
      for (int i = 0; i < 16 ;++i) {
        ret[i % 4][i / 4] = real(matrix[i]);
      }
      return ret;
    }

}


#endif