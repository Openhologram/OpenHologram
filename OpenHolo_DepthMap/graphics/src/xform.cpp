// xform.cpp: implementation of the xform class.
//
//////////////////////////////////////////////////////////////////////

#include "graphics/real.h"
#include "graphics/xform.h"
#include "graphics/vec.h"
#include "graphics/matrix.h"

//////////////////////////////////////////////////////////////////////
// Construction/Destruction
//////////////////////////////////////////////////////////////////////
namespace graphics {
static void SwapRow( real _matrix[4][4], int i0, int i1 )
{
  real* p0;
  real* p1;
  real t;
  p0 = &_matrix[i0][0];
  p1 = &_matrix[i1][0];
  t = *p0; *p0++ = *p1; *p1++ = t;
  t = *p0; *p0++ = *p1; *p1++ = t;
  t = *p0; *p0++ = *p1; *p1++ = t;
  t = *p0; *p0   = *p1; *p1 = t;
}

static void SwapCol( real _matrix[4][4], int j0, int j1 )
{
  real* p0;
  real* p1;
  real t;
  p0 = &_matrix[0][j0];
  p1 = &_matrix[0][j1];
  t = *p0; *p0 = *p1; *p1 = t;
  p0 += 4; p1 += 4;
  t = *p0; *p0 = *p1; *p1 = t;
  p0 += 4; p1 += 4;
  t = *p0; *p0 = *p1; *p1 = t;
  p0 += 4; p1 += 4;
  t = *p0; *p0 = *p1; *p1 = t;
}

static void ScaleRow( real _matrix[4][4], real c, int i )
{
  real* p = &_matrix[i][0];
  *p++ *= c;
  *p++ *= c;
  *p++ *= c;
  *p   *= c;
}

static void AddCxRow( real _matrix[4][4], real c, int i0, int i1 )
{
  const real* p0;
  real* p1;
  p0 = &_matrix[i0][0];
  p1 = &_matrix[i1][0];
  *p1++ += c* *p0++;
  *p1++ += c* *p0++;
  *p1++ += c* *p0++;
  *p1   += c* *p0;
}


static int Inv( const real* src, real dst[4][4], real* determinant, real* pivot )
{
    // returns rank (0, 1, 2, 3, or 4), inverse, and smallest pivot

    real M[4][4], I[4][4], x, c, d;
    int i, j, ix, jx;
    int col[4] = {0,1,2,3};
    int swapcount = 0;
    int rank = 0;

    *pivot = 0.0;
    *determinant = 0.0;

  memset( I, 0, sizeof(I) );
  I[0][0] = I[1][1] = I[2][2] = I[3][3] = 1.0;

  memcpy( M, src, sizeof(M) );

  // some loops unrolled for speed

  ix = jx = 0;
	x = fabs(M[0][0]);
  for ( i = 0; i < 4;++i ) for ( j = 0; j < 4; j++ ) {
    if ( fabs(M[i][j]) > x ) {
      ix = i;
      jx = j;
      x = fabs(M[i][j]);
    }
  }
  *pivot = x;
  if ( ix != 0 ) {
    SwapRow( M, 0, ix );
    SwapRow( I, 0, ix );
    swapcount++;
  }
  if ( jx != 0 ) {
    SwapCol( M, 0, jx );
    col[0] = jx;
    swapcount++;
  }

  if ( x > 0.0 ) {
    rank++;

    c = d = 1.0/M[0][0];
    M[0][1] *= c; M[0][2] *= c; M[0][3] *= c;
    ScaleRow( I, c, 0 );

    x *=  epsilon;

	  if (fabs(M[1][0]) > x) {
		  c = -M[1][0];
      M[1][1] += c*M[0][1]; M[1][2] += c*M[0][2]; M[1][3] += c*M[0][3];
      AddCxRow( I, c, 0, 1 );
	  }
	  if (fabs(M[2][0]) >  x) {
		  c = -M[2][0];
      M[2][1] += c*M[0][1]; M[2][2] += c*M[0][2]; M[2][3] += c*M[0][3];
      AddCxRow( I, c, 0, 2 );
	  }
	  if (fabs(M[3][0]) >  x) {
		  c = -M[3][0];
      M[3][1] += c*M[0][1]; M[3][2] += c*M[0][2]; M[3][3] += c*M[0][3];
      AddCxRow( I, c, 0, 3 );
	  }

    ix = jx = 1;
	  x = fabs(M[1][1]);
    for ( i = 1; i < 4;++i ) for ( j = 1; j < 4; j++ ) {
      if ( fabs(M[i][j]) > x ) {
        ix = i;
        jx = j;
        x = fabs(M[i][j]);
      }
    }
    if ( x < *pivot )
      *pivot = x;
    if ( ix != 1 ) {
      SwapRow( M, 1, ix );
      SwapRow( I, 1, ix );
      swapcount++;
    }
    if ( jx != 1 ) {
      SwapCol( M, 1, jx );
      col[1] = jx;
      swapcount++;
    }
    if ( x > 0.0 ) {
      rank++;

      c = 1.0/M[1][1];
      d *= c;
      M[1][2] *= c; M[1][3] *= c;
      ScaleRow( I, c, 1 );

      x *= epsilon;
      if (fabs(M[0][1]) >  x) {
        c = -M[0][1];
        M[0][2] += c*M[1][2]; M[0][3] += c*M[1][3];
        AddCxRow( I, c, 1, 0 );
      }
      if (fabs(M[2][1]) >  x) {
        c = -M[2][1];
        M[2][2] += c*M[1][2]; M[2][3] += c*M[1][3];
        AddCxRow( I, c, 1, 2 );
      }
      if (fabs(M[3][1]) >  x) {
        c = -M[3][1];
        M[3][2] += c*M[1][2]; M[3][3] += c*M[1][3];
        AddCxRow( I, c, 1, 3 );
      }

      ix = jx = 2;
	    x = fabs(M[2][2]);
      for ( i = 2; i < 4;++i ) for ( j = 2; j < 4; j++ ) {
        if ( fabs(M[i][j]) > x ) {
          ix = i;
          jx = j;
          x = fabs(M[i][j]);
        }
      }
      if ( x < *pivot )
        *pivot = x;
      if ( ix != 2 ) {
        SwapRow( M, 2, ix );
        SwapRow( I, 2, ix );
        swapcount++;
      }
      if ( jx != 2 ) {
        SwapCol( M, 2, jx );
        col[2] = jx;
        swapcount++;
      }
      if ( x > 0.0 ) {
        rank++;

        c = 1.0/M[2][2];
        d *= c;
        M[2][3] *= c;
        ScaleRow( I, c, 2 );

        x *= epsilon;
        if (fabs(M[0][2]) >  x) {
          c = -M[0][2];
          M[0][3] += c*M[2][3];
          AddCxRow( I, c, 2, 0 );
        }
        if (fabs(M[1][2]) >  x) {
          c = -M[1][2];
          M[1][3] += c*M[2][3];
          AddCxRow( I, c, 2, 1 );
        }
        if (fabs(M[3][2]) >  x) {
          c = -M[3][2];
          M[3][3] += c*M[2][3];
          AddCxRow( I, c, 2, 3 );
        }

        x = fabs(M[3][3]);
        if ( x < *pivot )
          *pivot = x;

        if ( x > 0.0 ) {
          rank++;

          c = 1.0/M[3][3];
          d *= c;
          ScaleRow( I, c, 3 );

          x *= epsilon;
          if (fabs(M[0][3]) >  x) {
            AddCxRow( I, -M[0][3], 3, 0 );
          }
          if (fabs(M[1][3]) >  x) {
            AddCxRow( I, -M[1][3], 3, 1 );
          }
          if (fabs(M[2][3]) >  x) {
            AddCxRow( I, -M[2][3], 3, 2 );
          }

          *determinant = (swapcount%2) ? -d : d;
        }
      }
    }
  }

  if ( col[3] != 3 )
    SwapRow( I, 3, col[3] );
  if ( col[2] != 2 )
    SwapRow( I, 2, col[2] );
  if ( col[1] != 1 )
    SwapRow( I, 1, col[1] );
  if ( col[0] != 0 )
    SwapRow( I, 0, col[0] );

  memcpy( dst, I, sizeof(I) );
	return rank;
}

///////////////////////////////////////////////////////////////
//
// xform constructors
//

xform::xform()
{
  memset( m_xform, 0, sizeof(m_xform) );
  m_xform[3][3] = 1.0;
}

xform::xform( int d )
{
  memset( m_xform, 0, sizeof(m_xform) );
  m_xform[0][0] = m_xform[1][1] = m_xform[2][2] = (real)d;
  m_xform[3][3] = 1.0;
}

xform::xform( real d )
{
  memset( m_xform, 0, sizeof(m_xform) );
  m_xform[0][0] = m_xform[1][1] = m_xform[2][2] = d;
  m_xform[3][3] = 1.0;
}



xform::xform( const real m[4][4] )
{
  memcpy( &m_xform[0][0], &m[0][0], sizeof(m_xform) );
}


xform::xform( const float m[4][4] )
{
  m_xform[0][0] = (real)m[0][0];
  m_xform[0][1] = (real)m[0][1];
  m_xform[0][2] = (real)m[0][2];
  m_xform[0][3] = (real)m[0][3];

  m_xform[1][0] = (real)m[1][0];
  m_xform[1][1] = (real)m[1][1];
  m_xform[1][2] = (real)m[1][2];
  m_xform[1][3] = (real)m[1][3];

  m_xform[2][0] = (real)m[2][0];
  m_xform[2][1] = (real)m[2][1];
  m_xform[2][2] = (real)m[2][2];
  m_xform[2][3] = (real)m[2][3];

  m_xform[3][0] = (real)m[3][0];
  m_xform[3][1] = (real)m[3][1];
  m_xform[3][2] = (real)m[3][2];
  m_xform[3][3] = (real)m[3][3];
}

xform::xform( const real* m )
{
  memcpy( &m_xform[0][0], m, sizeof(m_xform) );
}

xform::xform( const float* m )
{
  m_xform[0][0] = (real)m[0];
  m_xform[0][1] = (real)m[1];
  m_xform[0][2] = (real)m[2];
  m_xform[0][3] = (real)m[3];

  m_xform[1][0] = (real)m[4];
  m_xform[1][1] = (real)m[5];
  m_xform[1][2] = (real)m[6];
  m_xform[1][3] = (real)m[7];

  m_xform[2][0] = (real)m[8];
  m_xform[2][1] = (real)m[9];
  m_xform[2][2] = (real)m[10];
  m_xform[2][3] = (real)m[11];

  m_xform[3][0] = (real)m[12];
  m_xform[3][1] = (real)m[13];
  m_xform[3][2] = (real)m[14];
  m_xform[3][3] = (real)m[15];
}

xform::xform( const vec3& P,
	     const vec3& X,
	     const vec3& Y,
	     const vec3& Z)
{
  m_xform[0][0] = X[0];
  m_xform[1][0] = X[1];
  m_xform[2][0] = X[2];
  m_xform[3][0] = 0;

  m_xform[0][1] = Y[0];
  m_xform[1][1] = Y[1];
  m_xform[2][1] = Y[2];
  m_xform[3][1] = 0;

  m_xform[0][2] = Z[0];
  m_xform[1][2] = Z[1];
  m_xform[2][2] = Z[2];
  m_xform[3][2] = 0;

  m_xform[0][3] = P[0];
  m_xform[1][3] = P[1];
  m_xform[2][3] = P[2];
  m_xform[3][3] = 1;
}

xform::xform( const matrix& m )
{
  *this = m;
}

///////////////////////////////////////////////////////////////
//
// xform operator[]
//


real* xform::operator[](int i)
{
  return ( i >= 0 && i < 4 ) ? &m_xform[i][0] : NULL;
}

const real* xform::operator[](int i) const
{
  return ( i >= 0 && i < 4 ) ? &m_xform[i][0] : NULL;
}

///////////////////////////////////////////////////////////////
//
// xform operator=
//

xform& xform::operator=( int d )
{
  memset( m_xform, 0, sizeof(m_xform) );
  m_xform[0][0] = m_xform[1][1] = m_xform[2][2] = (real)d;
  m_xform[3][3] = 1.0;
  return *this;
}

xform& xform::operator=( float d )
{
  memset( m_xform, 0, sizeof(m_xform) );
  m_xform[0][0] = m_xform[1][1] = m_xform[2][2] = (real)d;
  m_xform[3][3] = 1.0;
  return *this;
}

xform& xform::operator=( real d )
{
  memset( m_xform, 0, sizeof(m_xform) );
  m_xform[0][0] = m_xform[1][1] = m_xform[2][2] = d;
  m_xform[3][3] = 1.0;
  return *this;
}
  
///////////////////////////////////////////////////////////////
//
// xform operator* operator- operator+
//
// All non-commutative operations have "this" as left hand side and
// argument as right hand side.
xform xform::operator*( const xform& rhs ) const
{
  xform m;
  m[0][0] = m_xform[0][0]*rhs[0][0] + m_xform[0][1]*rhs[1][0] + m_xform[0][2]*rhs[2][0] + m_xform[0][3]*rhs[3][0];
  m[0][1] = m_xform[0][0]*rhs[0][1] + m_xform[0][1]*rhs[1][1] + m_xform[0][2]*rhs[2][1] + m_xform[0][3]*rhs[3][1];
  m[0][2] = m_xform[0][0]*rhs[0][2] + m_xform[0][1]*rhs[1][2] + m_xform[0][2]*rhs[2][2] + m_xform[0][3]*rhs[3][2];
  m[0][3] = m_xform[0][0]*rhs[0][3] + m_xform[0][1]*rhs[1][3] + m_xform[0][2]*rhs[2][3] + m_xform[0][3]*rhs[3][3];

  m[1][0] = m_xform[1][0]*rhs[0][0] + m_xform[1][1]*rhs[1][0] + m_xform[1][2]*rhs[2][0] + m_xform[1][3]*rhs[3][0];
  m[1][1] = m_xform[1][0]*rhs[0][1] + m_xform[1][1]*rhs[1][1] + m_xform[1][2]*rhs[2][1] + m_xform[1][3]*rhs[3][1];
  m[1][2] = m_xform[1][0]*rhs[0][2] + m_xform[1][1]*rhs[1][2] + m_xform[1][2]*rhs[2][2] + m_xform[1][3]*rhs[3][2];
  m[1][3] = m_xform[1][0]*rhs[0][3] + m_xform[1][1]*rhs[1][3] + m_xform[1][2]*rhs[2][3] + m_xform[1][3]*rhs[3][3];

  m[2][0] = m_xform[2][0]*rhs[0][0] + m_xform[2][1]*rhs[1][0] + m_xform[2][2]*rhs[2][0] + m_xform[2][3]*rhs[3][0];
  m[2][1] = m_xform[2][0]*rhs[0][1] + m_xform[2][1]*rhs[1][1] + m_xform[2][2]*rhs[2][1] + m_xform[2][3]*rhs[3][1];
  m[2][2] = m_xform[2][0]*rhs[0][2] + m_xform[2][1]*rhs[1][2] + m_xform[2][2]*rhs[2][2] + m_xform[2][3]*rhs[3][2];
  m[2][3] = m_xform[2][0]*rhs[0][3] + m_xform[2][1]*rhs[1][3] + m_xform[2][2]*rhs[2][3] + m_xform[2][3]*rhs[3][3];

  m[3][0] = m_xform[3][0]*rhs[0][0] + m_xform[3][1]*rhs[1][0] + m_xform[3][2]*rhs[2][0] + m_xform[3][3]*rhs[3][0];
  m[3][1] = m_xform[3][0]*rhs[0][1] + m_xform[3][1]*rhs[1][1] + m_xform[3][2]*rhs[2][1] + m_xform[3][3]*rhs[3][1];
  m[3][2] = m_xform[3][0]*rhs[0][2] + m_xform[3][1]*rhs[1][2] + m_xform[3][2]*rhs[2][2] + m_xform[3][3]*rhs[3][2];
  m[3][3] = m_xform[3][0]*rhs[0][3] + m_xform[3][1]*rhs[1][3] + m_xform[3][2]*rhs[2][3] + m_xform[3][3]*rhs[3][3];
  return m;
}

xform xform::operator+( const xform& rhs ) const
{
  xform m;

  m[0][0] = m_xform[0][0] + rhs[0][0];
  m[0][1] = m_xform[0][1] + rhs[0][1];
  m[0][2] = m_xform[0][2] + rhs[0][2];
  m[0][3] = m_xform[0][3] + rhs[0][3];

  m[1][0] = m_xform[1][0] + rhs[1][0];
  m[1][1] = m_xform[1][1] + rhs[1][1];
  m[1][2] = m_xform[1][2] + rhs[1][2];
  m[1][3] = m_xform[1][3] + rhs[1][3];

  m[2][0] = m_xform[2][0] + rhs[2][0];
  m[2][1] = m_xform[2][1] + rhs[2][1];
  m[2][2] = m_xform[2][2] + rhs[2][2];
  m[2][3] = m_xform[2][3] + rhs[2][3];

  m[3][0] = m_xform[3][0] + rhs[3][0];
  m[3][1] = m_xform[3][1] + rhs[3][1];
  m[3][2] = m_xform[3][2] + rhs[3][2];
  m[3][3] = m_xform[3][3] + rhs[3][3];

  return m;
}

xform xform::operator-( const xform& rhs ) const
{
  xform m;

  m[0][0] = m_xform[0][0] + rhs[0][0];
  m[0][1] = m_xform[0][1] + rhs[0][1];
  m[0][2] = m_xform[0][2] + rhs[0][2];
  m[0][3] = m_xform[0][3] + rhs[0][3];

  m[1][0] = m_xform[1][0] + rhs[1][0];
  m[1][1] = m_xform[1][1] + rhs[1][1];
  m[1][2] = m_xform[1][2] + rhs[1][2];
  m[1][3] = m_xform[1][3] + rhs[1][3];

  m[2][0] = m_xform[2][0] + rhs[2][0];
  m[2][1] = m_xform[2][1] + rhs[2][1];
  m[2][2] = m_xform[2][2] + rhs[2][2];
  m[2][3] = m_xform[2][3] + rhs[2][3];

  m[3][0] = m_xform[3][0] + rhs[3][0];
  m[3][1] = m_xform[3][1] + rhs[3][1];
  m[3][2] = m_xform[3][2] + rhs[3][2];
  m[3][3] = m_xform[3][3] + rhs[3][3];

  return m;
}
  
///////////////////////////////////////////////////////////////
//
// xform
//


void xform::Zero()
{
  memset( m_xform, 0, sizeof(m_xform) );
}

void xform::Identity()
{
  memset( m_xform, 0, sizeof(m_xform) );
  m_xform[0][0] = m_xform[1][1] = m_xform[2][2] = m_xform[3][3] = 1.0;
}

void xform::Diagonal( real d )
{
  memset( m_xform, 0, sizeof(m_xform) );
  m_xform[0][0] = m_xform[1][1] = m_xform[2][2] = d;
  m_xform[3][3] = 1.0;
}

void xform::Scale( real x, real y, real z )
{
  memset( m_xform, 0, sizeof(m_xform) );
  m_xform[0][0] = x;
  m_xform[1][1] = y;
  m_xform[2][2] = z;
  m_xform[3][3] = 1.0;
}

void xform::Scale( const vec3& v )
{
  memset( m_xform, 0, sizeof(m_xform) );
  m_xform[0][0] = v[0];
  m_xform[1][1] = v[1];
  m_xform[2][2] = v[2];
  m_xform[3][3] = 1.0;
}

void xform::Scale(
		  vec3 fixed_point,
		  real scale_factor
		  )
{
  if ( fixed_point[0] == 0.0 && fixed_point[1] == 0.0 && fixed_point[2] == 0.0 )
  {
    Scale( scale_factor, scale_factor, scale_factor );
  }
  else
  {
    xform t0, t1, s;
    t0.Translation(-fixed_point );
    s.Scale( scale_factor, scale_factor, scale_factor );
    t1.Translation( fixed_point);
    operator=(t1*s*t0);
  }
}

void xform::Scale (
		   const frame& plane,
		   real x_scale_factor,
		   real y_scale_factor,
		   real z_scale_factor
		   )
{
  Shear( plane, x_scale_factor*plane.x_axis(), y_scale_factor*plane.y_axis(), z_scale_factor*plane.z_axis() );
}

void xform::Shear(
		  const frame& plane,
		  const vec3& x1,
		  const vec3& y1,
		  const vec3& z1
		  )
{
  xform t0, t1, s0(1), s1(1);
  t0.Translation( plane.get_origin() );
  s0.m_xform[0][0] = plane.x_axis()[0];
  s0.m_xform[0][1] = plane.x_axis()[1];
  s0.m_xform[0][2] = plane.x_axis()[2];
  s0.m_xform[1][0] = plane.y_axis()[0];
  s0.m_xform[1][1] = plane.y_axis()[1];
  s0.m_xform[1][2] = plane.y_axis()[2];
  s0.m_xform[2][0] = plane.z_axis()[0];
  s0.m_xform[2][1] = plane.z_axis()[1];
  s0.m_xform[2][2] = plane.z_axis()[2];
  s1.m_xform[0][0] = x1[0];
  s1.m_xform[1][0] = x1[1];
  s1.m_xform[2][0] = x1[2];
  s1.m_xform[0][1] = y1[0];
  s1.m_xform[1][1] = y1[1];
  s1.m_xform[2][1] = y1[2];
  s1.m_xform[0][2] = z1[0];
  s1.m_xform[1][2] = z1[1];
  s1.m_xform[2][2] = z1[2];
  t1.Translation( plane.get_origin());
  operator=(t1*s1*s0*t0);
}

void xform::Translation( real x, real y, real z )
{
  Identity();
  m_xform[0][3] = x;
  m_xform[1][3] = y;
  m_xform[2][3] = z;
  m_xform[3][3] = 1.0;
}

void xform::Translation( const vec3& v )
{
  Identity();
  m_xform[0][3] = v[0];
  m_xform[1][3] = v[1];
  m_xform[2][3] = v[2];
  m_xform[3][3] = 1.0;
}

void xform::PlanarProjection( const frame& plane )
{
  int i, j;
  real x[3] = {plane.x_axis()[0],plane.x_axis()[1],plane.x_axis()[2]};
  real y[3] = {plane.y_axis()[0],plane.y_axis()[1],plane.y_axis()[2]};
  real p[3] = {plane.get_origin()[0],plane.get_origin()[1],plane.get_origin()[2]};
  real q[3];
  for ( i = 0; i < 3;++i ) 
  {
    for ( j = 0; j < 3; j++ )
    {
      m_xform[i][j] = x[i]*x[j] + y[i]*y[j];
    }
    q[i] = m_xform[i][0]*p[0] + m_xform[i][1]*p[1] + m_xform[i][2]*p[2];
  }
  for ( i = 0; i < 3;++i )
  {
    m_xform[3][i] = 0.0;
    m_xform[i][3] = p[i]-q[i];
  }
  m_xform[3][3] = 1.0;
}

///////////////////////////////////////////////////////////////
//
// xform
//

void xform::ActOnLeft(real x,real y,real z,real w,real v[4]) const
{
  if ( v ) {
    v[0] = m_xform[0][0]*x + m_xform[0][1]*y + m_xform[0][2]*z + m_xform[0][3]*w;
    v[1] = m_xform[1][0]*x + m_xform[1][1]*y + m_xform[1][2]*z + m_xform[1][3]*w;
    v[2] = m_xform[2][0]*x + m_xform[2][1]*y + m_xform[2][2]*z + m_xform[2][3]*w;
    v[3] = m_xform[3][0]*x + m_xform[3][1]*y + m_xform[3][2]*z + m_xform[3][3]*w;
  }
}

void xform::ActOnRight(real x,real y,real z,real w,real v[4]) const
{
  if ( v ) {
    v[0] = m_xform[0][0]*x + m_xform[1][0]*y + m_xform[2][0]*z + m_xform[3][0]*w;
    v[1] = m_xform[0][1]*x + m_xform[1][1]*y + m_xform[2][1]*z + m_xform[3][1]*w;
    v[2] = m_xform[0][2]*x + m_xform[1][2]*y + m_xform[2][2]*z + m_xform[3][2]*w;
    v[3] = m_xform[0][3]*x + m_xform[1][3]*y + m_xform[2][3]*z + m_xform[3][3]*w;
  }
}

vec2 xform::operator*( const vec2& p ) const
{
  real xh[4], w;
  ActOnLeft(p[0],p[1],0.0,1.0,xh);
  w = (xh[3] != 0.0) ? 1.0/xh[3] : 1.0;
  return vec2( w*xh[0], w*xh[1] );
}

vec3 xform::operator*( const vec3& p ) const
{
  real xh[4], w;
  ActOnLeft(p[0],p[1],p[2],1.0,xh);
  w = (xh[3] != 0.0) ? 1.0/xh[3] : 1.0;
  return vec3( w*xh[0], w*xh[1], w*xh[2] );
}

vec4 xform::operator*( const vec4& h ) const
{
  real xh[4];
  ActOnLeft(h[0],h[1],h[2],h[3],xh);
  return vec4( xh[0],xh[1],xh[2],xh[3] );
}

bool xform::IsIdentity( real zero_tolerance ) const
{
  const real* v = &m_xform[0][0];
  for ( int i = 0; i < 3;++i )
  {
    if ( fabs(1.0 - *v++) > zero_tolerance )
      return false;
    if ( fabs(*v++) >  zero_tolerance )
      return false;
    if ( fabs(*v++) >  zero_tolerance )
      return false;
    if ( fabs(*v++) >  zero_tolerance )
      return false;
    if ( fabs(*v++) >  zero_tolerance )
      return false;
  }
  if ( fabs( 1.0 - *v ) > zero_tolerance )
    return false;
  return true;
}

int xform::IsSimilarity() const
{
  int rc = 0;
  if (    m_xform[3][0] != 0.0 
       || m_xform[3][1] != 0.0
       || m_xform[3][2] != 0.0
       || m_xform[3][3] != 1.0 )
  {
    rc = 0;
  }
  else
  {
    real tol = 1.0e-4;
    real dottol = 1.0e-3;
    real det = Determinant();
    if ( fabs(det) <= sqrt_epsilon )
    {
      // projection or worse
      rc = 0;
    }
    else
    {
      vec3 X(m_xform[0][0],m_xform[1][0],m_xform[2][0]);
      vec3 Y(m_xform[0][1],m_xform[1][1],m_xform[2][1]);
      vec3 Z(m_xform[0][2],m_xform[1][2],m_xform[2][2]);
      real sx = X.length();
      real sy = Y.length();
      real sz = Z.length();
      if (   sz == 0.0 || sy == 0.0 || sz == 0.0 
          || fabs(sx-sy) > tol || fabs(sy-sz) > tol || fabs(sz-sx) > tol )
      {
        // non-uniform scale or worse
        rc = 0;
      }
      else
      {
        real xy = inner(X,Y)/(sx*sy);
        real yz = inner(Y,Z)/(sy*sz);
        real zx = inner(Z,X)/(sz*sx);
        if ( fabs(xy) > dottol || fabs(yz) > dottol || fabs(zx) > dottol )
        {
          // shear or worse
          rc = 0;
        }
        else
        {
          rc = (det > 0.0) ? 1 : -1;
        }
      }
    }
  }
  return rc;
}


bool xform::IsZero() const
{
  const real* v = &m_xform[0][0];
  for ( int i = 0; i < 15;++i )
  {
    if ( *v++ != 0.0 )
      return false;
  }
  return true;
}


void xform::Transpose()
{
  real t;
  t = m_xform[0][1]; m_xform[0][1] = m_xform[1][0]; m_xform[1][0] = t;
  t = m_xform[0][2]; m_xform[0][2] = m_xform[2][0]; m_xform[2][0] = t;
  t = m_xform[0][3]; m_xform[0][3] = m_xform[3][0]; m_xform[3][0] = t;
  t = m_xform[1][2]; m_xform[1][2] = m_xform[2][1]; m_xform[2][1] = t;
  t = m_xform[1][3]; m_xform[1][3] = m_xform[3][1]; m_xform[3][1] = t;
  t = m_xform[2][3]; m_xform[2][3] = m_xform[3][2]; m_xform[3][2] = t;
}

int xform::Rank( real* pivot ) const
{
  real I[4][4], d = 0.0, p = 0.0;
  int r = Inv( &m_xform[0][0], I, &d, &p );
  if ( pivot )
    *pivot = p;
  return r;
}

real xform::Determinant( real* pivot ) const
{
  real I[4][4], d = 0.0, p = 0.0;
  //int rank = 
  Inv( &m_xform[0][0], I, &d, &p );
  if ( pivot )
    *pivot = p;
  if (d != 0.0 )
    d = 1.0/d;
  return d;
}

bool xform::Invert( real* pivot )
{
  real mrofx[4][4], d = 0.0, p = 0.0;
  int rank = Inv( &m_xform[0][0], mrofx, &d, &p );
  memcpy( m_xform, mrofx, sizeof(m_xform) );
  if ( pivot )
    *pivot = p;
  return (rank == 4) ? true : false;
}

xform xform::Inverse( real* pivot ) const
{
  xform inv;
  real d = 0.0, p = 0.0;
  //int rank = 
  Inv( &m_xform[0][0], inv.m_xform, &d, &p );
  if ( pivot )
    *pivot = p;
  return inv;
}

void xform::Rotation( 
        real angle,
        vec3 axis,  // 3d nonzero axis of rotation
        vec3 center  // 3d center of rotation
        )
{
  Rotation( sin(angle), cos(angle), axis, center );
}

void xform::Rotation(  
        real sin_angle,           // sin(rotation angle)
        real cos_angle,           // cos(rotation angle)
        vec3 axis,  // 3d nonzero axis of rotation
        vec3 center  // 3d center of rotation
        )
{
  Identity();
  if (sin_angle != 0.0 || cos_angle != 1.0) 
  {
    const real one_minus_cos_angle = 1.0 - cos_angle;
    vec3 a = axis;
    if ( fabs(inner(a, a) - 1.0) >  epsilon)
      a.unit();

    m_xform[0][0] = a[0]*a[0]*one_minus_cos_angle + cos_angle;
    m_xform[0][1] = a[0]*a[1]*one_minus_cos_angle - a[2]*sin_angle;
    m_xform[0][2] = a[0]*a[2]*one_minus_cos_angle + a[1]*sin_angle;

    m_xform[1][0] = a[1]*a[0]*one_minus_cos_angle + a[2]*sin_angle;
    m_xform[1][1] = a[1]*a[1]*one_minus_cos_angle + cos_angle;
    m_xform[1][2] = a[1]*a[2]*one_minus_cos_angle - a[0]*sin_angle;

    m_xform[2][0] = a[2]*a[0]*one_minus_cos_angle - a[1]*sin_angle;
    m_xform[2][1] = a[2]*a[1]*one_minus_cos_angle + a[0]*sin_angle;
    m_xform[2][2] = a[2]*a[2]*one_minus_cos_angle + cos_angle;

    if ( center[0] != 0.0 || center[1] != 0.0 || center[2] != 0.0 ) {
      m_xform[0][3] = -((m_xform[0][0]-1.0)*center[0] + m_xform[0][1]*center[1] + m_xform[0][2]*center[2]);
      m_xform[1][3] = -(m_xform[1][0]*center[0] + (m_xform[1][1]-1.0)*center[1] + m_xform[1][2]*center[2]);
      m_xform[2][3] = -(m_xform[2][0]*center[0] + m_xform[2][1]*center[1] + (m_xform[2][2]-1.0)*center[2]);
    }

    m_xform[3][0] = m_xform[3][1] = m_xform[3][2] = 0.0;
    m_xform[3][3] = 1.0;
  }
}


void xform::Rotation(
  const vec3&  X0, // initial frame X (X,Y,Z = right handed orthonormal frame)
  const vec3&  Y0, // initial frame Y
  const vec3&  Z0, // initial frame Z
  const vec3&  X1, // final frame X (X,Y,Z = another right handed orthonormal frame)
  const vec3&  Y1, // final frame Y
  const vec3&  Z1  // final frame Z
  )
{
  // transformation maps X0 to X1, Y0 to Y1, Z0 to Z1

  // F0 changes x0,y0,z0 to world X,Y,Z
  xform F0;
  F0[0][0] = X0[0]; F0[0][1] = X0[1]; F0[0][2] = X0[2];
  F0[1][0] = Y0[0]; F0[1][1] = Y0[1]; F0[1][2] = Y0[2];
  F0[2][0] = Z0[0]; F0[2][1] = Z0[1]; F0[2][2] = Z0[2];
  F0[3][3] = 1.0;

  // F1 changes world X,Y,Z to x1,y1,z1
  xform F1;
  F1[0][0] = X1[0]; F1[0][1] = Y1[0]; F1[0][2] = Z1[0];
  F1[1][0] = X1[1]; F1[1][1] = Y1[1]; F1[1][2] = Z1[1];
  F1[2][0] = X1[2]; F1[2][1] = Y1[2]; F1[2][2] = Z1[2];
  F1[3][3] = 1.0;

  *this = F1*F0;
}

void xform::Rotation( 
  const frame& plane0,
  const frame& plane1
  )
{
  Rotation( 
    plane0.get_origin(), plane0.x_axis(), plane0.y_axis(), plane0.z_axis(),
    plane1.get_origin(), plane1.x_axis(), plane1.y_axis(), plane1.z_axis()
    );
}


void xform::Rotation(   // (not strictly a rotation)
                            // transformation maps P0 to P1, P0+X0 to P1+X1, ...
  const vec3&   P0,  // initial frame center
  const vec3&  X0, // initial frame X
  const vec3&  Y0, // initial frame Y
  const vec3&  Z0, // initial frame Z
  const vec3&   P1,  // final frame center
  const vec3&  X1, // final frame X
  const vec3&  Y1, // final frame Y
  const vec3&  Z1  // final frame Z
  )
{
  // transformation maps P0 to P1, P0+X0 to P1+X1, ...

  // T0 translates point P0 to (0,0,0)
  xform T0;
  T0.Translation( -P0[0], -P0[1], -P0[2] );

  xform R;
  R.Rotation(X0,Y0,Z0,X1,Y1,Z1);

  // T1 translates (0,0,0) to point o1
  xform T1;
  T1.Translation( P1 );

  *this = T1*R*T0;
}

void xform::Mirror(
  vec3 point_on_mirror_plane,
  vec3 normal_to_mirror_plane
  )
{
  vec3 P = point_on_mirror_plane;
  vec3 N = normal_to_mirror_plane;
  N.unit();
  vec3 V = (2.0*(N[0]*P[0] + N[1]*P[1] + N[2]*P[2]))*N;
  m_xform[0][0] = 1 - 2.0*N[0]*N[0];
  m_xform[0][1] = -2.0*N[0]*N[1];
  m_xform[0][2] = -2.0*N[0]*N[2];
  m_xform[0][3] = V[0];

  m_xform[1][0] = -2.0*N[1]*N[0];
  m_xform[1][1] = 1.0 -2.0*N[1]*N[1];
  m_xform[1][2] = -2.0*N[1]*N[2];
  m_xform[1][3] = V[1];

  m_xform[2][0] = -2.0*N[2]*N[0];
  m_xform[2][1] = -2.0*N[2]*N[1];
  m_xform[2][2] = 1.0 -2.0*N[2]*N[2];
  m_xform[2][3] = V[2];

  m_xform[3][0] = 0.0;
  m_xform[3][1] = 0.0;
  m_xform[3][2] = 0.0;
  m_xform[3][3] = 1.0;
}



bool xform::ChangeBasis( 
  // General: If you have points defined with respect to planes, this
  //          computes the transformation to change coordinates from
  //          one plane to another.  The predefined world plane
  //          ON_world_plane can be used as an argument.
  // Details: If P = plane0.Evaluate( a0,b0,c0 ) and
  //          {a1,b1,c1} = ChangeBasis(plane0,plane1)*vec3(a0,b0,c0),
  //          then P = plane1.Evaluate( a1, b1, c1 )
  //          
  const frame& plane0, // initial plane
  const frame& plane1  // final plane
  )
{
  return ChangeBasis( 
    plane0.get_origin(), plane0.x_axis(), plane0.y_axis(), plane0.z_axis(),
    plane1.get_origin(), plane1.x_axis(), plane1.y_axis(), plane1.z_axis()
    );
}


bool xform::ChangeBasis(
  const vec3&  X0, // initial frame X (X,Y,Z = arbitrary basis)
  const vec3&  Y0, // initial frame Y
  const vec3&  Z0, // initial frame Z
  const vec3&  X1, // final frame X (X,Y,Z = arbitrary basis)
  const vec3&  Y1, // final frame Y
  const vec3&  Z1  // final frame Z
  )
{
  // Q = a0*X0 + b0*Y0 + c0*Z0 = a1*X1 + b1*Y1 + c1*Z1
  // then this transform will map the point (a0,b0,c0) to (a1,b1,c1)

  Zero();
  m_xform[3][3] = 1.0;
  real a,b,c,d;
  a = inner(X1, Y1);
  b = inner(X1, Z1);
  c = inner(Y1, Z1);
  real R[3][6] = {{inner(X1, X1),      a,      b,       inner(X1,X0), inner(X1,Y0), inner(X1,Z0)},
                    {    a,  inner(Y1, Y1),      c,       inner(Y1,X0), inner(Y1,Y0), inner(Y1,Z0)},
                    {    b,      c,  inner(Z1,Z1),       inner(Z1,X0), inner(Z1,Y0), inner(Z1,Z0)}};
  //real R[3][6] = {{X1*X1,      a,      b,       X0*X1, X0*Y1, X0*Z1},
  //                  {    a,  Y1*Y1,      c,       Y0*X1, Y0*Y1, Y0*Z1},
  //                  {    b,      c,  Z1*Z1,       Z0*X1, Z0*Y1, Z0*Z1}};

  // row reduce R
  int i0 = (R[0][0] >= R[1][1]) ? 0 : 1;
  if ( R[2][2] > R[i0][i0] )
    i0 = 2;
  int i1 = (i0+1)%3;
  int i2 = (i1+1)%3;
  if ( R[i0][i0] == 0.0 )
    return false;
  d = 1.0/R[i0][i0];
  R[i0][0] *= d;
  R[i0][1] *= d;
  R[i0][2] *= d;
  R[i0][3] *= d;
  R[i0][4] *= d;
  R[i0][5] *= d;
  R[i0][i0] = 1.0;
  if ( R[i1][i0] != 0.0 ) {
    d = -R[i1][i0];
    R[i1][0] += d*R[i0][0];
    R[i1][1] += d*R[i0][1];
    R[i1][2] += d*R[i0][2];
    R[i1][3] += d*R[i0][3];
    R[i1][4] += d*R[i0][4];
    R[i1][5] += d*R[i0][5];
    R[i1][i0] = 0.0;
  }
  if ( R[i2][i0] != 0.0 ) {
    d = -R[i2][i0];
    R[i2][0] += d*R[i0][0];
    R[i2][1] += d*R[i0][1];
    R[i2][2] += d*R[i0][2];
    R[i2][3] += d*R[i0][3];
    R[i2][4] += d*R[i0][4];
    R[i2][5] += d*R[i0][5];
    R[i2][i0] = 0.0;
  }

  if ( fabs(R[i1][i1]) < fabs(R[i2][i2]) ) {
    int i = i1; i1 = i2; i2 = i;
  }
  if ( R[i1][i1] == 0.0 )
    return false;
  d = 1.0/R[i1][i1];
  R[i1][0] *= d;
  R[i1][1] *= d;
  R[i1][2] *= d;
  R[i1][3] *= d;
  R[i1][4] *= d;
  R[i1][5] *= d;
  R[i1][i1] = 1.0;
  if ( R[i0][i1] != 0.0 ) {
    d = -R[i0][i1];
    R[i0][0] += d*R[i1][0];
    R[i0][1] += d*R[i1][1];
    R[i0][2] += d*R[i1][2];
    R[i0][3] += d*R[i1][3];
    R[i0][4] += d*R[i1][4];
    R[i0][5] += d*R[i1][5];
    R[i0][i1] = 0.0;
  }
  if ( R[i2][i1] != 0.0 ) {
    d = -R[i2][i1];
    R[i2][0] += d*R[i1][0];
    R[i2][1] += d*R[i1][1];
    R[i2][2] += d*R[i1][2];
    R[i2][3] += d*R[i1][3];
    R[i2][4] += d*R[i1][4];
    R[i2][5] += d*R[i1][5];
    R[i2][i1] = 0.0;
  }

  if ( R[i2][i2] == 0.0 )
    return false;
  d = 1.0/R[i2][i2];
  R[i2][0] *= d;
  R[i2][1] *= d;
  R[i2][2] *= d;
  R[i2][3] *= d;
  R[i2][4] *= d;
  R[i2][5] *= d;
  R[i2][i2] = 1.0;
  if ( R[i0][i2] != 0.0 ) {
    d = -R[i0][i2];
    R[i0][0] += d*R[i2][0];
    R[i0][1] += d*R[i2][1];
    R[i0][2] += d*R[i2][2];
    R[i0][3] += d*R[i2][3];
    R[i0][4] += d*R[i2][4];
    R[i0][5] += d*R[i2][5];
    R[i0][i2] = 0.0;
  }
  if ( R[i1][i2] != 0.0 ) {
    d = -R[i1][i2];
    R[i1][0] += d*R[i2][0];
    R[i1][1] += d*R[i2][1];
    R[i1][2] += d*R[i2][2];
    R[i1][3] += d*R[i2][3];
    R[i1][4] += d*R[i2][4];
    R[i1][5] += d*R[i2][5];
    R[i1][i2] = 0.0;
  }

  m_xform[0][0] = R[0][3];
  m_xform[0][1] = R[0][4];
  m_xform[0][2] = R[0][5];

  m_xform[1][0] = R[1][3];
  m_xform[1][1] = R[1][4];
  m_xform[1][2] = R[1][5];

  m_xform[2][0] = R[2][3];
  m_xform[2][1] = R[2][4];
  m_xform[2][2] = R[2][5];

  return true;
}

bool xform::ChangeBasis(
  const vec3&   P0,  // initial frame center
  const vec3&  X0, // initial frame X (X,Y,Z = arbitrary basis)
  const vec3&  Y0, // initial frame Y
  const vec3&  Z0, // initial frame Z
  const vec3&   P1,  // final frame center
  const vec3&  X1, // final frame X (X,Y,Z = arbitrary basis)
  const vec3&  Y1, // final frame Y
  const vec3&  Z1  // final frame Z
  )
{
  bool rc = false;
  // Q = P0 + a0*X0 + b0*Y0 + c0*Z0 = P1 + a1*X1 + b1*Y1 + c1*Z1
  // then this transform will map the point (a0,b0,c0) to (a1,b1,c1)

  xform F0(P0,X0,Y0,Z0);		// Frame 0

  // T1 translates by -P1
  xform T1;
  T1.Translation( -P1[0], -P1[1], -P1[2] );
	
  xform CB;
  rc = CB.ChangeBasis(vec3(1.0, 0.0, 0.0), vec3(0.0, 1.0, 0.0), vec3(0.0, 0.0, 1.0),X1,Y1,Z1);

  *this = CB*T1*F0;
  return rc;
}


xform& xform::operator=(const matrix& src)
{
  int i,j;
  i = src.n1;
  const int maxi = (i>4)?4:i;
  j = src.n2;
  const int maxj = (j>4)?4:j;
  Identity();
  for ( i = 0; i < maxi;++i ) for ( j = 0; j < maxj; j++ ) {
    m_xform[i][j] = src(i,j);
  }
  return *this;
}

};
