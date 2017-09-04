#include "graphics/solve.h"
#include <math.h>
#include "graphics/vec.h"
#include "graphics/_limits.h"
#include "string.h"

namespace graphics {

int
SolveQuadraticEquation(
       real a, real b, real c, 
       real *r0, real *r1
       )
/* Find solutions of a quadratic equation
 *
 * INPUT:
 *   a, b, c  coefficients defining the quadratic equation
 *            a*t^2 + b*t + c = 0
 *   r0, r1   address of reals
 * OUTPUT:
 *   ON_QuadraticEquation()
 *      0: successful - two distinct real roots (*r0 < *r1)
 *      1: successful - one real root (*r0 = *r1)
 *      2: successful - two complex conjugate roots (*r0 +/- (*r1)*sqrt(-1))
 *     -1: failure - a = 0, b != 0        (*r0 = *r1 = -c/b)
 *     -2: failure - a = 0, b  = 0 c != 0 (*r0 = *r1 = 0.0)
 *     -3: failure - a = 0, b  = 0 c  = 0 (*r0 = *r1 = 0.0)
 *
 * COMMENTS:
 *   The quadratic equation is solved using the formula
 *   roots = q/a, c/q, q = 0.5*(b + sgn(b)*sqrt(b^2 - 4ac)).
 *
 *   When |b^2 - 4*a*c| <= b*b*ON_EPSILON, the discriminant
 *   is numerical noise and is assumed to be zero.
 *
 *   If it is really important to have the best possible answer,
 *   you sould probably tune up the returned roots using
 *   Brent's algorithm.
 *
 * REFERENCE:
 *   Numerical Recipes in C, section 5.5
 *
 * RELATED FUNCTIONS:
 *   CubicEquation()
 */
{
  real q, x0, x1, y0, y1, y;

  if (a == 0.0) {
    if (b == 0.0) 
      {*r0 = *r1 = 0.0; return (c == 0.0) ? -3 : -2;}
    *r0 = *r1 = -c/b; return -1;
  }

  if (c == 0.0) {
    if (b == 0.0) 
      {*r0 = *r1 = 0.0; return 1;}
    b /= -a;
    if (b < 0.0) 
      {*r0=b;*r1=0.0;} 
    else
      {*r0=0.0;*r1=b;}
    return 0;
  }

  if (b == 0.0) {
    c /= -a;
    *r1 = sqrt(fabs(c));
    if (c < 0.0) 
      {*r0 = 0.0; return 2;}
    *r0 = -(*r1);
    return 0;
  }
  q = b*b - 4.0*a*c;
  if (fabs(q) <= b*b* epsilon) 
    q = 0.0; /* q is noise - set it to zero */
  if (q <= 0.0) {
    /* multiple real root or complex conjugate roots */
    *r0 = -0.5*b/a;
    if (q == 0.0) 
      {*r1 = *r0; return 1;}

    /* complex conjugate roots (probably) */
    *r1 = fabs(0.5*sqrt(fabs(q))/a); 
    x0 = *r0;
    x1 = *r1;
    y = (a*x0 + b)*x0 + c;            /* y = quadratic evaluated at -b/2a */
    if ((a > 0.0 && y <= 0.0) || (a < 0.0 && y >= 0.0))
      {*r1 = *r0; return 1;}
    y0 = y - a*x1*x1;                 /* y0 = real part of "zero" */
    y1 = (2.0*a*x0 + b)*x1;           /* y1 = imaginary part of "zero" */
    if (fabs(y) <= fabs(y0) || fabs(y) <= fabs(y1)) 
      {*r1 = *r0; return 1;}
    return 2;
  }

  /* distinct roots (probably) */
  q = 0.5*(fabs(b) + sqrt(q));
  if (b > 0.0) q = -q;
  x0 = q/a;
  x1 = c/q;
  if (x0 == x1) 
    {*r0 = *r1 = x0; return 1;}

  if (x0 > x1) 
    {y = x0; x0 = x1; x1 = y;}

  /* quick test to see if roots are numerically distinct from extrema */
  y = -0.5*b/a;
  if (x0 <= y && y <= x1) {
    y = (a*y + b)*y + c;              /* y = quadratic evaluated at -b/2a */
    y0 = (a*x0 + b)*x0 + c;
    y1 = (a*x1 + b)*x1 + c;
    if (fabs(y) <= fabs(y0) || fabs(y) <= fabs(y1)
        || (a > 0.0 && y > 0.0) || (a < 0.0 && y < 0.0))
      {*r0 = *r1 = -0.5*b/a; return 1;}
  }

  /* distinct roots */
  *r0 = x0;
  *r1 = x1;
  return 0;
}


static double CBRT2  = 1.2599210498948731647672;
static double CBRT4  = 1.5874010519681994747517;
static double CBRT2I = 0.79370052598409973737585;
static double CBRT4I = 0.62996052494743658238361;

real cbrt(real x)

{
int e, rem, sign;
double z;


if( x == 0 )
	return( x );
if( x > 0 )
	sign = 1;
else
	{
	sign = -1;
	x = -x;
	}

z = x;
/* extract power of 2, leaving
 * mantissa between 0.5 and 1
 */
x = frexp( x, &e );

/* Approximate cube root of number between .5 and 1,
 * peak relative error = 9.2e-6
 */
x = (((-1.3466110473359520655053e-1  * x
      + 5.4664601366395524503440e-1) * x
      - 9.5438224771509446525043e-1) * x
      + 1.1399983354717293273738e0 ) * x
      + 4.0238979564544752126924e-1;

/* exponent divided by 3 */
if( e >= 0 )
	{
	rem = e;
	e /= 3;
	rem -= 3*e;
	if( rem == 1 )
		x *= CBRT2;
	else if( rem == 2 )
		x *= CBRT4;
	}


/* argument less than 1 */

else
	{
	e = -e;
	rem = e;
	e /= 3;
	rem -= 3*e;
	if( rem == 1 )
		x *= CBRT2I;
	else if( rem == 2 )
		x *= CBRT4I;
	e = -e;
	}

/* multiply by power of 2 */
x = ldexp( x, e );

/* Newton iteration */
x -= ( x - (z/(x*x)) )*0.33333333333333333333;
#ifdef DEC
x -= ( x - (z/(x*x)) )/3.0;
#else
x -= ( x - (z/(x*x)) )*0.33333333333333333333;
#endif

if( sign < 0 )
	x = -x;
return(x);
}

static int isZero(double x)
{
return x > -zero_epsilon && x < zero_epsilon;
}
int solveLinear(double c[2], double s[1])
{
if (isZero(c[1]))
    return 0;
s[0] = - c[0] / c[1];
return 1;
}

/********************************************************
*							*
* This function determines the roots of a quadric	*
* equation.						*
* It takes as parameters a pointer to the three		*
* coefficient of the quadric equation (the c[2] is the	*
* coefficient of x2 and so on) and a pointer to the	*
* two element array in which the roots are to be	*
* placed.						*
* It outputs the number of roots found.			*
*							*
********************************************************/

int solveQuadric(double c[3], double s[2])
{
double p, q, D;


// make sure we have a d2 equation

if (isZero(c[2]))
    return solveLinear(c, s);


// normal for: x^2 + px + q
p = c[1] / (2.0 * c[2]);
q = c[0] / c[2];
D = p * p - q;

if (isZero(D))
    {
    // one double root
    s[0] = s[1] = -p;
    return 1;
    }

if (D < 0.0)
    // no real root
    return 0;

else
    {
    // two real roots
    double sqrt_D = sqrt(D);
    s[0] = sqrt_D - p;
    s[1] = -sqrt_D - p;
    return 2;
    }
}

int solveCubic(double c[4], double s[3])
{
int	i, num;
double	sub,
	A, B, C,
	sq_A, p, q,
	cb_p, D;

// normalize the equation:x ^ 3 + Ax ^ 2 + Bx  + C = 0
A = c[2] / c[3];
B = c[1] / c[3];
C = c[0] / c[3];

// substitute x = y - A / 3 to eliminate the quadric term: x^3 + px + q = 0

sq_A = A * A;
p = 1.0/3.0 * (-1.0/3.0 * sq_A + B);
q = 1.0/2.0 * (2.0/27.0 * A *sq_A - 1.0/3.0 * A * B + C);

// use Cardano's formula

cb_p = p * p * p;
D = q * q + cb_p;

if (isZero(D))
    {
    if (isZero(q))
	{
	// one triple solution
	s[0] = 0.0;
	num = 1;
	}
    else
	{
	// one single and one double solution
	double u = cbrt(-q);
	s[0] = 2.0 * u;
	s[1] = - u;
	num = 2;
	}
    }
else
    if (D < 0.0)
	{
	// casus irreductibilis: three real solutions
	double phi = 1.0/3.0 * acos(-q / sqrt(-cb_p));
	double t = 2.0 * sqrt(-p);
	s[0] = t * cos(phi);
	s[1] = -t * cos(phi + M_PI / 3.0);
	s[2] = -t * cos(phi - M_PI / 3.0);
	num = 3;
	}
    else
	{
	// one real solution
	double sqrt_D = sqrt(D);
	double u = cbrt(sqrt_D + fabs(q));
	if (q > 0.0)
	    s[0] = - u + p / u ;
	else
	    s[0] = u - p / u;
	num = 1;
	}

// resubstitute
sub = 1.0 / 3.0 * A;
for (i = 0; i < num; i++)
    s[i] -= sub;
return num;
}


/********************************************************
*							*
* This function determines the roots of a quartic	*
* equation.						*
* It takes as parameters a pointer to the five		*
* coefficient of the quartic equation (the c[4] is the	*
* coefficient of x4 and so on) and a pointer to the	*
* four element array in which the roots are to be	*
* placed. It outputs the number of roots found.		*
*							*
********************************************************/

int
solveQuartic(double c[5], double s[4])
{
double	    coeffs[4],
	    z, u, v, sub,
	    A, B, C, D,
	    sq_A, p, q, r;
int	    i, num;


// normalize the equation:x ^ 4 + Ax ^ 3 + Bx ^ 2 + Cx + D = 0

A = c[3] / c[4];
B = c[2] / c[4];
C = c[1] / c[4];
D = c[0] / c[4];

// subsitute x = y - A / 4 to eliminate the cubic term: x^4 + px^2 + qx + r = 0

sq_A = A * A;
p = -3.0 / 8.0 * sq_A + B;
q = 1.0 / 8.0 * sq_A * A - 1.0 / 2.0 * A * B + C;
r = -3.0 / 256.0 * sq_A * sq_A + 1.0 / 16.0 * sq_A * B - 1.0 / 4.0 * A * C + D;

if (isZero(r))
    {
    // no absolute term:y(y ^ 3 + py + q) = 0
    coeffs[0] = q;
    coeffs[1] = p;
    coeffs[2] = 0.0;
    coeffs[3] = 1.0;

    num = solveCubic(coeffs, s);
    s[num++] = 0;
    }
else
    {
    // solve the resolvent cubic...
    coeffs[0] = 1.0 / 2.0 * r * p - 1.0 / 8.0 * q * q;
    coeffs[1] = -r;
    coeffs[2] = -1.0 / 2.0 * p;
    coeffs[3] = 1.0;
    (void) solveCubic(coeffs, s);

    // ...and take the one real solution...
    z = s[0];

    // ...to build two quadratic equations
    u = z * z - r;
    v = 2.0 * z - p;

    if (isZero(u))
	u = 0.0;
    else if (u > 0.0)
	u = sqrt(u);
    else
	return 0;

    if (isZero(v))
	v = 0;
    else if (v > 0.0)
	v = sqrt(v);
    else
	return 0;

    coeffs[0] = z - u;
    coeffs[1] = q < 0 ? -v : v;
    coeffs[2] = 1.0;

    num = solveQuadric(coeffs, s);

    coeffs[0] = z + u;
    coeffs[1] = q < 0 ? v : -v;
    coeffs[2] = 1.0;

    num += solveQuadric(coeffs, s + num);
    }

// resubstitute
sub = 1.0 / 4 * A;
for (i = 0; i < num; i++)
    s[i] -= sub;

return num;

}

int
SolveTriDiagonal( int dim, int n, 
                          real* a, const real* b, real* c,
                          const real* d, real* X)
/*****************************************************************************
Solve a tridiagonal linear system of equations using backsubstution
 
INPUT:
  dim   (>=1) dimension of X and d
  n     (>=2) number of equations
  a,b,c,d
        coefficients of the linear system. a and c are arrays of n-1 reals.
        b and d are arrays of n reals.  Note that "a", "b" and "d" are
        not modified. "c" is modified.
  X     array of n reals 
OUTPUT:
  ON_SolveTriDiagonal()  0: success
                        -1: failure - illegal input
                        -2: failure - zero pivot encountered
                                      (can happen even when matrix is
                                       non-singular)

  X     if ON_SolveTriDiagonal() returns 0, then X is the solution to

  b[0]   c[0]                                X[0]        d[0]
  a[0]   b[1]  c[1]                          X[1]        d[1]
         a[1]  b[2]  c[2]                  * X[2]     =  d[2]
               ....  ....  ....              ...         ...
                     a[n-3] b[n-2] c[n-2]    X[n-2]      d[n-2]
                            a[n-2] b[n-1]    X[n-1]      d[n-1]

COMMENTS:
  If n <= 3, this function uses ON_Solve2x2() or ON_Solve3x3().  
  If n > 3, the system is solved in the fastest possible manner; 
  in particular,  no pivoting is performed, b[0] must be nonzero.
  If |b[i]| > |a[i-1]| + |c[i]|, then this function will succeed.
  The computation is performed in such a way that the output
  "X" pointer can be equal to the input "d" pointer; i.e., if the
  d array will not be used after the call to ON_SolveTriDiagonal(), then
  it is not necessary to allocate seperate space for X and d.
EXAMPLE:
REFERENCE:
  NRC, section 2.6
RELATED FUNCTIONS:
  Solve2x2
  Solve3x3
  SolveSVD
*****************************************************************************/
{
  real beta, g, q;
  int i, j;
  if (dim < 1 || n < 2 || !a || !b || !c || !d || !X)
    return -1;

  if (dim == 1) {
    /* standard tri-diagonal problem -  X and d are scalars */
    beta = *b++;
    if (beta == 0.0)
      return -2;
    beta = 1.0/beta;
    *X = *d++ *beta;
    i = n-1;
    while(i--) {
      g = (*c++ *= beta);
      beta = *b++ - *a * g;
      if (beta == 0.0) return -2;
      beta = 1.0/beta;
      X[1] = (*d++ - *a++ * *X)*beta;
      X++;      
    }
    X--;
    c--;
    i = n-1;
    while(i--) {
      *X -= *c-- * X[1];
      X--;
    }
  }
  else {
    /* X and d are vecs */
    beta = *b++;
    if (beta == 0.0)
      return -2;
    beta = 1.0/beta;
    j = dim;
    while(j--)
      *X++ = *d++ *beta;
    X -= dim;
    i = n-1;
    while(i--) {
      g = (*c++ *= beta);
      beta = *b++ - *a * g;
      if (beta == 0.0) return -2;
      beta = 1.0/beta;
      j = dim;
      q = *a++;
      while(j--) {
        X[dim] = (*d++ - q * *X)*beta;
        X++;      
      }
    }
    X--;
    c--;
    i = n-1;
    while(i--) {
      q = *c--;
      j = dim;
      while(j--) {
        *X -= q * X[dim];
        X--;
      }
    }
  }

  return 0;
}


int
Solve2x2( real m00, real m01, real m10, real m11, real d0, real d1,
                  real* x_addr, real* y_addr, real* pivot_ratio)
/* Solve a 2x2 system of linear equations
 *
 * INPUT:
 *   m00, m01, m10, m11, d0, d1
 *      coefficients for the 2x2 the linear system:
 *   x_addr, y_addr
 *      addresses of reals
 *   pivot_ratio
 *      address of real
 * OUTPUT:
 *   ON_Solve2x2() returns rank (0,1,2)
 *
 *   If ON_Solve2x2() is successful (return code 2), then
 *   the solution is returned in {*x_addr, *y_addr} and
 *   *pivot_ratio = min(|pivots|)/max(|pivots|).
 *
 *   WARNING: If the pivot ratio is small, then the matrix may
 *   be singular or ill conditioned.  You should test the results
 *   before you use them.
 *
 * COMMENTS:
 *      The system of 2 equations and 2 unknowns (x,y),
 *         m00*x + m01*y = d0
 *         m10*x + m11*y = d1,
 *      is solved using Gauss-Jordan elimination
 *      with full pivoting.
 * EXAMPLE:
 *      // Find the intersection of 2 2D lines where
 *      // P0, P1  are points on the lines and
 *      // D0, D1, are nonzero directions
 *      rc = ON_Solve2x2(D0[0],-D1[0],D0[1],-D1[1],P1[0]-P0[0],P1[1]-P0[1],
 *                       &x, &y,&pivot_ratio);
 *      switch(rc) {
 *      case  0: // P0 + x*D0 = P1 + y*D1 = intersection point
 *        if (pivot_ratio < 0.001) {
 *          // small pivot ratio - test answer before using ...
 *        }
 *        break;
 *      case -1: // both directions are zero!
 *        break;
 *      case -2: // parallel directions
 *        break;
 *      }
 *
 * REFERENCE:
 *      STRANG
 *
 * RELATED FUNCTIONS:
 *      Solve3x2(), Solve3x3
 */
{
  int i = 0;
  real maxpiv, minpiv;
  real x = fabs(m00);
  real y = fabs(m01); if (y > x) {x = y; i = 1;}
  y = fabs(m10); if (y > x) {x = y; i = 2;}
  y = fabs(m11); if (y > x) {x = y; i = 3;}
  *pivot_ratio = *x_addr = *y_addr = 0.0;
  if (x == 0.0) 
    return 0; // rank = 0
  minpiv = maxpiv = x;
  if (i%2) {
    {real* tmp = x_addr; x_addr = y_addr; y_addr = tmp;}
    x = m00; m00 = m01; m01 = x;
    x = m10; m10 = m11; m11 = x;
  }
  if (i > 1) {
    x = d0; d0 = d1; d1 = x;
    x = m00; m00 = m10; m10 = x;
    x = m01; m01 = m11; m11 = x;
  }
  
  x = 1.0/m00;
  m01 *= x; d0 *= x;
  if (m10 != 0.0) {m11 -= m10*m01; d1 -= m10*d0;}

  if (m11 == 0.0) 
    return 1; // rank = 1

  y = fabs(m11);
  if (y > maxpiv) maxpiv = y; else if (y < minpiv) minpiv = y;
  
  d1 /= m11;
  if (m01 != 0.0)
    d0 -= m01*d1;

  *x_addr = d0;
  *y_addr = d1;
  *pivot_ratio = minpiv/maxpiv;
  return 2;  
}


int 
Solve3x2(const real col0[3], const real col1[3], 
                real d0, real d1, real d2,
                real* x_addr, real* y_addr, real* err_addr, real* pivot_ratio)
/* Solve a 3x2 system of linear equations
 *
 * INPUT:
 *   col0, col1
 *      arrays of 3 reals
 *   d0, d1, d2
 *      right hand column of system
 *   x_addr, y_addr, err_addr, pivot_ratio
 *      addresses of reals
 * OUTPUT:
 *   TL_Solve3x2()
 *       2: successful
 *       0: failure - 3x2 matrix has rank 0
 *       1: failure - 3x2 matrix has rank 1
 *      If the return code is zero, then
 *        (*x_addr)*{col0} + (*y_addr)*{col1}
 *        + (*err_addr)*({col0 X col1}/|col0 X col1|)
 *        = {d0,d1,d2}.
 *      pivot_ratio = min(|pivots|)/max(|pivots|)  If this number
 *      is small, then the 3x2 matrix may be singular or ill 
 *      conditioned.
 * COMMENTS:
 *      The system of 3 equations and 2 unknowns (x,y),
 *              x*col0[0] + y*col1[1] = d0
 *              x*col0[0] + y*col1[1] = d1
 *              x*col0[0] + y*col1[1] = d2,
 *      is solved using Gauss-Jordan elimination
 *      with full pivoting.
 * EXAMPLE:
 *      // If A, B and T are 3D vectors, find a and b so that
 *      // T - a*A + b*B is perpindicular to both A and B.
 *      rc = TL_Solve3x3(A,B,T[0],T[1],T[2],&a,&b,&len);
 *      switch(rc) {
 *      case  0: // {x,y,z} = intersection point, len = T o (A X B / |A X B|)
 *        break;
 *      case -1: // both A and B are zero!
 *        break;
 *      case -2: // A and B are parallel, or one of A and B is zero.
 *        break;
 *      }
 * REFERENCE:
 *      STRANG
 * RELATED FUNCTIONS:
 *      Solve2x2, Solve3x3,
 */
{
  /* solve 3x2 linear system using Gauss-Jordan elimination with
   * full pivoting.  Input columns not modified.
   * returns 0: rank 0, 1: rank 1, 2: rank 2
   *         *err = signed distance from (d0,d1,d2) to plane
   *                through origin with normal col0 X col1.
   */
  int i;
  real x, y;
  vec3 c0, c1;

  *x_addr = *y_addr = *pivot_ratio = 0.0;
  *err_addr = kMaxReal;
  i = 0;
  x = fabs(col0[0]);
  y = fabs(col0[1]); if (y>x) {x = y; i = 1;}
  y = fabs(col0[2]); if (y>x) {x = y; i = 2;}
  y = fabs(col1[0]); if (y>x) {x = y; i = 3;}
  y = fabs(col1[1]); if (y>x) {x = y; i = 4;}
  y = fabs(col1[2]); if (y>x) {x = y; i = 5;}
  if (x == 0.0) return 0;
  *pivot_ratio = fabs(x);
  if (i >= 3) {
    /* swap columns */
    real* ptr = x_addr; x_addr = y_addr; y_addr = ptr;
    c0[0] = col1[0];
    c0[1] = col1[1];
    c0[2] = col1[2];

    c1[0] = col0[0];
    c1[1] = col0[1];
    c1[2] = col0[2];
  }
  else {
    c0[0] = col0[0];
    c0[1] = col0[1];
    c0[2] = col0[2];
    
    c1[0] = col1[0];
    c1[1] = col1[1];
    c1[2] = col1[2];
  }

  switch((i%=3)) {
  case 1: /* swap rows 0 and 1*/
    x=c0[1];c0[1]=c0[0];c0[0]=x;
    x=c1[1];c1[1]=c1[0];c1[0]=x;
    x=d1;d1=d0;d0=x;
    break;
  case 2: /* swap rows 0 and 2*/
    x=c0[2];c0[2]=c0[0];c0[0]=x;
    x=c1[2];c1[2]=c1[0];c1[0]=x;
    x=d2;d2=d0;d0=x;
    break;
  }

  c1[0] /= c0[0]; d0 /= c0[0];
  x = -c0[1]; if (x != 0.0) {c1[1] += x*c1[0]; d1 += x*d0;}
  x = -c0[2]; if (x != 0.0) {c1[2] += x*c1[0]; d2 += x*d0;}

  if (fabs(c1[1]) > fabs(c1[2])) {
    if (fabs(c1[1]) > *pivot_ratio)
      *pivot_ratio /= fabs(c1[1]); 
    else 
      *pivot_ratio = fabs(c1[1])/ *pivot_ratio;
    d1 /= c1[1];
    x = -c1[0]; if (x != 0.0) d0 += x*d1;
    x = -c1[2]; if (x != 0.0) d2 += x*d1;
    *x_addr = d0;
    *y_addr = d1;
    *err_addr = d2;
  }
  else if (c1[2] == 0.0) 
    return 1; /* 3x2 matrix has rank = 1 */
  else {
    if (fabs(c1[2]) > *pivot_ratio)
      *pivot_ratio /= fabs(c1[2]); 
    else 
      *pivot_ratio = fabs(c1[2])/ *pivot_ratio;
    d2 /= c1[2];
    x = -c1[0]; if (x != 0.0) d0 += x*d2;
    x = -c1[1]; if (x != 0.0) d1 += x*d2;
    *x_addr = d0;
    *err_addr = d1;
    *y_addr = d2;
  }

  return 2;
}


int
Solve3x3(const real row0[3], const real row1[3], const real row2[3],
                real d0, real d1, real d2,
                real* x_addr, real* y_addr, real* z_addr,
                real* pivot_ratio)
/* Solve a 3x3 system of linear equations
 *
 * INPUT:
 *   row0, row1, row2
 *      arrays of 3 reals
 *   d0, d1, d2
 *      right hand column of system
 *   x_addr, y_addr, x_addr
 *      addresses of reals
 *   pivot_ratio
 *      address of real
 * OUTPUT:
 *   ON_Solve3x3() returns rank (0,1,2,3)
 *   If ON_Solve3x3() is successful (return code 3), then
 *   the solution is returned in {*x_addr, *y_addr, *z_addr} and
 *   *pivot_ratio = min(|pivots|)/max(|pivots|).
 *
 *   WARNING: If the pivot ratio is small, then the matrix may
 *   be singular or ill conditioned.  You should test the results
 *   before you use them.
 *
 * COMMENTS:
 *      The system of 3 equations and 3 unknowns (x,y,z),
 *              x*row0[0] + y*row0[1] + z*row0[2] = d0
 *              x*row1[0] + y*row1[1] + z*row1[2] = d1
 *              x*row2[0] + y*row2[1] + z*row2[2] = d2,
 *      is solved using Gauss-Jordan elimination
 *      with full pivoting.
 * EXAMPLE:
 *      // Find the intersection of 3 3D planes where
 *      // P0, P1, P2 are points on the planes and
 *      // N0, N1, N2 are normals to the planes.  The normals
 *      // do not have to have unit length.
 *      real N0[3], N1[3], N2[3], P0[3], P1[3], P2[3], Q[3], pivot_ratio;
 *      rc = ON_Solve3x3(N0, N1, N2, 
 *                       N0[0]*P0[0] + N0[1]*P0[1] + N0[2]*P0[2],
 *                       N1[0]*P1[0] + N1[1]*P1[1] + N1[2]*P1[2],
 *                       N2[0]*P2[0] + N2[1]*P2[1] + N2[2]*P2[2],
 *                       &Q[0], &Q[1], &Q[2], &pivot_ratio);
 *      switch(rc) {
 *      case  0: // {x,y,z} = intersection point (probably)
 *        if (pivot_ratio < 0.001) {
 *          // small pivot ratio - test returned answer
 *          // compute err0 = N0 o (Q - P0), err1 = N1 o (Q - P1) 
 *          // and err2 = N2 o (Q - P2)
 *          // and determine if the errN terms are small enough.
 *        }
 *        break;
 *      case -1: // all normals are zero!
 *        break;
 *      case -2: // all normals are parallel
 *        break;
 *      case -3: // two of the three normals are parallel
 *        break;
 *      }
 *
 * REFERENCE:
 *      STRANG
 *
 * RELATED FUNCTIONS:
 *      Solve3x2(), Solve2x2
 */
{
  /* Solve a 3x3 linear system using Gauss-Jordan elimination 
   * with full pivoting.
   */
  int i, j;
  real* p0;
  real* p1;
  real* p2;
  real x, y, workarray[12], maxpiv, minpiv;

  const int sizeof_row = 3*sizeof(row0[0]);

  *pivot_ratio = *x_addr = *y_addr = *z_addr = 0.0;
  x = fabs(row0[0]); i=j=0;
  y = fabs(row0[1]); if (y>x) {x=y;j=1;}
  y = fabs(row0[2]); if (y>x) {x=y;j=2;}
  y = fabs(row1[0]); if (y>x) {x=y;i=1;j=0;}
  y = fabs(row1[1]); if (y>x) {x=y;i=1;j=1;}
  y = fabs(row1[2]); if (y>x) {x=y;i=1;j=2;}
  y = fabs(row2[0]); if (y>x) {x=y;i=2;j=0;}
  y = fabs(row2[1]); if (y>x) {x=y;i=2;j=1;}
  y = fabs(row2[2]); if (y>x) {x=y;i=2;j=2;}
  if (x == 0.0) 
    return 0;
  maxpiv = minpiv = fabs(x);
  p0 = workarray;
  switch(i) {
  case 1: /* swap rows 0 and 1 */
    memcpy(p0,row1,sizeof_row); p0[3] = d1; p0 += 4;
    memcpy(p0,row0,sizeof_row); p0[3] = d0; p0 += 4;
    memcpy(p0,row2,sizeof_row); p0[3] = d2;
    break;
  case 2: /* swap rows 0 and 2 */
    memcpy(p0,row2,sizeof_row); p0[3] = d2; p0 += 4;
    memcpy(p0,row1,sizeof_row); p0[3] = d1; p0 += 4;
    memcpy(p0,row0,sizeof_row); p0[3] = d0;
    break;
  default:
    memcpy(p0,row0,sizeof_row); p0[3] = d0; p0 += 4;
    memcpy(p0,row1,sizeof_row); p0[3] = d1; p0 += 4;
    memcpy(p0,row2,sizeof_row); p0[3] = d2;
    break;
  }
  switch(j) {
  case 1: /* swap columns 0 and 1 */
    p0 = x_addr; x_addr = y_addr; y_addr = p0;
    p0 = &workarray[0]; 
    x = p0[0]; p0[0]=p0[1]; p0[1]=x; p0 += 4;
    x = p0[0]; p0[0]=p0[1]; p0[1]=x; p0 += 4;
    x = p0[0]; p0[0]=p0[1]; p0[1]=x;
    break;
  case 2: /* swap columns 0 and 2 */
    p0 = x_addr; x_addr = z_addr; z_addr = p0;
    p0 = &workarray[0]; 
    x = p0[0]; p0[0]=p0[2]; p0[2]=x; p0 += 4;
    x = p0[0]; p0[0]=p0[2]; p0[2]=x; p0 += 4;
    x = p0[0]; p0[0]=p0[2]; p0[2]=x;
    break;
  }

  x = 1.0/workarray[0];
  /* debugger set workarray[0] = 1 */
  p0 = p1 = workarray + 1;
  *p1++ *= x; *p1++ *= x; *p1++ *= x;
  x = -(*p1++);
  /* debugger set workarray[4] = 0 */
  if (x == 0.0) 
    p1 += 3;
  else 
    {*p1++ += x*(*p0++); *p1++ += x*(*p0++); *p1++ += x*(*p0); p0 -= 2;}
  x = -(*p1++);
  /* debugger set workarray[8] = 0 */
  if (x != 0.0)
    {*p1++ += x*(*p0++); *p1++ += x*(*p0++); *p1++ += x*(*p0); p0 -= 2;}

  x = fabs(workarray[ 5]);i=j=0;
  y = fabs(workarray[ 6]);if (y>x) {x=y;j=1;}
  y = fabs(workarray[ 9]);if (y>x) {x=y;i=1;j=0;}
  y = fabs(workarray[10]);if (y>x) {x=y;i=j=1;}
  if (x == 0.0) 
    return 1; // rank = 1;
  y = fabs(x);
  if (y > maxpiv) maxpiv = y; else if (y < minpiv) minpiv = y;
  if (j) {
    /* swap columns 1 and 2 */
    p0 = workarray+1;
    p1 = p0+1;
    x = *p0; *p0 = *p1; *p1 = x; p0 += 4; p1 += 4;
    x = *p0; *p0 = *p1; *p1 = x; p0 += 4; p1 += 4;
    x = *p0; *p0 = *p1; *p1 = x; p0 += 4; p1 += 4;
    p0 = y_addr; y_addr = z_addr; z_addr = p0;
  }

  if (i) {
    /* pivot is in row 2 */
    p0 = workarray+1;
    p1 = p0 + 8;
    p2 = p0 + 4;
  }
  else {
    /* pivot is in row 1 */
    p0 = workarray+1;
    p1 = p0 + 4;
    p2 = p0 + 8;
  }

  /* debugger set workarray[5+4*i] = 1 */
  x = 1.0/(*p1++); *p1++ *= x; *p1 *= x; p1--;
  x = -(*p0++);
  /* debugger set p0[-1] = 0 */
  if (x != 0.0) {*p0++ += x*(*p1++); *p0 += x*(*p1); p0--; p1--;}
  x = -(*p2++);
  /* debugger set p2[-1] = 0 */
  if (x != 0.0) {*p2++ += x*(*p1++); *p2 += x*(*p1); p2--; p1--;}
  x = *p2++;
  if (x == 0.0) 
    return 2; // rank = 2;
  y = fabs(x);
  if (y > maxpiv) maxpiv = y; else if (y < minpiv) minpiv = y;
  /* debugger set p2[-1] = 1 */
  *p2 /= x;
  x = -(*p1++);  if (x != 0.0) *p1 += x*(*p2);
  /* debugger set p1[-1] = 0 */
  x = -(*p0++);  if (x != 0.0) *p0 += x*(*p2);
  /* debugger set p0[-1] = 0 */
  *x_addr = workarray[3];
  if (i) {
    *y_addr = workarray[11];
    *z_addr = workarray[7];
  }
  else {
    *y_addr = workarray[7];
    *z_addr = workarray[11];
  }
  *pivot_ratio = minpiv/maxpiv;
  return 3;
}

}; // namespace graphics