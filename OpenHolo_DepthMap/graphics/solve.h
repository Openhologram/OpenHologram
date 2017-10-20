#ifndef __solve_h
#define __solve_h

#include "graphics/sys.h"
#include "graphics/log.h"
#include "graphics/real.h"

namespace graphics {

int SolveQuadraticEquation( // solve a*X^2 + b*X + c = 0
        // returns 0: two distinct real roots (r0 < r1)
        //         1: one real root (r0 = r1)
        //         2: two complex conjugate roots (r0 +/- (r1)*sqrt(-1))
        //        -1: failure - a = 0, b != 0        (r0 = r1 = -c/b)
        //        -2: failure - a = 0, b  = 0 c != 0 (r0 = r1 = 0.0)
        //        -3: failure - a = 0, b  = 0 c  = 0 (r0 = r1 = 0.0)
       real, real, real, // a, b, c
       real*, real*        // roots r0 and r1 returned here
       );
/********************************************************
*							*
* This function determines the roots of a cubic		*
* equation.						*
* It takes as parameters a pointer to the four		*
* coefficient of the cubic equation (the c[3] is the	*
* coefficient of x3 and so on) and a pointer to the	*
* three element array in which the roots are to be	*
* placed.						*
* It outputs the number of roots found			*
*							*
********************************************************/
int solveCubic(real c[4], real s[3]);


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
solveQuartic(double c[5], double s[4]);

int SolveTriDiagonal( // solve TriDiagMatrix( a,b,c )*X = d
        int,               // dimension of d and X (>=1)
        int,               // number of equations (>=2)
        real*,           // a[n-1] = sub-diagonal (a is modified)
        const real*,     // b[n] = diagonal
        real*,           // c[n-1] = supra-diagonal
        const real*,     // d[n*dim]
        real*            // X[n*dim] = unknowns
        );

// returns rank - if rank != 2, system is under determined
// If rank = 2, then solution to 
//
//          a00*x0 + a01*x1 = b0, 
//          a10*x0 + a11*x1 = b1 
//
// is returned

int Solve2x2( 
        real, real,   // a00 a01 = first row of 2x2 matrix
        real, real,   // a10 a11 = second row of 2x2 matrix
        real, real,   // b0 b1
        real*, real*, // x0, x1 if not NULL, then solution is returned here
        real*           // if not NULL, then pivot_ratio returned here
        );

// Description:
//   Solves a system of 3 linear equations and 2 unknowns.
//
//          x*col0[0] + y*col1[0] = d0
//          x*col0[1] + y*col1[1] = d0
//          x*col0[2] + y*col1[2] = d0
//
// Parameters:
//   col0 - [in] coefficents for "x" unknown
//   col1 - [in] coefficents for "y" unknown
//   d0 - [in] constants
//   d1 - [in]
//   d2 - [in]
//   x - [out]
//   y - [out]
//   error - [out]
//   pivot_ratio - [out]
//
// Returns:
//   rank of the system.  
//   If rank != 2, system is under determined
//   If rank = 2, then the solution is
//
//         (*x)*[col0] + (*y)*[col1]
//         + (*error)*((col0 X col1)/|col0 X col1|)
//         = (d0,d1,d2).

int Solve3x2( 
        const real[3], // col0
        const real[3], // col1
        real,  // d0
        real,  // d1
        real,  // d2
        real*, // x
        real*, // y
        real*, // error
        real*  // pivot_ratio
        );

// returns rank - if rank != 3, system is under determined

int Solve3x3( 
        const real row0[3], 
        const real row1[3], 
        const real row2[3],
        real d0, real d1, real d2,
        real* x_addr, real* y_addr, real* z_addr,
        real* pivot_ratio
        );
}; // namespace graphics 

#endif
