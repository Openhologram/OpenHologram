#ifndef __interval_h
#define __interval_h
/*|~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*/
/*| Interval Analysis,                                                       */
/*| Refer to the paper, John M. Snyder, "Interval Analysis for               */
/*| computer graphics", SIGRAPH`92 Conference Proceeding, ACM,               */
/*| pp121-130, 1992, and some books on interval analysis.                    */
/*| You should keep it in your mind that this was modified to                */
/*| be adjusted into our application.                                        */
/*| Last modified Dec. 2. 1999.                                              */
/*|__________________________________________________________________________*/


#include "graphics/sys.h"
#include "graphics/real.h"

#include <stdio.h>
#include <math.h>
#include "graphics/epsilon.h"

namespace graphics {


struct interval {
    real l, u;

    int constant;

    inline interval(): constant(0), l(0.0), u(0.0) {  }
    inline interval(const interval& a) : l(a.l), u(a.u), constant(a.constant) {  }
    inline interval(real ll, real uu): constant(0), l(ll), u(uu) {  }
    inline interval(real a): l(a), u(a), constant(0) {  }

    //| explicit constructor
    inline void init(real a) { l = u = a; }
    inline void set_const() { constant = 1; }
    inline void reset_const() { constant = 0; }
    inline bool  is_const() const { if(constant) return true; else return false; }

    inline interval operator= (real a) { return l = u = a; }
    inline interval operator= (const interval& a) { l = a.l; u = a.u; constant = a.constant; return a; }

    inline bool in(real a) const { return l <= a && a <= u; }
    inline bool in(const interval& a) const { return l <= a.l && a.u <= u; }
    inline bool proper_in(real a) const { return l < a && a < u; }
    inline bool proper_in(const interval& a) const { return l < a.l && a.u < u; }
    inline void ins(real a) { if(a < l) l = a; else if(a > u) u = a; }

    //| width of this interval
    inline real d() const { return u - l; }

    //| center of this interval
    inline real c() const { return (l + u) / 2; }
    
    //| check if this interval is valid
    inline bool  valid() const { return (u >= l)? true : false; }

    //| if this interval is invalid, validate it.
    inline void validate() { real t; if(u < l) {t = u; u = l; l = t;}}
};

//| testing inclusion of zero
bool is_zero(const interval& a)
;

//| __boolean
interval operator| (const interval& a, const interval& b)
;

interval operator& (const interval& a, const interval& b)
;

//| relational operator for valid intervals
interval operator < (const interval& a, const interval& b)
;

interval operator < (const interval& a, real b)
;

interval operator < (real a, const interval& b)
;

interval operator > (const interval& a, const interval& b)
;

interval operator > (const interval& a, real b)
;

interval operator > (real a, const interval& b)
;

interval operator <= (const interval& a, const interval& b)
;

interval operator <= (const interval& a, real b)
;

interval operator <= (real a, const interval& b)
;

interval operator >= (const interval& a, const interval& b)
;

interval operator >= (const interval& a, real b)
;

interval operator >= (real a, const interval& b)
;

//| logical operators for valid intervals
interval operator == (const interval& a, const interval& b)
;


interval operator == (real a, const interval& b)
;

interval operator == (const interval& a, real b)
;


interval is_equal(const interval& a, const interval& b)
;

interval operator && (const interval& a, const interval& b)
;

interval operator || (const interval& a, const interval& b)
;

//| correct : 1, undeterminable : -1, rejectable : 0
int determinable(const interval& a)
;

bool __boolean(const interval& a)
; 

//| arithmatic
interval operator+ (const interval& a, const interval& b)
;

interval operator+= (interval& a, const interval& b)
;

interval operator- (const interval& a)
;

interval operator- (const interval& a, const interval& b)
;

interval operator* (real a, const interval& b)
;

interval operator* (const interval& a, real b)
;

interval operator* (const interval& a, const interval& b)
;

interval operator/ (const interval& a, real b)
;

interval operator/ (const interval& a, const interval& b)
;

//| functions
interval square(const interval& a)
;

interval square_root(const interval& a)
;

interval cosine(const interval& a)
;

interval sine(const interval& a)
;

interval arc_tangent(const interval& a, const interval& b)
;

};

#endif
