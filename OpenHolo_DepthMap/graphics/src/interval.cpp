#include "graphics/interval.h"
/*|~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*/
/*| Interval Analysis,                                                       */
/*| Refer to the paper, John M. Snyder, "Interval Analysis for               */
/*| computer graphics", SIGRAPH`92 Conference Proceeding, ACM,               */
/*| pp121-130, 1992, and some books on interval analysis.                    */
/*| You should keep it in your mind that this was modified to                */
/*| be adjusted into our application.                                        */
/*| Last modified Dec. 2. 1999.                                              */
/*|__________________________________________________________________________*/




#include "graphics/log.h"
#include "graphics/minmax.h"


namespace graphics {

//| testing inclusion of zero
bool is_zero(const interval& a)
{
    return a.l < 0 && a.u > 0;
}

//| __boolean
interval operator| (const interval& a, const interval& b)
{
    if(a.valid() && b.valid())
	return interval(min(a.l, b.l), max(a.u, b.u));
    else
	return interval(0.0, 0.0);
}

interval operator& (const interval& a, const interval& b)
{
    if(a.valid() && b.valid())
	return interval(max(a.l, b.l), min(a.u, b.u));
    else
	return interval(0.0, 0.0);
}

//| relational operator for valid intervals
interval operator < (const interval& a, const interval& b)
{
    if(b.u <= a.l)
	return interval(0.0, 0.0);
    else if(a.u < b.l)
	return interval(1.0, 1.0);
    else
	return interval(0.0, 1.0);
}

interval operator < (const interval& a, real b)
{
    if(user_epsilon == epsilon)
    {
	if(a.u < b)
	    return interval(1.0, 1.0);
	else if(b > a.l && b <= a.u)
	    return interval(0.0, 1.0);
	else
	    return interval(0.0, 0.0);
    } else {
	interval temp(b - user_epsilon, b + user_epsilon);
	return (a < temp);
    }
}

interval operator < (real a, const interval& b)
{
    if(user_epsilon == epsilon)
    {
	if(a < b.l)
	    return interval(1.0, 1.0);
	else if(a >= b.l && a < b.u)
	    return interval(0.0, 1.0);
	else
	    return interval(0.0, 0.0);
    } else {
	interval temp(a - user_epsilon, a + user_epsilon);
	return (temp < b);
    }
}

interval operator > (const interval& a, const interval& b)
{
    if(a.u <= b.l)
	return interval(0.0, 0.0);
    else if(b.u < a.l)
	return interval(1.0, 1.0);
    else
	return interval(0.0, 1.0);
}

interval operator > (const interval& a, real b)
{
    if(user_epsilon == epsilon)
    {
	if(a.l > b)
	    return interval(1.0, 1.0);
	else if(a.l <= b && a.u > b)
	    return interval(0.0, 1.0);
	else
	    return interval(0.0, 0.0);
    } else {
	interval temp(b - user_epsilon, b + user_epsilon);
	return (a > temp);
    }
}

interval operator > (real a, const interval& b)
{
    if(user_epsilon == epsilon)
    {
	if(a > b.u)
	    return interval(1.0, 1.0);
	else if(a > b.l && a <= b.u)
	    return interval(0.0, 1.0);
	else
	    return interval(0.0, 0.0);
    } else {
	interval temp(a - user_epsilon, a + user_epsilon);
	return (temp > b);
    }
}

interval operator <= (const interval& a, const interval& b)
{
    if(b.u < a.l)
	return interval(0.0, 0.0);
    else if(a.u <= b.l)
	return interval(1.0, 1.0);
    else
	return interval(0.0, 1.0);
}

interval operator <= (const interval& a, real b)
{
    if(user_epsilon == epsilon)
    {
	if(a.u <= b)
	    return interval(1.0, 1.0);
	else if(b >= a.l && b < a.u)
	    return interval(0.0, 1.0);
	else
	    return interval(0.0, 0.0);
    } else {
	interval temp(b - user_epsilon, b + user_epsilon);
	return (a <= temp);
    }
}

interval operator <= (real a, const interval& b)
{
    if(user_epsilon == epsilon)
    {
	if(a <= b.l)
	    return interval(1.0, 1.0);
	else if(a > b.l && a <= b.u)
	    return interval(0.0, 1.0);
	else
	    return interval(0.0, 0.0);
    } else {
	interval temp(a - user_epsilon, a + user_epsilon);
	return (temp <= b);
    }
}

interval operator >= (const interval& a, const interval& b)
{
    if(a.u < b.l)
	return interval(0.0, 0.0);
    else if(b.u <= a.l)
	return interval(1.0, 1.0);
    else
	return interval(0.0, 1.0);
}

interval operator >= (const interval& a, real b)
{
    if(user_epsilon == epsilon)
    {
	if(a.l >= b)
	    return interval(1.0, 1.0);
	else if(a.l < b && a.u >= b)
	    return interval(0.0, 1.0);
	else
	    return interval(0.0, 0.0);
    } else {
	interval temp(b - user_epsilon, b + user_epsilon);
	return (a >= temp);
    }
}

interval operator >= (real a, const interval& b)
{
    if(user_epsilon == epsilon)
    {
	if(a >= b.u)
	    return interval(1.0, 1.0);
	else if(a >= b.l && a < b.u)
	    return interval(0.0, 1.0);
	else
	    return interval(0.0, 0.0);
    } else {
	interval temp(a - user_epsilon, a + user_epsilon);
	return (temp >= b);
    }
}

//| logical operators for valid intervals
interval operator == (const interval& a, const interval& b)
{ 
    interval one(1.0, 1.0);
    interval zero(0.0, 0.0);

    interval t1 = (a > b);
    interval t2 = (a < b);
    interval t3 = ((t1.l==1.0)&&(t1.u==1.0))?one:zero;
    interval t4 = ((t2.l==1.0)&&(t2.u==1.0))?one:zero;

    if(fabs(a.l - b.l) <= user_epsilon && fabs(a.u - b.u) <= user_epsilon)
        return interval(1.0, 1.0);
    else if(__boolean(t3) || __boolean(t4))
	return interval(0.0, 0.0);
    else 
	return interval(0.0, 1.0);
}


interval operator == (real a, const interval& b)
{ 
    if(user_epsilon == epsilon)
    {
	if(fabs(b.l - a) <= user_epsilon && fabs(b.u - a) <= user_epsilon)
	    return interval(1.0, 1.0);
	else if(b.in(a))
	    return interval(0.0, 1.0);
	else return interval(0.0, 0.0);
    } else {
	interval temp(a-user_epsilon, a+user_epsilon);
	return (temp == b);
    }
}

interval operator == (const interval& a, real b)
{ 
    if(user_epsilon == epsilon)
    {
	if(fabs(a.l - b) <= user_epsilon && fabs(a.u - b) <= user_epsilon)
	    return interval(1.0, 1.0);
	else if(a.in(b))
	    return interval(0.0, 1.0);
	else return interval(0.0, 0.0);
    } else {
	interval temp(b-user_epsilon, b+user_epsilon);
	return (a == temp);
    }
}


interval is_equal(const interval& a, const interval& b)
{ 
    interval one(1.0, 1.0);
    interval zero(0.0, 0.0);

    interval t1 = (a > b);
    interval t2 = (a < b);
    interval t3 = ((t1.l==1.0)&&(t1.u==1.0))?one:zero;
    interval t4 = ((t2.l==1.0)&&(t2.u==1.0))?one:zero;

    if(fabs(a.l - b.l) <= epsilon && fabs(a.u - b.u) <= epsilon)
        return interval(1.0, 1.0);
    else if(__boolean(t3) || __boolean(t4))
	return interval(0.0, 0.0);
    else 
	return interval(0.0, 1.0);
}

interval operator && (const interval& a, const interval& b)
{
    interval zero(0.0, 0.0);
    interval one(1.0, 1.0);
    interval ind(0.0, 1.0); //| indeterminable
    interval t1 = is_equal(a, zero);
    interval t2 = is_equal(b, zero);
    interval t3 = is_equal(a, one );
    interval t4 = is_equal(b, one );

    if(__boolean(t1) || __boolean(t2))
	return zero;
    if(__boolean(t3) && __boolean(t4))
	return one;
    return ind;
}

interval operator || (const interval& a, const interval& b)
{
    interval zero(0.0, 0.0);
    interval one(1.0, 1.0);
    interval t1 = is_equal(a,one);
    interval t2 = is_equal(b,one);
    interval t3 = is_equal(a,zero);
    interval t4 = is_equal(b,zero);

    if(__boolean(t1))
	return interval(1.0, 1.0);
    else if (__boolean(t2))
	return interval(1.0, 1.0);
    else if (__boolean(t3) && __boolean(t4))
	return interval(0.0, 0.0);
    else
	return interval(0.0, 1.0);
}

//| correct : 1, undeterminable : -1, rejectable : 0
int determinable(const interval& a)
{
    if(fabs(a.l - 1.0) <= epsilon && fabs(a.u - 1.0) <= epsilon)
	return 1;
    else if(fabs(a.l - 0.0) <= epsilon && fabs(a.u - 1.0) <= epsilon)
	return -1;
    else
	return 0;
}

bool __boolean(const interval& a)
{
    if(fabs(a.l - 1.0) <= epsilon && fabs(a.u - 1.0) <= epsilon)
	return true;
    else
	return false;
} 

//| arithmatic
interval operator+ (const interval& a, const interval& b)
{
    return interval(a.l + b.l, a.u + b.u);
}

interval operator+= (interval& a, const interval& b)
{
    return interval(a.l += b.l, a.u += b.u);
}

interval operator- (const interval& a)
{
    return interval(-a.u, -a.l);
}

interval operator- (const interval& a, const interval& b)
{
    return interval(a.l - b.u, a.u - b.l);
}

interval operator* (real a, const interval& b)
{
    real aa = a * b.l;
    real bb = a * b.u;
    return interval(min(aa, bb), max(aa, bb));
}

interval operator* (const interval& a, real b)
{
    real aa = a.l * b;
    real bb = a.u * b;
    return interval(min(aa, bb), max(aa, bb));
}

interval operator* (const interval& a, const interval& b)
{
    real aa = a.l * b.l;
    real bb = a.l * b.u;
    real cc = a.u * b.l;
    real dd = a.u * b.u;
    return interval(min(aa, bb, cc, dd), max(aa, bb, cc, dd));
}

interval operator/ (const interval& a, real b)
{
    if(fabs(b) < 1e-20) fatal("interval I/r: div by zero, %g", b);
    real aa = a.l / b;
    real bb = a.u / b;
    return interval(min(aa, bb), max(aa, bb));
}

interval operator/ (const interval& a, const interval& b)
{
    if(is_zero(b)) fatal("interval I/I: div by zero");
    real aa = a.l / b.l;
    real bb = a.l / b.u;
    real cc = a.u / b.l;
    real dd = a.u / b.u;
    return interval(min(aa, bb, cc, dd), max(aa, bb, cc, dd));
}

//| functions
interval square(const interval& a)
{
    if(is_zero(a))
	return interval(0, max(a.l * a.l, a.u * a.u));
    else
	return interval(min(a.l * a.l, a.u * a.u), max(a.l * a.l, a.u * a.u));
}

interval square_root(const interval& a)
{
    if(a.l < 0) fatal("interval:sqrt: negative");
    interval ret(sqrt(a.l), sqrt(a.u));

    return ret;
}

interval cosine(const interval& a)
{
    int aa = (int) ceil(a.l / M_PI);
    real bb = a.u / M_PI;
    real ca = cos(a.l);
    real cb = cos(a.u);

    if(1 + aa <= bb)
	return interval(-1, 1);
    else if(aa <= bb && aa % 2 == 1)
	return interval(-1, max(ca, cb));
    else if(aa <= bb && aa % 2 == 0)
	return interval(min(ca, cb), 1);
    else
	return interval(min(ca, cb), max(ca, cb));
}

interval sine(const interval& a)
{
    return cosine(a - M_PI / 2); 
}

interval arc_tangent(const interval& a, const interval& b)
{
    if(is_zero(a) && is_zero(b)) fatal("Iatan2: interval has zero");

    real t1 = atan2(a.l, b.l);
    real t2 = atan2(a.l, b.u);
    real t3 = atan2(a.u, b.l);
    real t4 = atan2(a.u, b.u);

    return interval(min(t1, t2, t3, t4), max(t1, t2, t3, t4));
}


};