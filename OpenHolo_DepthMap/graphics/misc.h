#ifndef __misc_h
#define __misc_h

#include "graphics/sys.h"
#include "graphics/real.h"
#include <math.h>

namespace graphics {


inline real absolute(real val) 
{ 
    return fabs(val); 
}

inline int ceil(int n, int a)
{
    if(n % a)
	return (n / a + 1) * a;
    else
	return n;
}

inline int odd(int a)
{
    return (a % 2);
}

inline real degree(real rad)
{
    return rad * (180 / M_PI);
}

inline real radian(real deg)
{
    return deg * (M_PI / 180);
}


inline int my_floor(real a) 
{
    int truncated = (int) a;
    real mantila = a - truncated;
    if (mantila > 0.0) return truncated;
    return (truncated + 1);
}


void byte_copy(int n, char* a, char* b);

void byte_swap(int n, char* a, char* b);

}; // namespace graphics
#endif
