#include "graphics/simple_curve.h"



#include <math.h>
#include "graphics/epsilon.h"


namespace graphics {

real bezier(const real& p1, const real& p2, const real& p3, const real& p4, real t)
{
    real a1 = (1 - t) * (1 - t) * (1 - t);
    real a2 = 3 * (1 - t) * (1 - t) * t;
    real a3 = 3 * (1 - t) * t * t;
    real a4 = t * t * t;

    return a1 * p1 + a2 * p2 + a3 * p3 + a4 * p4;
}

real bezier(int degree, const vector<real>& points, real t)
{
    vector<real> temp = points;

    for (int i = 1; i <= degree;++i)
	for (int j = 0; j <= degree-i; ++j) 
	    temp[j] = (1.0 - t) * temp[j] + t * temp[j+1];

    return temp[0];
}

real hermite(const real& p1, const real& d1, const real& p2, const real& d2, real t)
{
    return bezier(p1, p1 + d1 / 3, p2 - d2 / 3, p2, t);
}



vec2 bezier(const vec2& p1, const vec2& p2, const vec2& p3, const vec2& p4, real t)
{
    real a1 = (1 - t) * (1 - t) * (1 - t);
    real a2 = 3 * (1 - t) * (1 - t) * t;
    real a3 = 3 * (1 - t) * t * t;
    real a4 = t * t * t;

    return a1 * p1 + a2 * p2 + a3 * p3 + a4 * p4;
}

vec2 bezier(int degree, const vector<vec2>& points, real t)
{
    vector<vec2> temp = points;

    for (int i = 1; i <= degree;++i)
	for (int j = 0; j <= degree-i; ++j) 
	    temp[j] = (1.0 - t) * temp[j] + t * temp[j+1];

    return temp[0];
}

vec2 hermite(const vec2& p1, const vec2& d1, const vec2& p2, const vec2& d2, real t)
{
    return bezier(p1, p1 + d1 / 3, p2 - d2 / 3, p2, t);
}



vec3 bezier(const vec3& p1, const vec3& p2, const vec3& p3, const vec3& p4, real t)
{
    real a1 = (1 - t) * (1 - t) * (1 - t);
    real a2 = 3 * (1 - t) * (1 - t) * t;
    real a3 = 3 * (1 - t) * t * t;
    real a4 = t * t * t;

    return a1 * p1 + a2 * p2 + a3 * p3 + a4 * p4;
}

vec3 bezier(int degree, const vector<vec3>& points, real t)
{
    vector<vec3> temp = points;

    for (int i = 1; i <= degree;++i)
	for (int j = 0; j <= degree-i; ++j) 
	    temp[j] = (1.0 - t) * temp[j] + t * temp[j+1];

    return temp[0];
}

vec3 hermite(const vec3& p1, const vec3& d1, const vec3& p2, const vec3& d2, real t)
{
    return bezier(p1, p1 + d1 / 3, p2 - d2 / 3, p2, t);
}



//| B-spline : Uniform

real B_spline(int i, int k, real t)
{
    if(k == 1) {
	if(i <= t && t < i + 1) return 1;
	else return 0;
    }

    return (B_spline(i, k - 1, t) * (t - i)
	    + B_spline(i + 1, k - 1, t) * (i + k - t)) / (k - 1);
}

//| B-spline : Non-uniform
real B_spline(real knot[], int i, int k, real t)
{
    if(k == 1) {
	if(knot[i] <= t && t < knot[i + 1]) return 1;
	else return 0;
    }

    real d1 = knot[i + k - 1] - knot[i];
    if(fabs(d1) < epsilon) d1 = epsilon;

    real a1 = B_spline(knot, i, k - 1, t) * (t - knot[i]) / d1;

    real d2 = knot[i + k] - knot[i + 1];
    if(fabs(d2) < epsilon) d2 = epsilon;

    real a2 = B_spline(knot, i + 1, k - 1, t) * (knot[i + k] - t) / d2;

    return a1 + a2;
}

}; 	// namespace graphics
