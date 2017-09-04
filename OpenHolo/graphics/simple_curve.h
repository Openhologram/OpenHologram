#ifndef __simple_curve_h
#define __simple_curve_h

//|
//| A simple spline function evaluation only
//| for a job not requiring a good computer graphics.
//|
#include "graphics/real.h"
#include "graphics/vec.h"
#include "graphics/vector.h"

namespace graphics {




real bezier(const real& p1, const real& p2, const real& p3, const real& p4, real t)
;

real bezier(int degree, const vector<real>& points, real t)
;

real hermite(const real& p1, const real& d1, const real& p2, const real& d2, real t)
;



vec2 bezier(const vec2& p1, const vec2& p2, const vec2& p3, const vec2& p4, real t)
;

vec2 bezier(int degree, const vector<vec2>& points, real t)
;

vec2 hermite(const vec2& p1, const vec2& d1, const vec2& p2, const vec2& d2, real t)
;



vec3 bezier(const vec3& p1, const vec3& p2, const vec3& p3, const vec3& p4, real t)
;

vec3 bezier(int degree, const vector<vec3>& points, real t)
;

vec3 hermite(const vec3& p1, const vec3& d1, const vec3& p2, const vec3& d2, real t)
;



//| B-spline : Uniform

real B_spline(int i, int k, real t)
;

//| B-spline : Non-uniform
real B_spline(real knot[], int i, int k, real t)
;

};	// namespace graphics
#endif

