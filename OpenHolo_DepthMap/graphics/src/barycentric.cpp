#include "graphics/barycentric.h"
#include "graphics/geom.h"

namespace graphics {

vec3 
barycentric_coordinate(const vec2& one, const vec2& two, const  vec2& three, const vec2& x)
{
	real det = (one[0] - three[0])*(two[1]-three[1]) - (two[0]-three[0])*(one[1]-three[1]);
	real r1 = ((two[1]-three[1])*(x[0]-three[0]) + (three[0]-two[0])*(x[1]-three[1]))/det;
	real r2 = ((three[1]-one[1])*(x[0]-three[0]) + (one[0]-three[0])*(x[1]-three[1]))/det;
	real r3 = 1.0 - r1 - r2;
	return vec3(r1, r2, r3);
}

vec4 barycentric_coordinate(const vec2& a, const vec2& b, const  vec2& c, const  vec2& d, const vec2& x)
{
	real tri_1 = tri_area(d,a,b);
	real tri_2 = tri_area(a,b,c);
	real tri_3 = tri_area(b,c,d);
	real tri_4 = tri_area(c,d,a);

	real tri_23 = tri_area(b,c,x);
	real tri_34 = tri_area(c,d,x);
	real tri_41 = tri_area(d,a,x);
	real tri_12 = tri_area(a,b,x);

	real s = tri_1*tri_23*tri_34;
	real t = tri_2*tri_34*tri_41;
	real u = tri_3*tri_41*tri_12;
	real v = tri_4*tri_12*tri_23;

	real sum = s+t+u+v;

	return vec4(s/sum,t/sum,u/sum,v/sum);
}
};