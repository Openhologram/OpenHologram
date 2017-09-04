#include "graphics/Triangle.h"


namespace graphics {

Triangle::Triangle()
	: p1(0), p2(0), p3(0), normal(0,0,1)
{
}
Triangle::Triangle(const Triangle& c)
	: p1(c.p1), p2(c.p2), p3(c.p3), normal(c.normal)
{
}
Triangle::Triangle(const vec3& a, const vec3& b, const vec3& c)
	: p1(a), p2(b), p3(c)
{
	normal = cross((p2-p1), (p3-p2));
	if (apx_equal(norm(normal),0.0)) {
		normal = vec3(0,0,1);
	}
	normal = unit(normal);
}

Triangle::Triangle(const vec3& a, const vec3& b, const vec3& c, const vec3& n)
	: p1(a), p2(b), p3(c), normal(n)
{
}

};