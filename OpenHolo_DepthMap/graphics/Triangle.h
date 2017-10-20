#ifndef __Triangle_h
#define __Triangle_h
#include <graphics/vec.h>

namespace graphics {

struct Triangle {
	vec3 p1, p2, p3;
	vec3 normal;

	Triangle();
	Triangle(const Triangle& c);
	Triangle(const vec3& a, const vec3& b, const vec3& c);
	Triangle(const vec3& a, const vec3& b, const vec3& c, const vec3& n);
};

}

#endif