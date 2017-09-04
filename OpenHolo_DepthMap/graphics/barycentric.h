#ifndef __barycentric_h
#define __barycentric_h

#include	"graphics/vec.h"

namespace graphics {

vec3 barycentric_coordinate(const vec2& a, const vec2& b, const  vec2& c, const vec2& x);

vec4 barycentric_coordinate(const vec2& a, const vec2& b, const  vec2& c, const  vec2& d, const vec2& x);

};

#endif