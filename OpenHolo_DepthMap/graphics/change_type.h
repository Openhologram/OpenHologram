#ifndef __change_type_h
#define __change_type_h


#include "graphics/vec.h"
#include "graphics/vector.h"

namespace graphics {


struct vec2vector {

    vector<real> result;

    vec2vector(const vec3& input):result(3) 
    {
	result[0] = input[0];
	result[1] = input[1];
	result[2] = input[2];
    }

    vec2vector(const vec2& input):result(2)
    {
	result[0] = input[0];
	result[1] = input[1];
    }

    vec2vector(const vec4& input):result(4)
    {
	result[0] = input[0];
	result[1] = input[1];
	result[2] = input[2];
	result[3] = input[3];
    }

    vec2vector(const ivec3& input):result(3) 
    {
	result[0] = input[0];
	result[1] = input[1];
	result[2] = input[2];
    }

    vec2vector(const ivec2& input):result(2)
    {
	result[0] = input[0];
	result[1] = input[1];
    }

    vec2vector(const ivec4& input):result(4)
    {
	result[0] = input[0];
	result[1] = input[1];
	result[2] = input[2];
	result[3] = input[3];
    }

    const vector<real>& get() const { return result; }
    vector<real>& get() { return result; }
};

struct vector2vec2 {
    vec2 result;

    vector2vec2(const vector<real>& input) : result(0) 
    { 
	if (input.size() >= 2) {
	    result[0] = input[0];
	    result[1] = input[1];
	}
    }

    const vec2& get() const { return result; }
    vec2& get() { return result; }
};

struct vector2vec3 {
    vec3 result;

    vector2vec3(const vector<real>& input) : result(0) 
    { 
	if (input.size() >= 3) {
	    result[0] = input[0];
	    result[1] = input[1];
	    result[2] = input[2];
	}
    }

    const vec3& get() const { return result; }
    vec3& get() { return result; }
};

struct vector2vec4 {

    vec4 result;

    vector2vec4(const vector<real>& input) : result(0) 
    { 
	if (input.size() >= 4) {
	    result[0] = input[0];
	    result[1] = input[1];
	    result[2] = input[2];
	    result[3] = input[3];
	}
    }

    const vec4& get() const { return result; }
    vec4& get() { return result; }
};

}; // namespace graphics
#endif
