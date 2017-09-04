
#ifndef	    __vec_abs_h
#define	    __vec_abs_h

#include "graphics/ivec.h"
#include "graphics/vec.h"
#include <math.h>
#include "graphics/epsilon.h"

namespace graphics {

inline vec2 abs(const vec2& val)
{
    return vec2(fabs(val[0]), fabs(val[1]));
}

inline vec3 abs(const vec3& val)
{
    return vec3(fabs(val[0]), fabs(val[1]), fabs(val[2]));
}

inline vec4 abs(const vec4& val)
{
    return vec4(fabs(val[0]), fabs(val[1]), fabs(val[2]), fabs(val[3]));
}

inline ivec2 abs(const ivec2& val)
{
    return ivec2(fabs(val[0]), fabs(val[1]));
}

inline ivec3 abs(const ivec3& val)
{
    return ivec3(fabs(val[0]), fabs(val[1]), fabs(val[2]));
}

inline ivec4 abs(const ivec4& val)
{
    return ivec4(fabs(val[0]), fabs(val[1]), fabs(val[2]), fabs(val[3]));
}


};  // name space graphics

#endif