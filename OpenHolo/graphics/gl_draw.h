#ifndef __gl_draw_h
#define __gl_draw_h

#include "graphics/sys.h"
#include "graphics/real.h"
#include "graphics/quater.h"
#include "graphics/misc.h"
#include "graphics/gl.h"
#include "graphics/frame.h"



namespace graphics {



void draw_box(const vec3& c, real size);

void draw_box(const box3 &box);
void draw_wire_box(const box3 &box);
void draw_wire_box(const vec3& c, real size);


}; // namespace

#endif
