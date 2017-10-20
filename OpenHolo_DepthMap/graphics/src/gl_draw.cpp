#include "graphics/gl_draw.h"
#include "graphics/gl.h"
#include "graphics/Camera.h"

namespace graphics {




void draw_box(const box3& box)
{
	vec3 c = (box.minimum + box.maximum)/2.0;
	vec3 t = box.maximum - c;

	vec3 ex = vec3(t[0], 0, 0);
	vec3 ey = vec3(0, t[1], 0);
	vec3 ez = vec3(0, 0, t[2]);

	real size = 1.0;

   
    glBegin(GL_QUADS);
    
	
	gl_normal(-unit(ex));
	gl_vertex(c + size*ex - size*ey - size*ez);
	gl_vertex(c + size*ex - size*ey + size*ez);
	gl_vertex(c + size*ex + size*ey + size*ez);
	gl_vertex(c + size*ex + size*ey - size*ez);
    
	gl_normal(unit(ex));
	gl_vertex(c + (-size)*ex - size*ey - size*ez);
	gl_vertex(c + (-size)*ex + size*ey - size*ez);
	gl_vertex(c + (-size)*ex + size*ey + size*ez);
	gl_vertex(c + (-size)*ex - size*ey + size*ez);

    
	gl_normal(-unit(ey));
	gl_vertex(c + size*ey - size*ez - size*ex);
	gl_vertex(c + size*ey - size*ez + size*ex);
	gl_vertex(c + size*ey + size*ez + size*ex);
	gl_vertex(c + size*ey + size*ez - size*ex);
    
	gl_normal(unit(ey));
	gl_vertex(c + (-size)*ey - size*ez - size*ex);
	gl_vertex(c + (-size)*ey + size*ez - size*ex);
	gl_vertex(c + (-size)*ey + size*ez + size*ex);
	gl_vertex(c + (-size)*ey - size*ez + size*ex);

    
	gl_normal(-unit(ez));
	gl_vertex(c + size*ez - size*ex - size*ey);
	gl_vertex(c + size*ez - size*ex + size*ey);
	gl_vertex(c + size*ez + size*ex + size*ey);
	gl_vertex(c + size*ez + size*ex - size*ey);
    
	gl_normal(unit(ez));
	gl_vertex(c + (-size)*ez - size*ex - size*ey);
	gl_vertex(c + (-size)*ez + size*ex - size*ey);
	gl_vertex(c + (-size)*ez + size*ex + size*ey);
	gl_vertex(c + (-size)*ez - size*ex + size*ey);
	
    
    glEnd();
}


void draw_wire_box(const box3& box)
{
	vec3 c = (box.minimum + box.maximum)/2.0;
	vec3 t = box.maximum - c;

	vec3 ex = vec3(t[0], 0, 0);
	vec3 ey = vec3(0, t[1], 0);
	vec3 ez = vec3(0, 0, t[2]);

	real size = 1.0;

   
    glBegin(GL_LINE_LOOP);
    
	
	gl_normal(-unit(ex));
	gl_vertex(c + size*ex - size*ey - size*ez);
	gl_vertex(c + size*ex - size*ey + size*ez);
	gl_vertex(c + size*ex + size*ey + size*ez);
	gl_vertex(c + size*ex + size*ey - size*ez);
    
	glEnd();

	glBegin(GL_LINE_LOOP);
	gl_normal(unit(ex));
	gl_vertex(c + (-size)*ex - size*ey - size*ez);
	gl_vertex(c + (-size)*ex + size*ey - size*ez);
	gl_vertex(c + (-size)*ex + size*ey + size*ez);
	gl_vertex(c + (-size)*ex - size*ey + size*ez);
	glEnd();

    glBegin(GL_LINE_LOOP);
	gl_normal(-unit(ey));
	gl_vertex(c + size*ey - size*ez - size*ex);
	gl_vertex(c + size*ey - size*ez + size*ex);
	gl_vertex(c + size*ey + size*ez + size*ex);
	gl_vertex(c + size*ey + size*ez - size*ex);
    glEnd();

	glBegin(GL_LINE_LOOP);
	gl_normal(unit(ey));
	gl_vertex(c + (-size)*ey - size*ez - size*ex);
	gl_vertex(c + (-size)*ey + size*ez - size*ex);
	gl_vertex(c + (-size)*ey + size*ez + size*ex);
	gl_vertex(c + (-size)*ey - size*ez + size*ex);
	glEnd();


    glBegin(GL_LINE_LOOP);
	gl_normal(-unit(ez));
	gl_vertex(c + size*ez - size*ex - size*ey);
	gl_vertex(c + size*ez - size*ex + size*ey);
	gl_vertex(c + size*ez + size*ex + size*ey);
	gl_vertex(c + size*ez + size*ex - size*ey);
	glEnd();
    
	glBegin(GL_LINE_LOOP);
	gl_normal(unit(ez));
	gl_vertex(c + (-size)*ez - size*ex - size*ey);
	gl_vertex(c + (-size)*ez + size*ex - size*ey);
	gl_vertex(c + (-size)*ez + size*ex + size*ey);
	gl_vertex(c + (-size)*ez - size*ex + size*ey);
    
    glEnd();
}
static vec3 ex = vec3(1, 0, 0);
static vec3 ey = vec3(0, 1, 0);
static vec3 ez = vec3(0, 0, 1);

void draw_box(const vec3& c, real size)
{
    vec3 p;

    glBegin(GL_QUADS);
    
	
	gl_normal(ex);
	gl_vertex(c + size*ex - size*ey - size*ez);
	gl_vertex(c + size*ex - size*ey + size*ez);
	gl_vertex(c + size*ex + size*ey + size*ez);
	gl_vertex(c + size*ex + size*ey - size*ez);
    
	gl_normal(-ex);
	gl_vertex(c + (-size)*ex - size*ey - size*ez);
	gl_vertex(c + (-size)*ex + size*ey - size*ez);
	gl_vertex(c + (-size)*ex + size*ey + size*ez);
	gl_vertex(c + (-size)*ex - size*ey + size*ez);

    
	gl_normal(ey);
	gl_vertex(c + size*ey - size*ez - size*ex);
	gl_vertex(c + size*ey - size*ez + size*ex);
	gl_vertex(c + size*ey + size*ez + size*ex);
	gl_vertex(c + size*ey + size*ez - size*ex);
    
	gl_normal(-ey);
	gl_vertex(c + (-size)*ey - size*ez - size*ex);
	gl_vertex(c + (-size)*ey + size*ez - size*ex);
	gl_vertex(c + (-size)*ey + size*ez + size*ex);
	gl_vertex(c + (-size)*ey - size*ez + size*ex);

    
	gl_normal(ez);
	gl_vertex(c + size*ez - size*ex - size*ey);
	gl_vertex(c + size*ez - size*ex + size*ey);
	gl_vertex(c + size*ez + size*ex + size*ey);
	gl_vertex(c + size*ez + size*ex - size*ey);
    
	gl_normal(-ez);
	gl_vertex(c + (-size)*ez - size*ex - size*ey);
	gl_vertex(c + (-size)*ez + size*ex - size*ey);
	gl_vertex(c + (-size)*ez + size*ex + size*ey);
	gl_vertex(c + (-size)*ez - size*ex + size*ey);
	
    
    glEnd();
}


void draw_wire_box(const vec3& c, real size)
{
    vec3 p;

    //glBegin(GL_QUADS);
    
	glBegin(GL_LINE_LOOP);
	gl_normal(size*ex);
	gl_vertex(c + size*ex - size*ey - size*ez);
	gl_vertex(c + size*ex - size*ey + size*ez);
	gl_vertex(c + size*ex + size*ey + size*ez);
	gl_vertex(c + size*ex + size*ey - size*ez);
	glEnd();
    
	glBegin(GL_LINE_LOOP);
	gl_normal((-size)*ex);
	gl_vertex(c + (-size)*ex - size*ey - size*ez);
	gl_vertex(c + (-size)*ex - size*ey + size*ez);
	gl_vertex(c + (-size)*ex + size*ey + size*ez);
	gl_vertex(c + (-size)*ex + size*ey - size*ez);
	glEnd();
    

    
	glBegin(GL_LINE_LOOP);
	gl_normal(size*ey);
	gl_vertex(c + size*ey - size*ez - size*ex);
	gl_vertex(c + size*ey - size*ez + size*ex);
	gl_vertex(c + size*ey + size*ez + size*ex);
	gl_vertex(c + size*ey + size*ez - size*ex);
	glEnd();
    
	glBegin(GL_LINE_LOOP);
	gl_normal((-size)*ey);
	gl_vertex(c + (-size)*ey - size*ez - size*ex);
	gl_vertex(c + (-size)*ey - size*ez + size*ex);
	gl_vertex(c + (-size)*ey + size*ez + size*ex);
	gl_vertex(c + (-size)*ey + size*ez - size*ex);
	glEnd();
    

    
	glBegin(GL_LINE_LOOP);
	gl_normal(size*ez);
	gl_vertex(c + size*ez - size*ex - size*ey);
	gl_vertex(c + size*ez - size*ex + size*ey);
	gl_vertex(c + size*ez + size*ex + size*ey);
	gl_vertex(c + size*ez + size*ex - size*ey);
	glEnd();
    
	glBegin(GL_LINE_LOOP);
	gl_normal((-size)*ez);
	gl_vertex(c + (-size)*ez - size*ex - size*ey);
	gl_vertex(c + (-size)*ez - size*ex + size*ey);
	gl_vertex(c + (-size)*ez + size*ex + size*ey);
	gl_vertex(c + (-size)*ez + size*ex - size*ey);
	glEnd();
	
    
   // glEnd();
}

};  // namespace