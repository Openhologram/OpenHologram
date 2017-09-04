#ifndef __my_gl_h
#define __my_gl_h

#include "graphics/sys.h"
#include "graphics/real.h"
#include "graphics/vec.h"
#include "graphics/ivec.h"
#include "graphics/misc.h"
#include "graphics/geom.h"
#include "graphics/unsigned.h"
#include "graphics/color_buffer.h"


namespace graphics {

inline void gl_vertex(real a) { vec3 b; b[0] = a; b[1] = 0.0; b[2] = 0.0; glVertex3dv(b.v); }
inline void gl_vertex(const ivec2& a) {glVertex2iv(a.v); }
inline void gl_vertex(const vec2& a) { vec3 b; b[0] = a[0]; b[1] = a[1]; b[2] = 0.0; glVertex3dv(b.v); }
inline void gl_vertex(const vec3& a) { glVertex3dv(a.v); }
inline void gl_vertex(const ivec3& a) { glVertex3iv(a.v); }
inline void gl_vertex(const vec4& a) { glVertex3dv(a.v); }
inline void gl_normal(const vec3& a) { glNormal3dv(a.v); }
inline void gl_normal(const ivec3& a) { glNormal3iv(a.v); }
inline void gl_tex_coord(const vec2& a) { glTexCoord2dv(a.v); }
inline void gl_color(const vec3& a) { glColor3dv(a.v); }
inline void gl_color(const vec4& a) { glColor4dv(a.v); }
inline void gl_color(const ivec3& a) { glColor3iv(a.v); }
inline void gl_color(const ivec4& a) { glColor4iv(a.v); }
inline void gl_translate(const vec3& a) { glTranslated(a[0], a[1], a[2]); }
inline void gl_rotate(const vec3& a) { glRotated(degree(norm(a)), a[0], a[1], a[2]); }

inline void gl_multmatrix(float A[]) { glMultMatrixf(A); }
inline void gl_loadmatrix(float A[]) { glLoadMatrixf(A); }
inline void gl_getmatrix(float A[]) { glGetFloatv(GL_MODELVIEW_MATRIX, A); }

inline void gl_multmatrix(double A[]) { glMultMatrixd(A); }
inline void gl_loadmatrix(double A[]) { glLoadMatrixd(A); }
inline void gl_getmatrix(double A[]) { glGetDoublev(GL_MODELVIEW_MATRIX, A); }
inline void gl_color(GLuint v) { 
    GLubyte r = (v&0xFF000000)>>24, g = (v&0x00FF0000)>>16, b = (v&0x0000FF00)>>8, a = v& 0x000000FF;
    glColor4ub(r, g, b, a);
}


struct material {
    float ambient[4];
    float diffuse[4];
    float specular[4];
};


void gl_identity(real A[]);
void gl_tran(real A[], const vec3& a);
void gl_rot(real A[], const vec3& a);

inline unsigned int gl_dlist(void* ptr) { return (unsigned int) ptr | 0xff000000; }

extern real id_mat_real[];

void gl_material(material* mat);




}; // namespace graphics
#endif
