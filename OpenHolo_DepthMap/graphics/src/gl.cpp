#include "graphics/gl.h"
#include "graphics/frame.h"
#include "graphics/Camera.h"


namespace graphics {

void gl_identity(real A[])
{
	A[1] = 0;  A[2] = 0;  A[3] = 0;  A[4] = 0; 
	A[6] = 0;  A[7] = 0;  A[8] = 0;  A[9] = 0; 
	A[11] = 0;  A[12] = 0;  A[13] = 0;  A[14] = 0; 
	A[0] = 1;
	A[5] = 1;
	A[10] = 1;
	A[15] = 1;
}


void gl_tran(real A[], const vec3& a)
{
    glPushMatrix();
    gl_loadmatrix(A);
    gl_translate(a);
    gl_getmatrix(A);
    glPopMatrix();
}

void gl_rot(real A[], const vec3& a)
{
    glPushMatrix();
    gl_loadmatrix(A);
    gl_rotate(a);
    gl_getmatrix(A);
    glPopMatrix();
}


real id_mat_real[] = {
	1, 0, 0, 0,
	0, 1, 0, 0,
	0, 0, 1, 0,
	0, 0, 0, 1,
};




void gl_material(material* mat)
{
    if(!mat) return;
    glMaterialfv(GL_FRONT_AND_BACK, GL_AMBIENT, mat->ambient);
    glMaterialfv(GL_FRONT_AND_BACK, GL_DIFFUSE, mat->diffuse);
    glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR, mat->specular);
}




};  // namespace