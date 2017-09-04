///////////////////////////////////////////////////////////////////////////////
// teapot.h
// ========
// vertex, normal and index array for teapot model
// (6320 polygons, 3241 vertices, 3259 normals)
//
//  AUTHOR: Song Ho Ahn (song.ahn@gmail.com)
// CREATED: 2005-10-05
// UPDATED: 2005-10-05
//
// 3D model is converted by the PolyTrans from Okino Computer Graphics, Inc.
//
// Bounding box of geometry = (-3,0,-2) to (3.434,3.15,2).
//
// drawTeapot()   : render it with VA
// drawTeapotVBO(): render it with VBO
///////////////////////////////////////////////////////////////////////////////

#ifndef TEAPOT_H
#define TEAPOT_H


#include "graphics/sys.h"

namespace graphics {



///////////////////////////////////////////////////////////////////////////////
// draw teapot using absolute pointers to indexed vertex array.
///////////////////////////////////////////////////////////////////////////////
void drawTeapot();

///////////////////////////////////////////////////////////////////////////////
// create a display list for teapot
// Call creatTeapotDL() once to create a DL. createTeapotDL() will return a ID
// of display list. Use this ID to render later, glCallList(id).
//
// Since display lists are part of server state, the client state and commands
// cannot be stored in display list. Therefore, glEnableClientState,
// glDisableClientState, glVertexPointer, and glNormalPointer cannot be inside
// a display list. Above client calls must be reside outside of glNewList() and
// glEndList() function.
///////////////////////////////////////////////////////////////////////////////
GLuint createTeapotDL();



///////////////////////////////////////////////////////////////////////////////
// draw teapot using only offset instead of absolute pointers.
// The caller must bind buffer ids and set the starting offset before call this
// functions. (glBindBufferARB, glVertexPointer, glNormalPointer, glIndexPointer)
///////////////////////////////////////////////////////////////////////////////
void drawTeapotVBO();
};

#endif
