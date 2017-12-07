// Copyright (C) Tomoyoshi Shimobaba 2011-


#include "cwoGLDisp.h"
#include <stdio.h>

#include <GL/freeglut.h>
#include <GL/glext.h>
#include <cuda_gl_interop.h>
#include <cuda_runtime_api.h>


#include "cwo.h"
#include "cwoPLS.h"
#include "gwo.h"
#include "gwo_lib.h"


#ifdef _WIN32
	#include <windows.h>
	#define GET_PROC_ADDRESS(p)   wglGetProcAddress(p)
#elif defined(__APPLE__)
	extern void *OSXGetProcAddress(const char *name);
	#define GET_PROC_ADDRESS(p)   OSXGetProcAddress("_" p)
#elif defined(__ADM__)
	#include <GL/adm.h>
	#define GET_PROC_ADDRESS(p)   admGetProcAddress( (const GLubyte *) p)
#else
	#include <string.h>
	#include <GL/glx.h>
	#define GET_PROC_ADDRESS(p)   glXGetProcAddressARB( (const GLubyte *) p)
#endif

static PFNGLBINDBUFFERARBPROC    glBindBuffer;
static PFNGLDELETEBUFFERSARBPROC glDeleteBuffers;
static PFNGLGENBUFFERSARBPROC    glGenBuffers;
static PFNGLBUFFERDATAARBPROC    glBufferData;
static GLuint  bufferObj;
static cudaGraphicsResource *resource;
static uchar4* devPtr;
static int __Nx;
static int __Ny;


cwoGLDisp::cwoGLDisp(int Nx, int Ny)
{
	glBindBuffer     = NULL;
	glDeleteBuffers  = NULL;
	glGenBuffers     = NULL;
	glBufferData     = NULL;
	__Nx=Nx;
	__Ny=Ny;
}
cwoGLDisp::~cwoGLDisp()
{
	cudaGraphicsUnregisterResource(resource);
	glBindBuffer( GL_PIXEL_UNPACK_BUFFER_ARB, 0 );
	glDeleteBuffers( 1, &bufferObj );
}

float zzz=0.1;

void cwoGLDisp::key_func( unsigned char key, int x, int y ) {
    
	Key(key, x, y);

/*	switch (key) {
		case 56: //key 8
			zzz+=0.01;
			glutPostRedisplay();
			break;
		case 52: //key 4
			
			break;
		case 50: //key 2
			zzz-=0.01;
			 glutPostRedisplay();
			break;
		case 54: //key 6
			break;
        case 27:
            // clean up OpenGL and CUDA
            exit(0);
    }*/

}

CWO a;
GWO g;	
CWO rr,gg,bb;
GWO grr,ggg,gbb;

void cwoGLDisp::draw_func() 
{
//	printf("drwa func %e\n",zzz);

	
//	a.Load("mandrill.bmp");
//	g.Send(a);
//	g.Diffract(zzz,CWO_ANGULAR);
	
	rr.Load("mandrill1920.bmp",CWO_RED);
	gg.Load("mandrill1920.bmp",CWO_GREEN);
	bb.Load("mandrill1920.bmp",CWO_BLUE);
	grr.Send(rr);
	ggg.Send(gg);
	gbb.Send(bb);


	cudaGraphicsMapResources( 1, &resource, NULL ) ;
	size_t size;
	cudaGraphicsResourceGetMappedPointer( (void**)&devPtr, &size, resource);

	
//	g.Intensity();
//	g.Scale(255);
//	gwoDirectDispFloat(devPtr, (float*)g.GetBuffer(), __Nx , __Ny);

	gwoDirectDispFloatRGB(devPtr, 
		(float*)grr.GetBuffer(), 
		(float*)ggg.GetBuffer(), 
		(float*)gbb.GetBuffer(), 		
		__Nx , __Ny);
	

	cudaGraphicsUnmapResources( 1, &resource, NULL );


	glDrawPixels( __Nx, __Ny, GL_RGBA, GL_UNSIGNED_BYTE, 0 );
    glutSwapBuffers();

	
}

void cwoGLDisp::gl_init(int *argc, char **argv)
{
	int dev=0;
	cudaGLSetGLDevice( dev );
	glutInit( argc, argv );
	glutInitDisplayMode( GLUT_DOUBLE | GLUT_RGBA );
	//glutGameModeString ("1920x1080:32@60");
	//glutEnterGameMode ();
	glutInitWindowSize( __Nx, __Ny );
	glutCreateWindow( "bitmap" );

	glBindBuffer    = (PFNGLBINDBUFFERARBPROC)GET_PROC_ADDRESS("glBindBuffer");
	glDeleteBuffers = (PFNGLDELETEBUFFERSARBPROC)GET_PROC_ADDRESS("glDeleteBuffers");
	glGenBuffers    = (PFNGLGENBUFFERSARBPROC)GET_PROC_ADDRESS("glGenBuffers");
	glBufferData    = (PFNGLBUFFERDATAARBPROC)GET_PROC_ADDRESS("glBufferData");
	
	glGenBuffers( 1, &bufferObj );
	glBindBuffer( GL_PIXEL_UNPACK_BUFFER_ARB, bufferObj );
	glBufferData( GL_PIXEL_UNPACK_BUFFER_ARB, __Nx * __Ny * 4,NULL, GL_DYNAMIC_DRAW_ARB );
	cudaGraphicsGLRegisterBuffer( &resource, bufferObj, cudaGraphicsMapFlagsNone );	
	cudaGraphicsMapResources( 1, &resource, NULL );
	
	size_t  size;
	cudaGraphicsResourceGetMappedPointer( (void**)&devPtr, &size, resource);
}
void cwoGLDisp::gl_main()
{
	cudaGraphicsUnmapResources( 1, &resource, NULL );
	glutDisplayFunc( &draw_func );
	glutKeyboardFunc( &key_func );
	glutMainLoop();
}