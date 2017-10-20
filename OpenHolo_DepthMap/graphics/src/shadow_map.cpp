

#include "graphics/glinfo.h"
#include "graphics/shadow_map.h"

#include <iostream>
#include <sstream>
#include <string>

#include "graphics/teapot.h"

#include <iomanip>
#include <cstdlib>
using std::stringstream;
using std::string;
using std::cout;
using std::endl;
using std::ends;

namespace graphics {


static int kNumberOfShadowMap = 0;

GLhandleARB ShadowMap::shadow_shader_id_ = 0;

GLhandleARB ShadowMap::vertex_shader_handle_ = 0;

GLhandleARB ShadowMap::fragment_shader_handle_ = 0;


// function pointers for FBO extension
// Windows needs to get function pointers from ICD OpenGL drivers,
// because opengl32.dll does not support extensions higher than v1.1.



ShadowMap::ShadowMap(int w, int h ): Camera(w, h)
{
	id_ = kNumberOfShadowMap;
	kNumberOfShadowMap++;
}

ShadowMap::ShadowMap(const ShadowMap& c)
: frame_buffer_object_id_(c.frame_buffer_object_id_),
  depth_texture_id_(c.depth_texture_id_),
  depth_texture_unit_id_(c.depth_texture_unit_id_),
  Camera(c),
  id_(c.id_)
{
}


// Loading shader function
GLhandleARB ShadowMap::LoadShader(const char* source, unsigned int type)
{
	GLhandleARB handle;
	
	// shader Compilation variable
	GLint result;				// Compilation code result
	GLint errorLoglength ;
	char* errorLogText;
	GLsizei actualErrorLogLength;
	
	handle = glCreateShaderObjectARB(type);
	if (!handle)
	{
		//We have failed creating the vertex shader object.
		LOG("Failed creating vertex shader object.");
		exit(0);
	}
	

	glShaderSourceARB(
					  handle, //The handle to our shader
					  1, //The number of files.
					  &source, //An array of const char * data, which represents the source code of theshaders
					  NULL);
	
	glCompileShaderARB(handle);
	
	//Compilation checking.
	glGetObjectParameterivARB(handle, GL_OBJECT_COMPILE_STATUS_ARB, &result);
	
	// If an error was detected.
	if (!result)
	{
		//We failed to compile.
		LOG("Shader failed compilation.\n");
		
		//Attempt to get the length of our error log.
		glGetObjectParameterivARB(handle, GL_OBJECT_INFO_LOG_LENGTH_ARB, &errorLoglength);
		
		//Create a buffer to read compilation error message
		errorLogText = (char*)malloc(sizeof(char) * errorLoglength);
		
		//Used to get the final length of the log.
		glGetInfoLogARB(handle, errorLoglength, &actualErrorLogLength, errorLogText);
		
		// Display errors.
		LOG("%s\n",errorLogText);
		
		// Free the buffer malloced earlier
		free(errorLogText);
	}
	
	return handle;
}

void ShadowMap::LoadShadowShader()
{
	char* vshader = 
		"varying vec3 N, L;	"														
		"varying vec4 q; "
		"void main(void)"									
		"{"																
			"vec4 p = gl_ModelViewMatrix * gl_Vertex;						"
			"L = normalize(gl_LightSource[0].position.xyz - p.xyz);			"
			"N = normalize(gl_NormalMatrix * gl_Normal);					"		
			"gl_Position = ftransform();									"
			"gl_TexCoord[0] = gl_TextureMatrix[0] * gl_MultiTexCoord0;		"	
			"q = gl_TextureMatrix[7] * p;									"
			"gl_FrontColor = gl_Color;										"
		"}			\0";

	char* pshader =
			"uniform sampler2DShadow DepthMap0;		"
			"uniform sampler2DShadow VideoMap0;	"
			"varying vec3 N, L;	"
			"varying vec4 q;			"
			"float lookup( vec2 offSet)"
			"{"
			"	float xPixelOffset = 0.001;"
			"	float yPixelOffset = 0.001;"
			"	return shadow2DProj(DepthMap0, q + vec4(offSet.x * xPixelOffset * q.w, offSet.y * yPixelOffset * q.w, 0.0, 0.0) ).w;"
			"}"
			"void main(void)"
			"{	"
			"	float shadow ;"
			"	if (q.w > 0.0)"
			"	{"
			"		float x,y;"
			"		for (y = -1.5 ; y <=1.5 ; y+=1.0)"
			"			for (x = -1.5 ; x <=1.5 ; x+=1.0)"
			"				shadow += lookup(vec2(x,y));"
			"		shadow /= 16.0 ;"
			"	}"
			"	vec3 R = -normalize(reflect(L, N));"
			"	vec4 ambient = gl_FrontLightProduct[0].ambient;	"
			"	vec4 diffuse = gl_FrontLightProduct[0].diffuse * max(abs(dot(N, L)), 0.0);	"
			"	gl_FragColor = ambient + (0.8 + 0.2 * shadow) * diffuse;			"
			"}  \0";


	shadow_shader_id_ = glCreateProgramObjectARB();
	vertex_shader_handle_   = LoadShader(vshader,GL_VERTEX_SHADER);
	fragment_shader_handle_ = LoadShader(pshader,GL_FRAGMENT_SHADER);
	
	glAttachObjectARB(shadow_shader_id_,vertex_shader_handle_);
	glAttachObjectARB(shadow_shader_id_,fragment_shader_handle_);
	glLinkProgramARB(shadow_shader_id_);
}


void ShadowMap::BeginDraw()
{
	set_view();
    // set the rendering destination to FBO
    glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, frame_buffer_object_id_);

	glEnable(GL_DEPTH_TEST);
    // clear buffer
    glClearColor(1, 1, 1, 1);
    glClear(GL_DEPTH_BUFFER_BIT);

    glPushMatrix();

}

void
ShadowMap::EndDraw()
{
    glPopMatrix();
    glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, 0); // unbind
}

void ShadowMap::set_view()
{
	glMatrixMode(GL_PROJECTION);
	glViewport(0, 0,viewport_.GetWidth(),viewport_.GetHeight());
    glLoadIdentity();

	if (camera_mode_ == kPerspective) {
		gluPerspective(angle, aspect_ratio, 1.0, 50.0);
	}
	else if (camera_mode_ == kParallel) {
		real left = -(grid_size_ * aspect_ratio)/2.0;
		real right = -left;
		real bottom = -grid_size_/2.0;
		real top = -bottom;
		glOrtho(left / scale_, right / scale_, bottom / scale_, top / scale_, near_dist(), far_dist());
	}

	vec3 org = camera_pose_.get_origin();
	vec3 ref = org + camera_pose_.basis[2];
	vec3 up = camera_pose_.basis[1];
	
	gluLookAt(org[0], org[1], org[2], ref[0], ref[1], ref[2], up[0], up[1], up[2]);	
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();

	glGetDoublev(GL_MODELVIEW_MATRIX, modelView);
	glGetDoublev(GL_PROJECTION_MATRIX, projection);
}



void 
ShadowMap::DrawShadowMap()
{
	glEnable(GL_TEXTURE_2D);
    glBindTexture(GL_TEXTURE_2D, depth_texture_id_);

    glBegin(GL_QUADS);
        glColor4f(1, 1, 1, 1);

        // face v0-v1-v2-v3
        glNormal3f(0,0,1);
        glTexCoord2f(1, 1);  glVertex3f(1,1,1);
        glTexCoord2f(0, 1);  glVertex3f(-1,1,1);
        glTexCoord2f(0, 0);  glVertex3f(-1,-1,1);
        glTexCoord2f(1, 0);  glVertex3f(1,-1,1);

        // face v0-v3-v4-v5
        glNormal3f(1,0,0);
        glTexCoord2f(0, 1);  glVertex3f(1,1,1);
        glTexCoord2f(0, 0);  glVertex3f(1,-1,1);
        glTexCoord2f(1, 0);  glVertex3f(1,-1,-1);
        glTexCoord2f(1, 1);  glVertex3f(1,1,-1);

        // face v0-v5-v6-v1
        glNormal3f(0,1,0);
        glTexCoord2f(1, 0);  glVertex3f(1,1,1);
        glTexCoord2f(1, 1);  glVertex3f(1,1,-1);
        glTexCoord2f(0, 1);  glVertex3f(-1,1,-1);
        glTexCoord2f(0, 0);  glVertex3f(-1,1,1);

        // face  v1-v6-v7-v2
        glNormal3f(-1,0,0);
        glTexCoord2f(1, 1);  glVertex3f(-1,1,1);
        glTexCoord2f(0, 1);  glVertex3f(-1,1,-1);
        glTexCoord2f(0, 0);  glVertex3f(-1,-1,-1);
        glTexCoord2f(1, 0);  glVertex3f(-1,-1,1);

        // face v7-v4-v3-v2
        glNormal3f(0,-1,0);
        glTexCoord2f(0, 0);  glVertex3f(-1,-1,-1);
        glTexCoord2f(1, 0);  glVertex3f(1,-1,-1);
        glTexCoord2f(1, 1);  glVertex3f(1,-1,1);
        glTexCoord2f(0, 1);  glVertex3f(-1,-1,1);

        // face v4-v7-v6-v5
        glNormal3f(0,0,-1);
        glTexCoord2f(0, 0);  glVertex3f(1,-1,-1);
        glTexCoord2f(1, 0);  glVertex3f(-1,-1,-1);
        glTexCoord2f(1, 1);  glVertex3f(-1,1,-1);
        glTexCoord2f(0, 1);  glVertex3f(1,1,-1);
    glEnd();

    glBindTexture(GL_TEXTURE_2D, depth_texture_id_);
	glDisable(GL_TEXTURE_2D);

}

void
ShadowMap::BeginShader()
{
	glHint(GL_PERSPECTIVE_CORRECTION_HINT,GL_NICEST);

	glUseProgramObjectARB(shadow_shader_id_);

	const GLdouble bias[16] = {	
		0.5, 0.0, 0.0, 0.0, 
		0.0, 0.5, 0.0, 0.0,
		0.0, 0.0, 0.5, 0.0,
	0.5, 0.5, 0.5, 1.0};

	glMatrixMode(GL_TEXTURE);
	glActiveTextureARB(GL_TEXTURE7 + id_);
	
	glLoadIdentity();	
	glLoadMatrixd(bias);
	
	// concatating all matrice into one.
	glMultMatrixd (projection);
	glMultMatrixd (modelView);
	
	// Go back to normal matrix mode
	glMatrixMode(GL_MODELVIEW);

	glUniform1iARB(depth_texture_unit_id_, 7);
	glActiveTextureARB(GL_TEXTURE7_ARB + id_);
	glEnable(GL_TEXTURE_2D);
	glBindTexture(GL_TEXTURE_2D, depth_texture_id_);
}


void
ShadowMap::EndShader()
{
	glUseProgramObjectARB(0);
	glActiveTextureARB(GL_TEXTURE0_ARB);
	glBindTexture(GL_TEXTURE_2D, 0);
}


void ShadowMap::GetShaderVariableId() 
{	char buf[100];
	sprintf(buf, "%d", id_);
	char out[100] = "DepthMap";
	strcat(out, buf);

	depth_texture_unit_id_ = glGetUniformLocationARB(shadow_shader_id_, out);
}


 
void
ShadowMap::Initialize()
{

	glEnable(GL_TEXTURE_2D);
    // create a texture object
    glGenTextures(1, &depth_texture_id_);
    glBindTexture(GL_TEXTURE_2D, depth_texture_id_);
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);

    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

	glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_COMPARE_MODE, GL_COMPARE_R_TO_TEXTURE);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_COMPARE_FUNC, GL_LEQUAL);

	glTexParameterf(GL_TEXTURE_2D, GL_DEPTH_TEXTURE_MODE, GL_INTENSITY);

	//glTexParameteri(GL_TEXTURE_2D, GL_GENERATE_MIPMAP, GL_TRUE);

    glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT, viewport_.GetWidth(), viewport_.GetHeight(), 0, GL_DEPTH_COMPONENT, GL_FLOAT, 0);

    glBindTexture(GL_TEXTURE_2D, 0);

    // create a framebuffer object, you need to delete them when program exits.
    glGenFramebuffersEXT(1, &frame_buffer_object_id_);
    glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, frame_buffer_object_id_);


    // attach a texture to FBO depth attachement point
    glFramebufferTexture2DEXT(GL_FRAMEBUFFER_EXT, GL_DEPTH_ATTACHMENT_EXT, GL_TEXTURE_2D, depth_texture_id_, 0);

    glDrawBuffer(GL_NONE);
    glReadBuffer(GL_NONE);

   GLenum status = glCheckFramebufferStatusEXT(GL_FRAMEBUFFER_EXT);
   if(status != GL_FRAMEBUFFER_COMPLETE_EXT)
        LOG("no frame buffer object(FBO) can be used!!!!!!!!!!!!!!\n");

        glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, 0);

	if (kNumberOfShadowMap == 1) {
		LoadShadowShader();
	}

	GetShaderVariableId();

	camera_mode_ = kParallel;
	angle = 30.0;
	box3 bound;
	bound.extend(vec3(2));
	bound.extend(vec3(-2));
	FitViewToBoundingBox(bound);
}

void ShadowMap::ResetLightGeometry(const box3& model)
{
	FitViewToBoundingBox(model);
}

};