#include "graphics/camera_map.h"
#include "graphics/glinfo.h"
#include <iostream>
#include <sstream>
#include <string>


#include <iomanip>
#include <cstdlib>

using std::stringstream;
using std::string;
using std::cout;
using std::endl;
using std::ends;

namespace graphics {


GLhandleARB CameraMap::camera_shader_id_ = 0;

GLhandleARB CameraMap::vertex_shader_handle_ = 0;

GLhandleARB CameraMap::fragment_shader_handle_ = 0;

static GLuint disp_list_camera_ = 0;

CameraMap::CameraMap(int w, int h ): Camera(w, h)
{
	forced_near_depth_ = 5.0;
	forced_far_depth_ = 100.0;
	video_texture_id_ = 0;
	selection_buffer.resize(0,0,0,0);
}

CameraMap::CameraMap(const CameraMap& c)
:  Camera(c)
{

	selection_buffer.resize(0,0,0,0);

}

int CameraMap::video_texture_unit_id() const
{ 
	return video_texture_unit_id_; 
}


// Loading shader function
GLhandleARB CameraMap::LoadShader(const char* source, unsigned int type)
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

void CameraMap::LoadShadowShader()
{
	const char* vertex_shader_source ="uniform mat4 tmat0;												"
						"varying vec3 N, L;													"		
						"varying vec4 q1;													"
						"void main(void)													"
						"{																	"
						"	vec4 p = gl_ModelViewMatrix * gl_Vertex;						"
						"	L = normalize(gl_LightSource[0].position.xyz - p.xyz);			"
						"	N = normalize(gl_NormalMatrix * gl_Normal);						"	
						"	gl_Position = ftransform();										"
						"	gl_TexCoord[0] = gl_TextureMatrix[0] * gl_MultiTexCoord0;		"	
						"	q1 = tmat0 * p;													"
						"	gl_FrontColor = gl_Color;										"
						"}																\0";

	const char*	pixel_shader_source ="uniform sampler2D VideoMap0;															"
						"varying vec3 N, L;																		"
						"varying vec4 q1;																		"
						"void projective_sample(vec4 a, sampler2D video)				"
						"{																						"
						"	vec4 q = a;																			"
						"	vec4 qq = q/q.w;																	"
						"	vec2 qq1 = qq.st;																	"
						"	qq1.y = 1.0f - qq.y;																"
						"	if (qq.x < 0.0f)	return;													"
						"	else if (qq.x > 1.0f)	return;												"
						"	else if (qq.y < 0.0f)	return;												"
						"	else if (qq.y > 1.0f)	return;											"
						"	else if (qq.z > 1.0f)	return;												"
						"	else if (qq.z < 0.0f)	return;												"
						"   gl_FragColor = texture2D(video,qq1); 											"
						"}																						"
						"void main(void)																		"
						"{																						"
						"	projective_sample(q1, VideoMap0);										"
						"}																						\0";

	camera_shader_id_ = glCreateProgramObjectARB();
	vertex_shader_handle_   = LoadShader(vertex_shader_source, GL_VERTEX_SHADER);
	fragment_shader_handle_ = LoadShader(pixel_shader_source, GL_FRAGMENT_SHADER);
	
	glAttachObjectARB(camera_shader_id_,vertex_shader_handle_);
	glAttachObjectARB(camera_shader_id_,fragment_shader_handle_);
	glLinkProgramARB(camera_shader_id_);
}


vec2 CameraMap::projectToImagePlane(const vec3& p) const
{
	if (camera_mode_ == kParallel) {
		real left = (-(grid_size_ * aspect_ratio)/2.0) / scale_;
		real bottom = (-grid_size_/2.0) / scale_;
		vec3 pnt = camera_pose_.to_model(p);
		
		real lsize = left * -2.0;
		real rsize = bottom * -2.0;
		real a = -pnt[0] - left;
		real b = pnt[1] - bottom;

		real x = (a * viewport_.GetWidth())/lsize;
		real y = (b * viewport_.GetHeight())/rsize;

		vec2 ret(x, y);

		return ret;
	}

	if (camera_mode_ == kImage) {
		plane pl(vec3(0.0,0.0,1.0), near_);
		vec3 pnt = camera_pose_.to_model(p);
		line ray(vec3(0.0,0.0,0.0), pnt);
		pl.intersect(ray, pnt);

		vec2 a(-pnt[0], pnt[1]);
		vec2 ret = (a-bottom_left_)/(top_right_ - bottom_left_);
		ret = ret * vec2(viewport_.GetWidth(), viewport_.GetHeight());
		return ret;
	}

	plane pl(vec3(0.0,0.0,1.0), forced_near_depth_);
	vec3 pnt = camera_pose_.to_model(p);

	line ray(vec3(0.0,0.0,0.0), pnt);
	pl.intersect(ray, pnt);
	vec2 a(pnt[0], pnt[1]);

	real hr = tan(radian(angle)/2.0) * forced_near_depth_;
	real wr = hr * aspect_ratio;

	a = a/vec2(-wr, hr);

	real cenw= ((real)viewport_.GetWidth())/2.0;
	real cenh= ((real)viewport_.GetHeight())/2.0;

	a = a * vec2(cenw, cenh);

	vec2 r = a + vec2(cenw, cenh);
	return r;
}


void CameraMap::getViewRay(const vec2& a, line& ray) const
{
	if (camera_mode_ == kParallel) {

		//LOG("get view ray %f %f\n", a[0], a[1]);
		real left = (-(grid_size_ * aspect_ratio)/2.0) / scale_;
		real bottom = (-grid_size_/2.0) / scale_;

		real a0 = a[0] / viewport_.GetWidth();
		real a1 = a[1] / viewport_.GetHeight();

		
		real lsize = left * -2.0;
		real rsize = bottom * -2.0;

		a0 = lsize * a0;
		a1 = rsize * a1;		
		
		real a00 = a0 + left;
		real a11 = a1 + bottom;

		//LOG("%f %f\n", a00, a11);

		vec3 pnt(-a00, a11, 0.0);
		pnt = camera_pose_.to_world(pnt);
		line ray1(pnt, pnt+camera_pose_.z_axis());
		ray = ray1;
		return;
	}

	if (camera_mode_ == kImage) {
		vec2 norm_coord = a/vec2(viewport_.GetWidth(), viewport_.GetHeight());
		vec2 new_coord = (norm_coord * (top_right_ - bottom_left_)) + bottom_left_;
		vec3 r2(-new_coord[0], new_coord[1], near_);
		vec3 r1 = camera_pose_.get_origin();

		r2 = camera_pose_.to_world(r2);
		ray.set_value(r1, r2);
		return;
	}

	real cenw= ((real)viewport_.GetWidth())/2.0;
	real cenh= ((real)viewport_.GetHeight())/2.0;

	vec2 norm_coord = (a - vec2(cenw, cenh))/vec2(cenw, cenh);

	real hr = tan(radian(angle)/2.0) * forced_near_depth_;
	real wr = hr * aspect_ratio;

	vec2 new_coord = norm_coord * vec2(-wr, hr);

	vec3 coord(new_coord[0], new_coord[1],forced_near_depth_);

	coord = camera_pose_.to_world(coord);
	ray.set_value(camera_pose_.get_origin(), coord);

}

void CameraMap::set_view()
{
	//LOG("camera map set view %f %f\n", forced_near_depth_,forced_far_depth_ );
	glMatrixMode(GL_PROJECTION);
	glViewport(0,0,viewport_.GetWidth(),viewport_.GetHeight());
    glLoadIdentity();

	if (camera_mode_ == kPerspective) {
		gluPerspective(angle, aspect_ratio, forced_near_depth_, forced_far_depth_);
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

	glGetDoublev(GL_MODELVIEW_MATRIX, model_view_matrix);
	glGetDoublev(GL_PROJECTION_MATRIX, projection_matrix);
}

void CameraMap::DrawTexturedCamera() const
{
	if(disp_list_camera_ != 0)
	{
		glPushAttrib(GL_ALL_ATTRIB_BITS);

		GLfloat light_position1[] = {1,1,0,0};
		GLfloat light_position2[] = {-1,-1,0,0};
		GLfloat light_diffuse[] = {1,1,1,1};
		GLfloat light_ambient[] = {0.4,0.4,0.4,1};	
		GLfloat light_specular[] = {1,1,1,1};
		GLfloat metal_shininess[] = { 50.0 };		
		GLfloat global_ambient[] = {1,1,1,1};		

		glLightfv(GL_LIGHT1, GL_POSITION, light_position1);
		glLightfv(GL_LIGHT1, GL_AMBIENT, light_ambient);
		glLightfv(GL_LIGHT1, GL_DIFFUSE, light_diffuse);
		glLightfv(GL_LIGHT1, GL_SPECULAR, light_specular);

		glLightfv(GL_LIGHT2, GL_POSITION, light_position2);
		glLightfv(GL_LIGHT2, GL_AMBIENT, light_ambient);
		glLightfv(GL_LIGHT2, GL_DIFFUSE, light_diffuse);
		glLightfv(GL_LIGHT2, GL_SPECULAR, light_specular);

		glMateriali(GL_FRONT_AND_BACK, GL_SHININESS, 128);

		glEnable(GL_LIGHTING);
		glEnable(GL_LIGHT1);
		glEnable(GL_LIGHT2);

		glEnable(GL_LINE_SMOOTH);
		glEnable(GL_POLYGON_SMOOTH);
		glEnable(GL_MULTISAMPLE);
		glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
		glLightModeli(GL_LIGHT_MODEL_TWO_SIDE, GL_TRUE);
		glLightModelfv(GL_LIGHT_MODEL_AMBIENT, global_ambient);
		glEnable(GL_ALPHA_TEST);		
		glEnable(GL_DEPTH_TEST);
		glEnable(GL_BLEND);
		glDisable(GL_COLOR_MATERIAL);
		glBlendFunc(GL_SRC_ALPHA,GL_ONE_MINUS_SRC_ALPHA);
		glShadeModel(GL_SMOOTH);

		glCallList(disp_list_camera_);
		glPopAttrib();
		glEnable(GL_COLOR_MATERIAL);
		return;
	}

	disp_list_camera_ = glGenLists(1);
	glNewList(disp_list_camera_, GL_COMPILE);

	std::vector<vec3> lens;			
	std::vector<vec3> lens_inner;	
	std::vector<vec3> lens_outer;	
	std::vector<vec3> lens_outer_outer;	
	std::vector<vec3> lens_normal;			
	real r(1);
	real interval = 2*M_PI*r/36.0;
	// ¸öÃ¼¿ë
	for(real i=0; i<2*M_PI*r; i+=interval)
	{
		vec3 v = vec3(cos(i), sin(i), 0);
		vec3 n = unit(v-vec3(0));
		lens.push_back(v);
		lens_normal.push_back(n);
		lens_inner.push_back(vec3(cos(i)*0.8, sin(i)*0.8, 0.3));
	}
	// ¶Ñ²±¿ë
	for(real i=0; i<M_PI*r; i+=interval)
	{
		real y=sin(i);
		lens_outer.push_back(vec3(cos(i)*1.1, sin(i)*1.1, -1.2*y));
		lens_outer_outer.push_back(vec3(cos(i)*1.2, sin(i)*1.2, -1.2*y));
	}

	glPushMatrix();
	glRotatef(180, 0, 1, 0);

	// ¾Õ ¸é
	GLfloat lens_cab[] = {0.3, 0.3, 0.3, 1.0};
	glMaterialfv(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE, lens_cab);
	glBegin(GL_QUADS);
	for(int i=0; i<lens.size()-1;++i)
	{
		gl_vertex(lens[i]);
		gl_vertex(lens[i+1]);
		gl_vertex(lens_inner[i+1]);
		gl_vertex(lens_inner[i]);
	}
	glEnd();

	glDisable(GL_LIGHTING);
	glColor3f(0,0,0);
	glBegin(GL_LINES);
	for(int i=0; i<lens.size()-1;++i)
	{
		gl_vertex(lens[i]);
		gl_vertex(lens_inner[i]);
	}
	glEnd();
	glBegin(GL_LINE_STRIP);
	for(int i=0; i<lens.size();++i)
	{
		gl_vertex(lens[i]);
	}
	glEnd();
	glBegin(GL_LINE_STRIP);
	for(int i=0; i<lens_inner.size();++i)
	{
		gl_vertex(lens_inner[i]);
	}
	glEnd();

	glEnable(GL_LIGHTING);

	// ¸öÅë		
	GLfloat metal[] = {0.3, 0.3, 0.3, 1.0};
	glMaterialfv(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE, metal);

	glBegin(GL_QUAD_STRIP);
	for(int i=0; i<lens.size();++i)
	{
		vec3 v = lens[i];
		glNormal3f(lens_normal[i].v[0], lens_normal[i].v[1], lens_normal[i].v[2]);
		gl_vertex(v);
		v.v[2] = 2.9;
		gl_vertex(v);
	}
	glEnd();
	//µÞ¸é		
	glBegin(GL_POLYGON);
	for(int i=0; i<lens.size();++i)
	{
		glNormal3f(0,0,1);
		vec3 v = lens[i];
		v.v[2] = 2.9;
		gl_vertex(v);
	}
	glEnd();

	//¶Ñ²±
	GLfloat metal_cab[] = {0.1, 0.1, 0.1, 1};
	glMaterialfv(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE, metal_cab);
	glBegin(GL_QUAD_STRIP);
	for(int i=0; i<lens_outer.size();++i)
	{
		vec3 v = lens_outer[i];
		gl_normal(lens_normal[i]);
		gl_vertex(v);
		v.v[2] = 3;
		gl_vertex(v);
	}
	glEnd();
	glBegin(GL_QUAD_STRIP);
	for(int i=0; i<lens_outer_outer.size();++i)
	{
		vec3 v = lens_outer_outer[i];
		gl_normal(lens_normal[i]);
		gl_vertex(v);
		v.v[2] = 3;
		gl_vertex(v);
	}
	glEnd();
	glBegin(GL_QUADS);
	for(int i=0; i<lens_outer_outer.size()-1;++i)
	{
		gl_normal(vec3(0,0,1));
		gl_vertex(lens_outer[i]);
		gl_vertex(lens_outer[i+1]);
		gl_vertex(lens_outer_outer[i+1]);
		gl_vertex(lens_outer_outer[i]);
	}
	glEnd();

	glBegin(GL_POLYGON);
	for(int i=0; i<lens_outer_outer.size()-1;++i)
	{
		vec3 v1 = lens_outer_outer[i];
		vec3 v2 = lens_outer_outer[i+1];
		//vec3 v3 = lens_outer_outer[i+1];
		//vec3 v4 = lens_outer_outer[i];
		v1.v[2] = 3;
		v2.v[2] = 3;
		//v3.v[2] = 3;
		//v4.v[2] = 3;
		gl_normal(vec3(0,0,1));
		gl_vertex(v1);
		gl_vertex(v2);
		//gl_vertex(v3);
		//gl_vertex(v4);
	}
	glEnd();

	{
		vec3 v1 = lens_outer[0];
		vec3 v2 = lens_outer_outer[0];
		vec3 v3 = v2; v3.v[2] = 3;
		vec3 v4 = v1; v4.v[2] = 3;
		glBegin(GL_QUADS);
		gl_normal(vec3(0,1,0));
		gl_vertex(v1);
		gl_vertex(v2);
		gl_vertex(v3);
		gl_vertex(v4);
		glEnd();

		vec3 v5 = lens_outer[lens_outer.size()-1];
		vec3 v6 = lens_outer_outer[lens_outer.size()-1];
		vec3 v7 = v6; v6.v[2] = 3;
		vec3 v8 = v5; v5.v[2] = 3;
		glBegin(GL_QUADS);
		gl_normal(vec3(0,1,0));
		gl_vertex(v5);
		gl_vertex(v6);
		gl_vertex(v7);
		gl_vertex(v8);
		glEnd();
	}

	GLfloat lens_glass[] = {0, 0.2, 0.7, 0.3};
	glMaterialfv(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE, lens_glass);
	glBegin(GL_POLYGON);
	for(int i=0; i<lens_inner.size();++i)
	{
		glNormal3f(0,0,1);
		vec3 v = lens_inner[i];
		gl_vertex(v);
	}
	glEnd();

	// arm
	GLfloat arm[] = {0.1, 0.1, 0.1, 1.0};
	glMaterialfv(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE, arm);
	glRotatef(90, -1, 0, 0);
	glTranslatef(0, -2, -1.2);
	GLUquadric* cylinder = gluNewQuadric();
	gluCylinder(cylinder, 0.3, 0.4, 0.3, 30, 30);

	GLUquadric* elbow = gluNewQuadric();
	glTranslatef(0, 0, -0.1);
	gluSphere(elbow, 0.3, 30, 30);

	glTranslatef(0, 0, -1.1);
	gluCylinder(cylinder, 0.2, 0.2, 1, 30, 30);

	glTranslatef(0, 0, -0.1);
	gluSphere(elbow, 0.3, 30, 30);

	glRotatef(90, -1, 0, 0);
	glTranslatef(0, 0, -1.4);
	gluCylinder(cylinder, 0.2, 0.2, 1.3, 30, 30);

	glPopMatrix();

	glEndList();
}

void CameraMap::DrawActiveMarker(void) const
{
	glMatrixMode(GL_MODELVIEW);	
	glPushMatrix();
	camera_pose_.push_to_world();
	
	glTranslatef(0, 2, -2);
	glColor3f(1,0,0);	
	GLUquadric* shpere = gluNewQuadric();	
	gluSphere(shpere, 1, 30, 30);

	camera_pose_.pop();
	glPopMatrix();
}

void  CameraMap::DrawCameraGeometry(bool full) const
{
	glPushAttrib(GL_ALL_ATTRIB_BITS);
	glMatrixMode(GL_MODELVIEW);
	glPushMatrix();
	glLoadIdentity();

	camera_pose_.push_to_world();

	DrawTexturedCamera();	

	glDisable(GL_DEPTH_TEST);

	real y = (full?forced_far_depth_:forced_near_depth_ * 0.5) * tan(radian(angle/2.0));
	real x = y * aspect_ratio;
	vec3 a(x,y,(full?forced_far_depth_:forced_near_depth_ * 0.5));
	vec3 b(-x,y,(full?forced_far_depth_:forced_near_depth_ * 0.5));
	vec3 c(-x,-y, (full?forced_far_depth_:forced_near_depth_ * 0.5));
	vec3 d(x,-y, (full?forced_far_depth_:forced_near_depth_ * 0.5));

	glEnable(GL_LINE_STIPPLE);
	glLineStipple(2, 0xAAAA);
	gl_color(vec3(0.3,0.3,0.3));
	if (full) gl_color(vec3(1,0,0));
	glLineWidth(1.0);

	glDisable(GL_LIGHTING);
	glBegin(GL_LINES);
	gl_vertex(vec3(0));
	gl_vertex(a);
	gl_vertex(vec3(0));
	gl_vertex(b);
	gl_vertex(vec3(0));
	gl_vertex(c);
	gl_vertex(vec3(0));
	gl_vertex(d);
	gl_vertex(a);
	gl_vertex(b);
	gl_vertex(b);
	gl_vertex(c);
	gl_vertex(c);
	gl_vertex(d);
	gl_vertex(d);
	gl_vertex(a);

	glEnd();

	if (full) {
		glPointSize(4);
		gl_color(vec3(0,0,1));
		glBegin(GL_POINTS);
		gl_vertex(vec3(0,0,forced_near_depth_));
		gl_vertex(vec3(0,0,forced_far_depth_));
		glEnd();
		gl_color(vec3(1,0,0));
		glBegin(GL_LINES);
		gl_vertex(vec3(0,0,forced_near_depth_));
		gl_vertex(vec3(0,0,forced_far_depth_));
		glEnd();
	}

	glEnable(GL_DEPTH_TEST);
	glDisable(GL_LINE_STIPPLE);
	glBegin(GL_LINES);
	gl_vertex(vec3(0));
	gl_vertex(a);
	gl_vertex(vec3(0));
	gl_vertex(b);
	gl_vertex(vec3(0));
	gl_vertex(c);
	gl_vertex(vec3(0));
	gl_vertex(d);
	gl_vertex(a);
	gl_vertex(b);
	gl_vertex(b);
	gl_vertex(c);
	gl_vertex(c);
	gl_vertex(d);
	gl_vertex(d);
	gl_vertex(a);

	glEnd();
	if (full) {
		glPointSize(4);
		gl_color(vec3(0,0,1));
		glBegin(GL_POINTS);
		gl_vertex(vec3(0,0,forced_near_depth_));
		gl_vertex(vec3(0,0,forced_far_depth_));
		glEnd();
		gl_color(vec3(1,0,0));
		glBegin(GL_LINES);
		gl_vertex(vec3(0,0,forced_near_depth_));
		gl_vertex(vec3(0,0,forced_far_depth_));
		glEnd();
	}
	camera_pose_.pop();

	glPopMatrix();
	glPopAttrib();

}

void
CameraMap::BeginShader()
{
	//LOG("begin shader\n");
	glHint(GL_PERSPECTIVE_CORRECTION_HINT,GL_NICEST);

	glUseProgramObjectARB(camera_shader_id_);

	GetShaderVariableId();

	const GLdouble bias[16] = {	
		0.5, 0.0, 0.0, 0.0, 
		0.0, 0.5, 0.0, 0.0,
		0.0, 0.0, 0.5, 0.0,
		0.5, 0.5, 0.5, 1.0
	};

	glMatrixMode(GL_MODELVIEW);
	
	glPushMatrix();
	glLoadIdentity();	
	glLoadMatrixd(bias);
	
	// concatating all matrice into one.
	glMultMatrixd (projection_matrix);
	glMultMatrixd (model_view_matrix);

	float A[16];
	glGetFloatv(GL_MODELVIEW_MATRIX, A);
	glPopMatrix();
	
	// Go back to normal matrix mode
	glMatrixMode(GL_MODELVIEW);

	glUniformMatrix4fvARB(texture_matrix_id_, 1, false, A);


	if (video_texture_id_) {
		glUniform1iARB(video_texture_unit_id_, 4);
		glActiveTextureARB(GL_TEXTURE3_ARB + 1);
	}

	if (video_texture_id_) {
		glBindTexture(GL_TEXTURE_2D, video_texture_id_);
	}
}

void 
CameraMap::EndShader()
{
	glUseProgramObjectARB(0);
	glActiveTextureARB(GL_TEXTURE0_ARB);
	glBindTexture(GL_TEXTURE_2D, 0);
}


void CameraMap::SetVideoTexture(GLuint id)
{
	video_texture_id_ = id;
}

void CameraMap::GetShaderVariableId() 
{	int id = 0;
	char buf[100];
	sprintf(buf, "%d", id);
	char out[100] = "DepthMap";
	strcat(out, buf);


	strcpy(out, "VideoMap");
	strcat(out, buf);
	video_texture_unit_id_ = glGetUniformLocationARB(camera_shader_id_, out);

	strcpy(out, "tmat");
	strcat(out, buf);
	texture_matrix_id_ = glGetUniformLocationARB(camera_shader_id_, out);
}



void
CameraMap::Initialize()
{

	if (camera_shader_id_ == 0) {
		LoadShadowShader();
	}


}

void CameraMap::ResetCameraGeometry(const box3& model)
{
	FitViewToBoundingBox(model);
}

};