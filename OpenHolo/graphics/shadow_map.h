#ifndef __ShadowMap_h
#define __ShadowMap_h

#include "graphics/camera.h"

namespace graphics {

class ShadowMap: public Camera {

public:

    ShadowMap(int w = 1000, int h = 1000);

    ShadowMap(const ShadowMap& val);

	void Initialize();

	
	virtual void  set_view();

	// 1. before shadow map rendering
	void BeginDraw();

	// 2. here comes

	// 3. after shadow map rendering
	void EndDraw();

	void EndShader();


	void BeginShader();

	void DrawShadowMap();


	// Reset light geometry to contain as many geometries as possible
	void ResetLightGeometry(const box3& model);

private:

	static void LoadShadowShader();

	static GLhandleARB LoadShader(const char* shader_source, unsigned int type);

	void GetShaderVariableId() ;

protected:

	double modelView[16];
	double projection[16];

	// Hold id of the framebuffer for POV rendering
	GLuint frame_buffer_object_id_;

	// depth texture from frame buffer object
	GLuint depth_texture_id_;

	// this is texture unit id within the GPU
	GLuint depth_texture_unit_id_;


	static GLhandleARB shadow_shader_id_;

	static GLhandleARB vertex_shader_handle_;
	
	static GLhandleARB fragment_shader_handle_;

	int id_;

	GLuint render_buffer_object_id_;                       // ID of Renderbuffer object

};

};

#endif