#ifndef __CameraMap_h
#define __CameraMap_h

#include "graphics/camera.h"

namespace graphics {

class CameraMap: public Camera {

public:

	CameraMap(int w = 0, int h = 0);

	CameraMap(const CameraMap& val);

	void Initialize();


	virtual void  set_view();

	void EndShader();


	void BeginShader();


	void DrawActiveMarker(void) const;
	virtual void  DrawCameraGeometry(bool full = false) const;

	void DrawTexturedCamera(void) const;

	// Reset light geometry to contain as many geometries as possible
	void ResetCameraGeometry(const box3& model);

	void SetVideoTexture(GLuint id);

	unsigned int video_texture_id() const { return video_texture_id_; }

	int video_texture_unit_id() const;

	int depth_texture_unit_id() const;

	void SetForcedNearDepth(real a) { forced_near_depth_ = a; }

	void SetForcedFarDepth(real a) { forced_far_depth_ = a; }

	real forced_near_depth() const { return forced_near_depth_; }

	real forced_far_depth() const { return forced_far_depth_; }

	virtual vec2  projectToImagePlane(const vec3& p) const; 

	virtual void  getViewRay(const vec2& a, line& ray) const;

private:

	static void LoadShadowShader();

	static GLhandleARB LoadShader(const char* source, unsigned int type);

	void GetShaderVariableId() ;


protected:

	double		model_view_matrix[16];
	double		projection_matrix[16];


	GLint		video_texture_unit_id_;

	GLuint		video_texture_id_;

	GLint		texture_matrix_id_;


	real		forced_near_depth_;

	real		forced_far_depth_;


	static GLhandleARB camera_shader_id_;

	static GLhandleARB vertex_shader_handle_;

	static GLhandleARB fragment_shader_handle_;


	int cctv_texture_id_;
	int cctv_texture_width_;
	int cctv_texture_height_;
};

};

#endif