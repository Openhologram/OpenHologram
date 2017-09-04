#ifndef __render_homography_h
#define __render_homography_h


#include "graphics/Shader.h"
#include "graphics/Homography.h"


namespace graphics {

class RenderHomography: public Shader {

public:

	RenderHomography();

	virtual void Initialize();

	virtual void BeginShader();
	virtual void EndShader();
	void SetTexture(GLint val);
	void SetHomography(Homography& h);

protected:

	GLint	texture_matrix_id_;
	GLint	texture_unit_id_;
	GLint	texture_;
	Homography homography_;
	float   matrix_[16];
};

};

#endif