#ifndef __render_bezier_h
#define __render_bezier_h


#include "graphics/Shader.h"
#include "graphics/vec.h"
#include <vector>


namespace graphics {

class RenderBezier: public Shader {

public:

	RenderBezier();

	virtual void Initialize();

	void BeginShader();

	void set_bezier(const std::vector<vec2>& bezier, float width, float model[16], float proj[16]);

private:

	float quad_bezier_[6];
	float stroke_width_;
	float model_mat_[16];
    float proj_mat_[16];

};

};

#endif