#ifndef __render_bezier_join_h
#define __render_bezier_join_h


#include "graphics/Shader.h"
#include "graphics/vec.h"
#include <vector>


namespace graphics {

class RenderBezierJoin: public Shader {

public:

	RenderBezierJoin();

	virtual void Initialize();

	void BeginShader();

	void set_bezier_join(const vec2& joint, float width, float model[16], float proj[16]);

private:

	float joint_[2];
	float stroke_width_;
	float model_mat_[16];
    float proj_mat_[16];

};

};

#endif