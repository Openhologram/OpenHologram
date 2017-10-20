#ifndef __render_line_h
#define __render_line_h


#include "graphics/Shader.h"
#include "graphics/vec.h"
#include <vector>


namespace graphics {

class RenderLine: public Shader {

public:

	RenderLine();

	virtual void Initialize();

	void BeginShader();

	void set_line(const std::vector<vec2>& bezier, float width, float model[16], float proj[16]);

private:

	float line_[4];
	float stroke_width_;
	float model_mat_[16];
    float proj_mat_[16];

};

};

#endif