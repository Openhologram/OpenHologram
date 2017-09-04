#ifndef __render_bezier_path_h
#define __render_bezier_path_h


#include "graphics/Shader.h"


namespace graphics {

class RenderBezierPath: public Shader {

public:

	RenderBezierPath();

	virtual void Initialize();

	void BeginShader();
	
};

};

#endif