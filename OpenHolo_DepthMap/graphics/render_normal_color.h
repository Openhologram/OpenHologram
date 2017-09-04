#ifndef __render_normal_color_h
#define __render_normal_color_h

#include "graphics/Shader.h"


namespace graphics {

class RenderNormalColor: public Shader {

public:

	RenderNormalColor();

	virtual void Initialize();

	void BeginShader();

protected:

};

};

#endif