#ifndef __render_normal_h
#define __render_normal_h


#include "graphics/Shader.h"


namespace graphics {

class RenderNormal: public Shader {

public:

	RenderNormal();

	virtual void Initialize();

	void BeginShader();

protected:

};

};

#endif