#ifndef __render_path_h
#define __render_path_h


#include "graphics/Shader.h"


namespace graphics {

class RenderPath: public Shader {

public:

	RenderPath();

	virtual void Initialize();

	void BeginShader();
	
};

};

#endif