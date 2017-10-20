#ifndef __render_alpha_h
#define __render_alpha_h


#include "graphics/Shader.h"


namespace graphics {

class RenderAlpha: public Shader {

public:

	RenderAlpha();

	virtual void Initialize();	
};

};

#endif