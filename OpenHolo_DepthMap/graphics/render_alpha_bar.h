#ifndef __render_alpha_bar_h
#define __render_alpha_bar_h


#include "graphics/Shader.h"


namespace graphics {

class RenderAlphaBar: public Shader {

public:

	RenderAlphaBar();

	virtual void Initialize();	

	void BeginShader();

	void SetWidth(int w) { width_ = w; }
	void SetHeight(int h) { height_ = h; }

protected:

	int     width_;
	int     height_;

};

};

#endif