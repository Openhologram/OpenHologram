#ifndef __render_hsv_h
#define __render_hsv_h


#include "graphics/Shader.h"


namespace graphics {

class RenderHsv: public Shader {

public:

	RenderHsv();

	virtual void Initialize();

	void BeginShader();

	void SetHSVBrightness(float d);
	void SetTransparency(float d);
protected:

	float	hsv_brightness_;
	float	transparency_;
	
};

};

#endif