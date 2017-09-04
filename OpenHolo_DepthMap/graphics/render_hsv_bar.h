#ifndef __render_hsv_bar_h
#define __render_hsv_bar_h


#include "graphics/Shader.h"


namespace graphics {

class RenderHsvBar: public Shader {
public:
	enum Type {Hue, Saturation, Value, Red, Green, Blue, Alpha};
	// Hue        Hsv bar shows varying Hue for fixed v1(sat), v2(val),
	// Saturation Hsv bar shows varying Hue for fixed v1(hue), v2(val),
	// Value      Hsv bar shows varying Hue for fixed v1(hue), v2(sat),
public:

	RenderHsvBar();

	virtual void Initialize();

	void BeginShader();

	void SetV1(float d);
	void SetV2(float d);

	void SetTransparency(float d);
	void SetType(Type d) { type_ = d; }

	void SetWidth(int w) { width_ = w; }
	void SetHeight(int h) { height_ = h; }

protected:

	float	v1_;
	float   v2_;

	float	transparency_;
	int		type_;
	int     width_;
	int     height_;
	
};

};

#endif