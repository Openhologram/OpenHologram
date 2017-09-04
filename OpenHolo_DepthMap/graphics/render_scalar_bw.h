#ifndef RENDERSCALARBW_H_
#define RENDERSCALARBW_H_


#include "graphics/Shader.h"


namespace graphics {

class RenderScalarBW: public Shader {

public:

	RenderScalarBW();

	virtual void Initialize();

	void BeginShader();

	void SetNumberOfLegend(int n);

		// 0~1사이의 배정밀도 부동소수
	void SetHSVBrightness(float d);
	void SetTransparency(float d);

protected:

	int	   number_of_legend_;
	float hsv_brightness_;
	float transparency_;
};

};

#endif