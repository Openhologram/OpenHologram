

#include "graphics/render_legend.h"


namespace graphics {


RenderLegend::RenderLegend(): number_of_legend_(10), hsv_brightness_(1), transparency_(0), Shader()
{
}

void
RenderLegend::BeginShader()
{
	Shader::BeginShader();

	char* out = "number_of_legend";
	int loc = glGetUniformLocationARB(shader_id_, out);
	glUniform1iARB(loc, number_of_legend_);
	out = "hsv_brightness";
	loc = glGetUniformLocationARB(shader_id_, out);
	glUniform1fARB(loc, hsv_brightness_);
	out = "transparency";
	loc = glGetUniformLocationARB(shader_id_, out);
	glUniform1fARB(loc, transparency_);
}

void RenderLegend::SetNumberOfLegend(int n)
{
	number_of_legend_ = n;
}

void RenderLegend::SetHSVBrightness(float d)
{
	hsv_brightness_ = d;
}

void RenderLegend::SetTransparency(float d)
{
	transparency_ = d;
}

void
RenderLegend::Initialize()
{
	const char* shader_prog = 		"uniform int number_of_legend;					"
		"uniform float hsv_brightness;					"
		"uniform float transparency;					"
		"vec3 HSVtoRGB(vec3 color)"
		"{"
		"	float f,p,q,t, hueRound;"
		"	int hueIndex;"
		"	float hue, saturation, value;"
		"	vec3 result;"
		"	hue = color.r;"
		"	saturation = color.g;"
		"	value = color.b;"
		"	hueRound = floor(hue * 6.0);"
		"	hueIndex = int(hueRound) % 6;"
		"	f = (hue * 6.0) - hueRound;"
		"	p = value * (1.0 - saturation);" 
		"	q = value * (1.0 - f*saturation);"
		"	t = value * (1.0 - (1.0 - f)*saturation);"
		"   if (hueIndex == 0) 	result = vec3(value,t,p);"
		"   else if (hueIndex == 1) result = vec3(q,value,p);"
		"   else if (hueIndex == 2) result = vec3(p,value,t);"
		"   else if (hueIndex == 3) result =vec3(p,q,value);"
		"   else if (hueIndex == 4) result =vec3(t,p,value);"
		"   else if (hueIndex == 5) result =vec3(value,p,q);"
		"	return result;"
		"}"
		"void main(void)															"
		"{																			"
		"   float value = clamp(gl_Color.r + 0.0001f, 0.0f, 1.0f);												"
		"	float div = number_of_legend;											"
		"	float extend = value * div;												"
		"	float step = floor(extend);												"
		"	value = (step/div) * 0.6666666666f;											"
		"   vec3 hsv = HSVtoRGB(vec3(value, 1.0, hsv_brightness));					"
		"	gl_FragColor = vec4(1,0,0,0);							"
		"}																			\0";


	SetPixelShaderSource(shader_prog);

	Shader::Initialize();
}


};