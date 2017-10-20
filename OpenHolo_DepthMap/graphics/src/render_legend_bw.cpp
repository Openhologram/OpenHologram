
#include "graphics/render_legend_bw.h"


namespace graphics {


RenderLegendBW::RenderLegendBW(): number_of_legend_(10), hsv_brightness_(1), transparency_(0), Shader()
{
}

void
RenderLegendBW::BeginShader()
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

void RenderLegendBW::SetNumberOfLegend(int n)
{
	number_of_legend_ = n;
}

void RenderLegendBW::SetHSVBrightness(float d)
{
	hsv_brightness_ = d;
}

void RenderLegendBW::SetTransparency(float d)
{
	transparency_ = d;
}

void RenderLegendBW::Initialize()
{

	const char* shader_prog = 		"uniform int number_of_legend;					"
		"uniform float hsv_brightness;					"
		"uniform float transparency;					"
		"void main(void)															"
		"{																			"
		"   float value = clamp(gl_Color.r + 0.0001f, 0.0f, 1.0f);												"
		"	float div = number_of_legend;											"
		"	float extend = value * div;												"
		"	float step = floor(extend);												"
		"	value = step/div;														"
		"	gl_FragColor = vec4(value*hsv_brightness, value*hsv_brightness, value*hsv_brightness, 1.0-transparency);								"
		"}																			\0";


	SetPixelShaderSource(shader_prog);

	Shader::Initialize();
}


};