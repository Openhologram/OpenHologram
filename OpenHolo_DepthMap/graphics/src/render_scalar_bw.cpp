
#include "graphics/render_scalar_bw.h"


namespace graphics {


RenderScalarBW::RenderScalarBW(): number_of_legend_(10), hsv_brightness_(1), transparency_(0), Shader()
{
}

void
RenderScalarBW::BeginShader()
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

void RenderScalarBW::SetNumberOfLegend(int n)
{
	number_of_legend_ = n;
}

void RenderScalarBW::SetHSVBrightness(float d)
{
	hsv_brightness_ = d;
}

void RenderScalarBW::SetTransparency(float d)
{
	transparency_ = d;
}

void
RenderScalarBW::Initialize()
{

	const char* shader_prog = 		"uniform int number_of_legend;					"
		"uniform float hsv_brightness;					"
		"uniform float transparency;					"
		"void main(void)															"
		"{																			"
		"   float value = gl_Color.r;												"
		"	gl_FragColor = vec4(value*hsv_brightness,value*hsv_brightness,value*hsv_brightness, 1.0-transparency);								"
		"}																			\0";

	SetPixelShaderSource(shader_prog);

	Shader::Initialize();
}


};