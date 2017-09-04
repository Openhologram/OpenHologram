#include "graphics/render_alpha_bar.h"


namespace graphics {


RenderAlphaBar::RenderAlphaBar(): Shader()
{
}

void RenderAlphaBar::BeginShader()
{
	Shader::BeginShader();

	char* out;
	int loc;
	out = "width";
	loc = glGetUniformLocationARB(shader_id_, out);
	glUniform1iARB(loc, width_);
	out = "height";
	loc = glGetUniformLocationARB(shader_id_, out);
	glUniform1iARB(loc, height_);
}



void RenderAlphaBar::Initialize()
{
	const char* shader_prog = 	
		"uniform int   width; "
		"uniform int   height; "
		" void main(void)															"
		"{																			"
		"   int x = int(floor(gl_TexCoord[0].s * float(width))); "
		"   int y = int(floor(gl_TexCoord[0].t * float(height))); "
		"   int xx = (x / 5) % 2; "
		"   int yy = (y / 5) % 2; "
		"   int xy = (xx + yy) %2; "
		"   vec4 bground =  (xy != 0) ? vec4(1,1,1,1): vec4(0.875, 0.875, 0.875, 1.0); "
		"   vec4 fground = vec4(gl_Color.r, gl_Color.g, gl_Color.b, 1.0); "
		"   if ( x <= width/2 )    "
		"		gl_FragColor = fground * gl_Color.a + (1.0 - gl_Color.a) * bground; "
		"	else	"
		"		gl_FragColor = bground; "
		"}																			\0";

	SetPixelShaderSource(shader_prog);

	Shader::Initialize();
}


};