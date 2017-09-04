

#include "graphics/render_normal_color.h"


namespace graphics {


RenderNormalColor::RenderNormalColor(): Shader()
{
}

void
RenderNormalColor::BeginShader()
{
	Shader::BeginShader();
}

void
RenderNormalColor::Initialize()
{
	const char* v_shader_source ="varying vec3 N;										"		
						"void main(void)													"
						"{																	"
						"	vec4 p = gl_ModelViewMatrix * gl_Vertex;						"
						"	N = gl_Normal;													"	
						"	gl_Position = ftransform();										"
						"	gl_FrontColor = gl_Color;										"
						"}																\0";
	const char* shader_prog = 	"varying vec3 N;								"
		"void main(void)													"
		"{																	"
		"   int a = N[0] * 65535;											"
		"   int b = N[1] * 65535;											"
		"	float red = ((0xff00&a)>>8)/255.0f;								"
		"	float green = ((0x00ff&a))/255.0f;								"
		"	float blue = ((0xff00&b)>>8)/255.0f;							"
		"	float alpha = ((0x00ff&b))/255.0f;								"
		"	gl_FragColor = vec4(red,green,blue,alpha);						"
		"}																	\0";


	SetVertexShaderSource(v_shader_source);
	SetPixelShaderSource(shader_prog);

	Shader::Initialize();
}


};