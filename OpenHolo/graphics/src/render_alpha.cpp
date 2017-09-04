#include "graphics/render_alpha.h"


namespace graphics {


RenderAlpha::RenderAlpha(): Shader()
{
}


void
RenderAlpha::Initialize()
{

	const char* shader_prog = 	
		" void main(void)															"
		"{																			"
		"   vec2 cc = gl_TexCoord[0].st;							"
		"   cc.x = (cc.x - 0.5) * 2.0; "   
		"   cc.y = (cc.y - 0.5) * 2.0; "
		"   vec2 v = cc; "
		"   int x = int(v.x * 100) + 100; "
		"   int y = int(v.y * 100) + 100; "
		"   int xx = (x / 20) % 2; "
		"   int yy = (y / 20) % 2; "
		"   int xy = (xx + yy) %2; "
		"   float len = length(v); "
		"   if (len < 1.0) { "
		"       vec4 bground =  (xy != 0) ? vec4(1,1,1,1): vec4(0,0,0,1.0); "
		"       vec4 fground = vec4(gl_Color.r, gl_Color.g, gl_Color.b, 1.0); "
		"       gl_FragColor = fground * gl_Color.a + (1.0 - gl_Color.a) * bground; "
		"   } else { "
		"       gl_FragColor = (xy != 0) ? vec4(1,1,1,1): vec4(0,0,0,1.0); "
		"   } "
		"}																			\0";


	SetPixelShaderSource(shader_prog);

	Shader::Initialize();
}


};