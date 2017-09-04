

#include "graphics/render_bezier_path.h"


namespace graphics {


RenderBezierPath::RenderBezierPath(): Shader()
{
}

void
RenderBezierPath::BeginShader()
{
	Shader::BeginShader();

}



void
RenderBezierPath::Initialize()
{
	char* vshader =
			"varying vec3 q; "
			"void main(void)"
			"{	"
			"   q = gl_Normal;"
			"	gl_Position = ftransform(); "
			"}  \0";
	char* pshader =
			"varying vec3 q; "
			"void main(void)"
			"{	"
			"   float val = q.x*q.x - q.y;"
			"   if (val*q.z >= 0.0f) { discard; }"
			"    else gl_FragColor = vec4(1,1,1,1); "
			"}  \0";

	SetVertexShaderSource(vshader);
	SetPixelShaderSource(pshader);

	Shader::Initialize();
}


};