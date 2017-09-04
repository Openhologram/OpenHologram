
#include "graphics/render_normal.h"


namespace graphics {


RenderNormal::RenderNormal(): Shader()
{
}

void
RenderNormal::BeginShader()
{
	Shader::BeginShader();
}

void
RenderNormal::Initialize()
{
	const char* v_shader_source ="varying vec3 N;							"	
						" varying vec3 p; "
						" void main(void)									"
						"{													"
						"	p = vec3(gl_ModelViewMatrix * gl_Vertex);		"
						"	N = normalize(vec3(gl_ModelViewMatrix * vec4(gl_Normal, 0.0))); "	
						"	gl_Position = ftransform();						"
						"}													\0";
	const char* shader_prog = 	"varying vec3 N;							"
		"						 varying vec3 p;                            "
		"void main(void)													"
		"{																	"
		"	vec3 R = normalize(gl_LightSource[0].position.xyz - p);			"
		"   float pp = dot(N,R);	"
		"   pp = pp > 0.0?pp:-pp;								"
		"   unsigned int  intensity = (unsigned int)(pp * 16777215.0); "   
		"   unsigned int  ib = (intensity&0x00ff0000)>>16; "
		"   unsigned int  ig = (intensity&0x0000ff00)>>8; "
		"   unsigned int  ir = (intensity&0x000000ff); "
		"	gl_FragColor = vec4(clamp((float)ir/255.0, 0.0, 1.0), clamp((float)ig/255.0, 0.0, 1.0), clamp((float)ib/255.0, 0.0, 1.0),1.0);	"
		//"   gl_FragColor = vec4(pp,pp,pp,1.0); "
		"}																	\0";


	SetVertexShaderSource(v_shader_source);
	SetPixelShaderSource(shader_prog);

	Shader::Initialize();
}


};