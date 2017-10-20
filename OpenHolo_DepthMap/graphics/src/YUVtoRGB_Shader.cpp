#include "graphics/YUVtoRGB_Shader.h"

namespace graphics
{
	YUVtoRGB_Shader::YUVtoRGB_Shader()
	{
	}

	void YUVtoRGB_Shader::BeginShader()
	{
		Shader::BeginShader();

		char* out = "Texture_height";
		int loc(0);

		loc = glGetUniformLocationARB(shader_id_, out);		
		glUniform1iARB(loc, texture_height_);

		out = "Ytex";
		loc = glGetUniformLocationARB(shader_id_, out);		
		glUniform1iARB(loc, ytex_);

		out = "Utex";
		loc = glGetUniformLocationARB(shader_id_, out);
		glUniform1iARB(loc, utex_);

		out = "Vtex";
		loc = glGetUniformLocationARB(shader_id_, out);
		glUniform1iARB(loc, vtex_);
	}

	int YUVtoRGB_Shader::GetTextureHeight(void) const
	{
		return texture_height_;
	}

	void YUVtoRGB_Shader::SetTextureHeight(int i)
	{
		texture_height_= i;
	}

	GLuint YUVtoRGB_Shader::GetUTex(void) const
	{
		return utex_;
	}

	void YUVtoRGB_Shader::SetUTex(GLuint a)
	{
		utex_ = a;
	}

	GLuint YUVtoRGB_Shader::GetVTex(void) const
	{
		return vtex_;
	}

	void YUVtoRGB_Shader::SetVTex(GLuint a)
	{
		vtex_ = a;
	}

	GLuint YUVtoRGB_Shader::GetYTex(void) const
	{
		return ytex_;
	}

	void YUVtoRGB_Shader::SetYTex(GLuint a)
	{
		ytex_ = a;
	}

	void YUVtoRGB_Shader::Initialize()
	{
		//const char* vertex_shader_prog = "void main(void) {\n"
		//	"gl_TexCoord[0] = gl_MultiTexCoord0;\n"
		//	"gl_TexCoord[1] = gl_MultiTexCoord1;\n"
		//	"gl_TexCoord[2] = gl_MultiTexCoord2;\n"
		//	"gl_Position = ftransform();\n"
		//	"}\n";

		//SetVertexShaderSource(vertex_shader_prog);

		const char* shader_prog = 	
			"uniform sampler2D Ytex;\n"
			"uniform sampler2D Utex, Vtex;\n"
			"uniform int Texture_height;\n"
			"void main(void) {\n"

			"  float nx,ny,r,g,b,y,u,v;\n"
			"  vec4 txl,ux,vx;"
			"  nx=gl_TexCoord[0].x;\n"
			"  ny = gl_TexCoord[0].y;\n"

			"  y=texture2D(Ytex,vec2(nx,ny)).r;\n"
			"  u=texture2D(Utex,vec2(nx,ny)).r;\n"
			"  v=texture2D(Vtex,vec2(nx,ny)).r;\n"

			"  y=1.1643*(y-0.0625);\n"
			"  u=u-0.5;\n"
			"  v=v-0.5;\n"

			"  r=y+1.5958*v;\n"
			"  g=y-0.39173*u-0.81290*v;\n"
			"  b=y+2.017*u;\n"
			"  gl_FragColor = vec4(r, g, b, 1.0);\n"
			"}\0";

		SetPixelShaderSource(shader_prog);

		Shader::Initialize();
	}
}