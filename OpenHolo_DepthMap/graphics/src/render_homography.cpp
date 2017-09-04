
#include "graphics/render_homography.h"


namespace graphics {


RenderHomography::RenderHomography(): Shader()
{
}

void
RenderHomography::SetTexture(GLint val)
{
	texture_ = val;
}

void 
RenderHomography::SetHomography(Homography& h)
{
	homography_ = h;
}

void 
RenderHomography::EndShader()
{
	Shader::EndShader();
	glActiveTextureARB(0);
	glBindTexture(GL_TEXTURE_2D, 0);
}

void
RenderHomography::BeginShader()
{
	Shader::BeginShader();

	texture_matrix_id_ = glGetUniformLocationARB(shader_id_, "homography");
	texture_unit_id_ = glGetUniformLocationARB(shader_id_, "texture");

	for (int i = 0 ; i < 16 ;++i)
		matrix_[i] = 0.0;

	matrix_[15] = 1.0;
	matrix_[0] = homography_.GetMatrix()[0];
	matrix_[4] = homography_.GetMatrix()[1];
	matrix_[8] = homography_.GetMatrix()[2];
	matrix_[1] = homography_.GetMatrix()[3];
	matrix_[5] = homography_.GetMatrix()[4];
	matrix_[9] = homography_.GetMatrix()[5];
	matrix_[2] = homography_.GetMatrix()[6];
	matrix_[6] = homography_.GetMatrix()[7];
	matrix_[10] = homography_.GetMatrix()[8];


	glUniformMatrix4fvARB(texture_matrix_id_, 1, false, matrix_);

	glUniform1iARB(texture_unit_id_, 0);
	glActiveTextureARB(GL_TEXTURE0_ARB);


	glEnable(GL_TEXTURE_2D);
	glBindTexture(GL_TEXTURE_2D, texture_);
}

void
RenderHomography::Initialize()
{
	const char* shader_prog = 	"uniform mat4 homography;												"
		"uniform sampler2D texture;															"
		"void main(void)													"
		"{																	"
		"   vec4 cc;												"
		"   cc.st = gl_TexCoord[0].st; "
		"   cc.z = 1.0f;													"
		"   cc.w = 1.0f; "
		"   cc = homography*cc;												"
		"   cc.st = cc.st/cc.z; "
		"   gl_FragColor = texture2D(texture, cc.st);						"
		"}																	\0";

	SetPixelShaderSource(shader_prog);

	Shader::Initialize();
}

};