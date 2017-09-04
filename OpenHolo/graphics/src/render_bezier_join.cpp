

#include "graphics/render_bezier_join.h"


namespace graphics {


RenderBezierJoin::RenderBezierJoin(): Shader()
{
}

void
RenderBezierJoin::BeginShader()
{
	Shader::BeginShader();
	char out[100];
	int loc;
	strcpy(out, "joint");
	loc = glGetUniformLocationARB(shader_id_, out);
	glUniform1fvARB(loc, 2, joint_);
	strcpy(out, "stroke_width");
	loc = glGetUniformLocationARB(shader_id_, out);
	glUniform1fARB(loc, stroke_width_);
	loc = glGetUniformLocation(shader_id_, "model_matrix");
	glUniformMatrix4fv(loc, 1, GL_FALSE, model_mat_);
	loc = glGetUniformLocation(shader_id_, "projection_matrix");
	glUniformMatrix4fv(loc, 1, GL_FALSE, proj_mat_);
}


void RenderBezierJoin::set_bezier_join(const vec2& joint, float width, float model[16], float proj[16])
{
	joint_[0] = joint[0];
	joint_[1] = joint[1];
	stroke_width_ = width;
	memcpy(model_mat_, model, sizeof(float)*16);
	memcpy(proj_mat_, proj, sizeof(float)*16);
}

void
RenderBezierJoin::Initialize()
{
	char* vshader =
		"#version 330 core\n"
			"layout (location = 0) in vec3 position;\n"
			"uniform float stroke_width; \n"
			"uniform mat4 model_matrix; \n"
			"uniform mat4 projection_matrix; \n"
			"uniform float joint[2]; \n"
			"out vec4 pixel_pose;\n "
			"void main(void) \n"
			"{	\n"
			"	pixel_pose = vec4(position.xyz,1.0); \n"
			"	gl_Position = projection_matrix * (model_matrix * vec4(position.xyz,1.0)); \n"
			"}  \0";
	char* pshader =
		"#version 330 core\n"
			"uniform float stroke_width; \n"
			"uniform mat4 model_matrix; \n"
			"uniform mat4 projection_matrix; \n"
			"uniform float joint[2]; \n"
			"in vec4 pixel_pose; \n"
			"layout (location = 0) out vec4 color;"
			"void main(void)"
			"{	"
			"    vec2 a1 = vec2(joint[0], joint[1]); \n"
				"vec2 p = pixel_pose.xy; \n"
				"float len = length(a1-p); \n"
				"if (len < stroke_width) {\n"
					"float val = 1.0f - len/stroke_width;"
					"color = vec4(val,val,val,1.0); \n"
				"}\n"
				"else { "
					"discard; "
				"}\n"
			"}  \n";

	SetVertexShaderSource(vshader);
	SetPixelShaderSource(pshader);

	Shader::Initialize();
}


};