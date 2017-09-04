

#include "graphics/render_line.h"


namespace graphics {


RenderLine::RenderLine(): Shader()
{
}

void
RenderLine::BeginShader()
{
	Shader::BeginShader();
	char out[100];
	int loc;
	strcpy(out, "line");
	loc = glGetUniformLocationARB(shader_id_, out);
	glUniform1fvARB(loc, 4, line_);
	strcpy(out, "stroke_width");
	loc = glGetUniformLocationARB(shader_id_, out);
	glUniform1fARB(loc, stroke_width_);
	loc = glGetUniformLocation(shader_id_, "model_matrix");
	glUniformMatrix4fv(loc, 1, GL_FALSE, model_mat_);
	loc = glGetUniformLocation(shader_id_, "projection_matrix");
	glUniformMatrix4fv(loc, 1, GL_FALSE, proj_mat_);
}


void RenderLine::set_line(const std::vector<vec2>& line, float width, float model[16], float proj[16])
{
	line_[0] = line[0][0];
	line_[1] = line[0][1];
	line_[2] = line[1][0];
	line_[3] = line[1][1];

	stroke_width_ = width;
	memcpy(model_mat_, model, sizeof(float)*16);
	memcpy(proj_mat_, proj, sizeof(float)*16);

}

void
RenderLine::Initialize()
{
	char* vshader =
		"#version 330 core\n"
			"layout (location = 0) in vec3 position;\n"
			"uniform float stroke_width; \n"
			"uniform mat4 model_matrix; \n"
			"uniform mat4 projection_matrix; \n"
			"uniform float line[4]; \n"
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
			"uniform float line[6]; \n"
			"in vec4 pixel_pose; \n"
			"layout (location = 0) out vec4 color;"
			"void main(void)"
			"{	"
			"   vec4 aa1 = vec4(line[0], line[1], 0.0, 1.0); \n"
			"   vec4 aa2 = vec4(line[2], line[3], 0.0, 1.0); \n"
			"   vec2 p0 = aa1.xy; \n"
			"   vec2 p1 = aa2.xy; \n"
			"	vec2 p = pixel_pose.xy; \n"
			"   vec2 dir1 = p-p0; \n"
			"   vec2 dir = p1-p0; \n"
			"   float range = length(dir); \n"
			"   dir = normalize(dir); \n"
			"   float t = dot(dir, dir1);\n"
			"   if (t < 0.0) discard; \n"
			"   else if (t > range) discard; \n"
			"   else { \n"
			"		float d = length(p-(t*dir + p0));\n"
			"		if (d < stroke_width) {\n"
			"			float val = 1.0f - d/stroke_width;"
			"			color = vec4(val,val,val,1.0); \n"
			"		}\n"
			"		else discard; \n"
			"	}\n"
			"}  \0";

	SetVertexShaderSource(vshader);
	SetPixelShaderSource(pshader);

	Shader::Initialize();
}


};