

#include "graphics/render_bezier.h"


namespace graphics {


RenderBezier::RenderBezier(): Shader()
{
}

void
RenderBezier::BeginShader()
{
	Shader::BeginShader();
	char out[100];
	int loc;
	strcpy(out, "bezier");
	loc = glGetUniformLocationARB(shader_id_, out);
	glUniform1fvARB(loc, 6, quad_bezier_);
	strcpy(out, "stroke_width");
	loc = glGetUniformLocationARB(shader_id_, out);
	glUniform1fARB(loc, stroke_width_);
	loc = glGetUniformLocation(shader_id_, "model_matrix");
	glUniformMatrix4fv(loc, 1, GL_FALSE, model_mat_);
	loc = glGetUniformLocation(shader_id_, "projection_matrix");
	glUniformMatrix4fv(loc, 1, GL_FALSE, proj_mat_);
}


void RenderBezier::set_bezier(const std::vector<vec2>& bezier, float width, float model[16], float proj[16])
{
	quad_bezier_[0] = bezier[0][0];
	quad_bezier_[1] = bezier[0][1];
	quad_bezier_[2] = bezier[1][0];
	quad_bezier_[3] = bezier[1][1];
	quad_bezier_[4] = bezier[2][0];
	quad_bezier_[5] = bezier[2][1];
	stroke_width_ = width;
	memcpy(model_mat_, model, sizeof(float)*16);
	memcpy(proj_mat_, proj, sizeof(float)*16);

}

void
RenderBezier::Initialize()
{
	char* vshader =
		"#version 330 core\n"
			"layout (location = 0) in vec3 position;\n"
			"uniform float stroke_width; \n"
			"uniform mat4 model_matrix; \n"
			"uniform mat4 projection_matrix; \n"
			"uniform float bezier[6]; \n"
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
			"uniform float bezier[6]; \n"
			"in vec4 pixel_pose; \n"
			"layout (location = 0) out vec4 color;"
			"vec2 evalQuadraticBezier(vec2 points_0, vec2 points_1, vec2 points_2, float x) \n"
			"{\n"
				"vec2 p1 = points_0;\n"
				"vec2 p2 = points_1;\n"
				"vec2 p3 = points_2;\n"
				"p1 = (1.0-x)*p1 + x*p2;\n"
				"p2 = (1.0-x)*p2 + x*p3;\n"
				"return (1.0-x)*p1 + x*p2;\n"
			"}"
			"void main(void)"
			"{	"
			"   vec4 a1 = vec4(bezier[0], bezier[1], 0.0, 1.0); \n"
			"   vec4 aa2 = vec4(bezier[2], bezier[3], 0.0, 1.0); \n"
			"   vec4 a3 = vec4(bezier[4], bezier[5], 0.0, 1.0);\n "
			"   vec2 points_0 = a1.xy; \n"
			"   vec2 points_1 = aa2.xy; \n"
			"   vec2 points_2 = a3.xy; \n"
			"   float M_PI = 3.141592653589f; \n"
				"vec2 A = points_0- (2.0f*points_1) + points_2;\n" 
				"vec2 B = 2.0f*(points_1-points_0);\n"
				"vec2 C = points_0;\n"
				"float AA = dot(A,A); \n"
				"float a1_div_3 = dot(A,B)/(AA*2.0f);\n"
				"float f = -A[0]/AA;\n"
				"float i = -0.25f*B[0]/AA;\n"
				"float h = (dot(C,A) + 0.5f * dot(B,B))/AA; \n"
				"float g = -A[1]/AA;\n"
				"float j = -0.25f*B[1]/AA;\n"
				"float k = (0.25f * dot(C,B))/AA;\n"
				" vec2 p = pixel_pose.xy; \n"
				"float a2 = f*p[0] + g * p[1] + h;\n"
				"float a3_div_2 = i*p[0] + j * p[1] + k; \n"
				"float q = a1_div_3 * a1_div_3 - a2/3.0f;\n"
				"float r = (a1_div_3 * a1_div_3 * a1_div_3) - 0.5f * (a1_div_3 * a2 )  + a3_div_2;\n"
				"float x1 = -1.0f;"
				"float x2 = -1.0f; "
				"float x3 = -1.0f;\n"
				"if (q*q*q - r*r > 0.0f) { \n"
				"	float Q = acos(r/sqrt(q*q*q))/3.0f;\n"
				"	x1 = -2.0f*sqrt(q)*cos(Q)-a1_div_3;\n"
				"	x2 = -2.0f*sqrt(q)*cos(Q+(2.0f*M_PI)/3.0f)-a1_div_3;\n"
				"	x3 = -2.0f*sqrt(q)*cos(Q-(2.0f*M_PI)/3.0f)-a1_div_3;\n"
				"}\n"
				" else {\n "
				"	float u = (r>=0.0f?-1.0f:1.0f)*pow((abs(r) + sqrt(r*r - q*q*q)), 1.0f/3.0f); \n"
				"	float v = (u != 0.0f)?q/u:0.0f;\n"
				"	x1 = (u + v)- a1_div_3;\n"
				"}\n"
			
				"if (x1 >= 0.0f && x1 <= 1.0f) {\n"
					"vec2 pnt = evalQuadraticBezier(points_0, points_1, points_2, x1);\n"
					"float len = length(pnt-p);\n"
					"if (len < stroke_width) {\n"
					 "  float val = 1.0f - len/stroke_width;"
						"color = vec4(val,val,val,1.0); \n"
					"}\n"
					"else discard; "
				"}\n"
				"else if (x2 >= 0.0f && x2 <= 1.0f) {\n"
					"vec2 pnt = evalQuadraticBezier(points_0, points_1, points_2, x2);\n"
					"float len = length(pnt-p);\n"
					"if (len < stroke_width) {\n"
					 "  float val = 1.0f-len/stroke_width;"
						"color = vec4(val,val,val,1.0); \n"
					"}\n"
					"else discard; "
				"}\n"
				"else if (x3 >= 0.0f && x3 <= 1.0f) {\n"
					"vec2 pnt = evalQuadraticBezier(points_0, points_1, points_2, x3);\n"
					"float len = length(pnt-p);\n"
					"if (len < stroke_width) {\n"
					 "  float val = 1.0f-len/stroke_width;"
						"color = vec4(val,val,val,1.0); \n"
					"}\n"
					"else discard; "
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