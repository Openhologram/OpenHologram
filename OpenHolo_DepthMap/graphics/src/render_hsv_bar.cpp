
#include "graphics/render_hsv_bar.h"


namespace graphics {


RenderHsvBar::RenderHsvBar(): v1_(1), v2_(1), transparency_(0), Shader()
{
}

void
RenderHsvBar::BeginShader()
{
	Shader::BeginShader();

	char* out;
	int loc;
	char* out2 = "v1";
	int loc2 = glGetUniformLocationARB(shader_id_, out2);
	glUniform1fARB(loc2, v1_);
	out = "transparency";
	loc = glGetUniformLocationARB(shader_id_, out);
	glUniform1fARB(loc, transparency_);
	out = "v2";
	loc = glGetUniformLocationARB(shader_id_, out);
	glUniform1fARB(loc, v2_);
	out = "type";
	loc = glGetUniformLocationARB(shader_id_, out);
	glUniform1iARB(loc, type_);
	out = "width";
	loc = glGetUniformLocationARB(shader_id_, out);
	glUniform1iARB(loc, width_);
	out = "height";
	loc = glGetUniformLocationARB(shader_id_, out);
	glUniform1iARB(loc, height_);
}


void RenderHsvBar::SetV1(float d)
{
	v1_ = d;
}

void RenderHsvBar::SetV2(float d)
{
	v2_ = d;
}

void RenderHsvBar::SetTransparency(float d)
{
	transparency_ = d;
}

void
RenderHsvBar::Initialize()
{

	const char* shader_prog = 	
		"uniform float v1;"
		"uniform float v2;"
		"uniform int   type;"
		"uniform int   width; "
		"uniform int   height; "
		"uniform float transparency;					"
		"vec3 HSVtoRGB(vec3 color)"
		"{"
		"	float f,p,q,t, hueRound;"
		"	int hueIndex;"
		"	float hue, saturation, value;"
		"	vec3 result;"
		"	hue = color.r;"
		"	saturation = color.g;"
		"	value = color.b;"
		"	hueRound = floor(hue * 6.0);"
		"	hueIndex = int(int(hueRound) % 6);"
		"	f = (hue * 6.0) - hueRound;"
		"	p = value * (1.0 - saturation);" 
		"	q = value * (1.0 - f*saturation);"
		"	t = value * (1.0 - (1.0 - f)*saturation);"
		"   if (hueIndex == 0) 	result = vec3(value,t,p);"
		"   else if (hueIndex == 1) result = vec3(q,value,p);"
		"   else if (hueIndex == 2) result = vec3(p,value,t);"
		"   else if (hueIndex == 3) result =vec3(p,q,value);"
		"   else if (hueIndex == 4) result =vec3(t,p,value);"
		"   else if (hueIndex == 5) result =vec3(value,p,q);"
		"	return result;"
		"}"
		" void main(void)															"
		"{																			"
		"   float c = gl_TexCoord[0].s;							"
		"   int x = int(floor(gl_TexCoord[0].s * float(width))); "
		"   int y = int(floor(gl_TexCoord[0].t * float(height))); "
		"   int xx = (x / 5) % 2; "
		"   int yy = (y / 5) % 2; "
		"   int xy = (xx + yy) %2; "
		"   vec4 bground = vec4(0,0,0,0); "
		"   vec4 fground = vec4(0,0,0,0); "
		"   vec4 out_comp = vec4(0,0,0,0); "
		"   float vv1 = clamp(v1, 0.0f, 0.99999f);"
		"   float vv2 = clamp(v2, 0.0f, 0.99999f);"
		"   vec3 hsv = vec3(0,0,0); "
		"   if (type == 0) "
		"		hsv = HSVtoRGB(vec3(c, vv1, vv2)); "
		"   else if (type == 1) "
		"       hsv = HSVtoRGB(vec3(vv1, c, vv2));"
		"   else if (type == 2) "
		"       hsv = vec3(c, c, c);"
		"   else if (type == 3) " 
		"       hsv = vec3(c, 0, 0);"
		"   else if (type == 4) "
		"       hsv = vec3(0, c, 0);"
		"   else if (type == 5) "
		"       hsv = vec3(0, 0, c);"
		"   else if (type == 6) { "
		"       bground =  (xy!=0) ? vec4(1,1,1,1): vec4(0,0,0,1.0); "
		"       fground = vec4(c, c, c, 1.0); "
		"       out_comp = fground * c + (1.0 - c) * bground; "
		"       hsv = vec3(out_comp.x, out_comp.y, out_comp.z); "
		"   } "
		"	gl_FragColor = vec4(clamp(hsv[0], 0.0f, 1.0f), clamp(hsv[1], 0.0f, 1.0f), clamp(hsv[2], 0.0f, 1.0f), 1.0-transparency);		"
		"}																			\0";

	SetPixelShaderSource(shader_prog);

	Shader::Initialize();
}


};