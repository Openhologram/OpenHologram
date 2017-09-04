

#include "graphics/render_hsv.h"


namespace graphics {


RenderHsv::RenderHsv(): hsv_brightness_(1), transparency_(0), Shader()
{
}

void
RenderHsv::BeginShader()
{
	Shader::BeginShader();

	char* out;
	int loc;
	char* out2 = "hsv_brightness";
	int loc2 = glGetUniformLocationARB(shader_id_, out2);
	glUniform1fARB(loc2, hsv_brightness_);
	out = "transparency";
	loc = glGetUniformLocationARB(shader_id_, out);
	glUniform1fARB(loc, transparency_);
}


void RenderHsv::SetHSVBrightness(float d)
{
	hsv_brightness_ = d;
}

void RenderHsv::SetTransparency(float d)
{
	transparency_ = d;
}

void
RenderHsv::Initialize()
{

	const char* shader_prog = 	"uniform float hsv_brightness;					"
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
		"	hueIndex = int(hueRound) % 6;"
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
		"vec2 pos2sv(vec2 pos)"
		"{"
		"	float len = length(pos);	"
		"   float pi = 3.141592653589; "
		"   pos = normalize(pos);		"
		"   vec2 base = vec2(1.0,0.0);		"
		"   float v = dot(pos, base);	"
		"   v = clamp(v, -1.0f, 1.0f); "
		"   v =	 acos(v); "
		"   if (pos.y < 0) v = pi + (pi-v); "
		"   base = vec2(v/(2.0*pi), len);"
		"   return base; "
		"}"
		" void main(void)															"
		"{																			"
		"   vec2 cc = gl_TexCoord[0].st;							"
		"   cc.x = (cc.x - 0.5) * 2.0; "   
		"   cc.y = (cc.y - 0.5) * 2.0; "
		"   vec2 v = pos2sv(cc); "
		"   if (v.y < 0.9) { "
		"		vec3 hsv = HSVtoRGB(vec3(v.x, v.y/0.9, hsv_brightness)); "
		"		gl_FragColor = vec4(clamp(hsv[0], 0.0f, 1.0f), clamp(hsv[1], 0.0f, 1.0f), clamp(hsv[2], 0.0f, 1.0f), 1.0-transparency);		"
		"	} "
		"   else if (v.y < 1.0) {  "
		"		vec3 hsv = HSVtoRGB(vec3(v.x, 1.0, 0.7)); "
		"		gl_FragColor = vec4(clamp(hsv[0], 0.0f, 1.0f), clamp(hsv[1], 0.0f, 1.0f), clamp(hsv[2], 0.0f, 1.0f), 1.0-transparency);		"
		"   }"
		"   else { "
		//"       gl_FragColor = vec4(0.875, 0.875, 0.875, 1.0); "
		"       gl_FragColor = vec4(1.0, 1.0, 1.0, 1.0); "
		"   } "
		"}																			\0";


	SetPixelShaderSource(shader_prog);

	Shader::Initialize();
}


};