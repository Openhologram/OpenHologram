
#include "graphics/sys.h"
#include "graphics/glinfo.h"

#include "graphics/gl_extension.h"
#include "graphics/displacement_map.h"


namespace graphics {


DisplacementMap::DisplacementMap(): Shader(), max_height_(1.0)
{
}

void
DisplacementMap::SetDisplacementTexture(GLint val)
{
	displacement_texture_ = val;
	color_texture_ = val;
}

void
DisplacementMap::SetColorTexture(GLint val)
{
	color_texture_ = val;
}

void DisplacementMap::SetMaxHeight(float val)
{
	max_height_ = val;
}

void 
DisplacementMap::EndShader()
{
	Shader::EndShader();
	glActiveTextureARB(0);
	glBindTexture(GL_TEXTURE_2D, 0);
}

void
DisplacementMap::BeginShader()
{
	Shader::BeginShader();


	texture_unit_id_ = glGetUniformLocationARB(shader_id_, "displacementMap");
	colormap_unit_id_ = glGetUniformLocationARB(shader_id_, "colorMap");
	max_height_id_ = glGetUniformLocationARB(shader_id_, "max_height");

	glUniform1iARB(texture_unit_id_, 0);
	glUniform1iARB(colormap_unit_id_, 1);
	glUniform1fARB(max_height_id_, max_height_);

	glActiveTextureARB(GL_TEXTURE0_ARB);
	
	glEnable(GL_TEXTURE_2D);
	glBindTexture(GL_TEXTURE_2D, displacement_texture_);
	glActiveTextureARB(GL_TEXTURE1_ARB);
	glBindTexture(GL_TEXTURE_2D, color_texture_);
}

void
DisplacementMap::Initialize()
{
	const char* vertex_prog = "uniform sampler2D displacementMap;"
		"uniform float max_height; "
		"void main(void)"
		"{"
		"	vec4 newVertexPos; "
		"	vec4 dv; "
		"	float df; "
		"	gl_TexCoord[0].xy = gl_MultiTexCoord0.xy; "
		"	dv = texture2D(displacementMap, gl_MultiTexCoord0.xy);  "
		"	df = max_height*dv.x; "
		"	newVertexPos = vec4(vec3(0,0,1)*df, 0.0) + gl_Vertex;"
		"	gl_Position = gl_ModelViewProjectionMatrix * newVertexPos; "
		"}\0";

	const char* pixel_prog = 	"uniform sampler2D colorMap;															"
		"void main(void)													"
		"{																	"
		" gl_FragColor = texture2D(colorMap, gl_TexCoord[0].xy); "
		"}																	\0";

	SetVertexShaderSource(vertex_prog);
	SetPixelShaderSource(pixel_prog);

	Shader::Initialize();
}

};