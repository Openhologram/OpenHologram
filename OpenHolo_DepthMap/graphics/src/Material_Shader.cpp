
#include "graphics/sys.h"
#include "graphics/glinfo.h"

#include "graphics/gl_extension.h"
#include "graphics/Material_Shader.h"

MaterialShader* kMaterialShader=0;

MaterialShader::MaterialShader(): Shader()
{
}

void MaterialShader::SetHasDiffuse(bool val)
{
	hasDiffuse_ = val;
}
void MaterialShader::SetDiffuseColorTexture(GLuint val)
{
	DiffuseColor_Texture_ = val;
}
void MaterialShader::SetDiffuseColor(float* val)
{
	DiffuseColor_[0] = val[0];
	DiffuseColor_[1] = val[1];
	DiffuseColor_[2] = val[2];
	DiffuseColor_[3] = val[3];
}

void MaterialShader::SetHasBumpMap(bool val)
{
	hasBumMap_ = val;
}

void MaterialShader::SetBumpMapTexture(GLuint val)
{
	BumpMap_Texture_ = val;
}

void MaterialShader::SetNumberofLights(int val)
{
	num_lights_ = val;
}

void MaterialShader::EndShader()
{
	Shader::EndShader();
	glActiveTextureARB(GL_TEXTURE0_ARB);
	glBindTexture(GL_TEXTURE_2D, 0);
	glActiveTextureARB(GL_TEXTURE1_ARB);
	glBindTexture(GL_TEXTURE_2D, 0);

}

void MaterialShader::BeginShader()
{
	Shader::BeginShader();

	GLenum err = glGetError();
	//LOG("Material Shader Error12: %d\n", err);

	int loc;
	loc = glGetUniformLocation(shader_id_, "hasDiffuse");
	glUniform1i(loc, hasDiffuse_);

	loc = glGetUniformLocation(shader_id_, "DiffuseColor");
	glUniform1fv(loc, 4, DiffuseColor_);

	loc = glGetUniformLocation(shader_id_, "DiffuseColorMap");
	glUniform1i(loc, 0);
	glActiveTexture(GL_TEXTURE0_ARB);
	glBindTexture(GL_TEXTURE_2D, DiffuseColor_Texture_);

	//LOG("Material Shader loc: %d\n", loc);

	loc = glGetUniformLocation(shader_id_, "hasBumpMapping");
	glUniform1i(loc, hasBumMap_);
	
	loc = glGetUniformLocation(shader_id_, "BumpMap");
	glUniform1i(loc, 1);
	glActiveTexture(GL_TEXTURE1_ARB);
	glBindTexture(GL_TEXTURE_2D, BumpMap_Texture_);

	loc = glGetUniformLocation(shader_id_, "num_lights");
	glUniform1i(loc, num_lights_);

}

/*
//test version
void MaterialShader::Initialize()
{
	const char* vertex_prog = 
		"varying vec3 normal_VSp;																		"
		"varying vec3 vertex_VSp;																		"
		"void main(void)																			    "
		"{																							    "
		"	gl_TexCoord[0] = gl_MultiTexCoord0;														    "
		"   gl_Position = ftransform();																	"
		"   vec4 vertex_pos = gl_ModelViewMatrix * gl_Vertex;											"
		"   vertex_VSp = vertex_pos.xyz;																"
		"	normal_VSp = normalize(gl_NormalMatrix * gl_Normal);										"
		"}\0";

	const char* pixel_prog =  	
		"uniform bool hasDiffuse;																		"
		"uniform float[4] DiffuseColor;																	"
		"uniform sampler2D DiffuseColorMap;																"
		"varying vec3 normal_VSp;																		"
		"varying vec3 vertex_VSp;																		"
		"void main(void)																			    "
		"{																							    "
		"	vec3 diffuse = vec3(DiffuseColor[0],DiffuseColor[1],DiffuseColor[2]);						"
		"	if (hasDiffuse)																				"
		"		diffuse = pow(texture2D(DiffuseColorMap,  gl_TexCoord[0].st).rgb, 2.2);					"
		"   vec3 L, N, E, R;																			"
		"   float NdotL = 0.0, NdotH = 0.0;																"
		"   N = normalize(normal_VSp);																	"
		"	E = normalize(-vertex_VSp);																"
		"	L = normalize(gl_LightSource[0].position.xyz - vertex_VSp);								"
		"   R = reflect(-L, N);																			"
		"   NdotL = max(dot(N, L), 0.0);																"
		"   if (NdotL >= 0 && max(dot(R, E), 0.0) != 0.0 && gl_FrontMaterial.shininess > 0.0)			"
		"		NdotH = pow(max(dot(R, E), 0.0), gl_FrontMaterial.shininess) ;							"
		"   vec4 ambientT, diffuseT, specularT;															" 
		"	ambientT = gl_LightSource[0].ambient * gl_FrontMaterial.ambient;							"
		"	diffuseT = gl_LightSource[0].diffuse * NdotL * vec4(diffuse, 1.0);							"
		"	specularT = gl_LightSource[0].specular * NdotH * gl_FrontMaterial.specular;					"
		"   vec3 gamma = vec3(1.0/2.2);																	"
		"	vec4 linear_color = gl_FrontLightModelProduct.sceneColor + ambientT + diffuseT + specularT; "
		"   gl_FragColor = vec4(pow(linear_color.rgb, gamma), linear_color.w);							"
		"	gl_FragColor = gl_FrontLightModelProduct.sceneColor + ambientT + diffuseT + specularT;		"
		"}\0";

	SetVertexShaderSource(vertex_prog);
	SetPixelShaderSource(pixel_prog);

	Shader::Initialize();
}
*/

// multiple light version
void MaterialShader::Initialize()
{
	const char* vertex_prog = 
		"uniform bool hasBumpMapping;																	"
		"varying vec3 normal_VSp;																		"
		"varying vec3 vertex_VSp;																		"
		"varying vec3 bump_tangent;																		"
		"varying vec3 bump_binormal;																	"
		"varying vec3 bump_normal;																		"
		"vec3 get_tangent(){						   												    "
		"	vec3 c1 = cross( gl_NormalMatrix * gl_Normal, vec3(0.0, 0.0, 1.0) );					    "
		"   vec3 c2 = cross( gl_NormalMatrix * gl_Normal, vec3(0.0, 1.0, 0.0) );						"
		"   vec3 t;																					    "
		"   if( length(c1)>length(c2) )																    "
		"		t = c1;																				    "
		"	else																					    "
		"		t = c2;																				    "
		"   return normalize(t);																	    "
		"}																							    "
		"void main(void)																			    "
		"{																							    "
		"	gl_TexCoord[0] = gl_MultiTexCoord0;														    "
		"   gl_Position = ftransform();																	"
		"	vertex_VSp = vec3(gl_ModelViewMatrix * gl_Vertex);											"
		"	normal_VSp = normalize(gl_NormalMatrix * gl_Normal);										"
		"   if (hasBumpMapping){																		" 
		"		bump_tangent = get_tangent();															"
		"		bump_binormal = normalize(cross(gl_NormalMatrix * gl_Normal, bump_tangent));			"
		"		bump_normal = normalize(gl_NormalMatrix * gl_Normal);									"
		"   }																							"
		"}\0";

	const char* pixel_prog =
		"uniform int num_lights;																		"
		"uniform bool hasDiffuse;																		"
		"uniform bool hasBumpMapping;																	"
		"uniform float[4] DiffuseColor;																	"
		"uniform sampler2D DiffuseColorMap;																"
		"uniform sampler2D BumpMap;																		"
		"varying vec3 normal_VSp;																		"
		"varying vec3 vertex_VSp;																		"
		"varying vec3 bump_tangent;																		"
		"varying vec3 bump_binormal;																	"
		"varying vec3 bump_normal;																		"
		"vec3 mypow(vec3 a, float b) { return vec3(pow(a[0],b),pow(a[1],b),pow(a[2],b)); } "
		"void ComputeTangentSP(in vec3 light_VSp, in vec3 view_VSp, inout vec3 L, inout vec3 E, inout vec3 N)"
		"{																								"
		"	N = texture2D(BumpMap,  gl_TexCoord[0].st).rgb * 2.0 - 1.0;									"
		"	L.x = dot(bump_tangent,light_VSp);															"
		"	L.y = dot(bump_binormal,light_VSp);															"
		"	L.z = dot(bump_normal,light_VSp);												   			"
		"	L = normalize(L);																			"
		"	E.x = dot(bump_tangent, view_VSp);															"
		"	E.y = dot(bump_binormal, view_VSp);															"
		"	E.z = dot(bump_normal, view_VSp);															"
		"}																								"
		"void ComputeLightSource(in int type, in uint i, inout vec4 ambient, inout vec4 diffuse, inout vec4 specular)"
		"{																								"
		"	vec3 L, E, N, light_VSp, view_VSp;															"
		"   double attenuation = 1.0, spot_attenuation = 0.0;											"
		"	view_VSp = -normalize(vertex_VSp);															"
		"   if (type == 0)	{																			"
		"		light_VSp = normalize(gl_LightSource[i].position.xyz);									"
		"   }else if (type == 1 || type == 2) {															"
		"		light_VSp = gl_LightSource[i].position.xyz - vertex_VSp;								"
		"		double len = length(light_VSp);															"
		"		light_VSp = normalize(light_VSp);														"
		"		attenuation = 1.0 / (gl_LightSource[i].constantAttenuation + gl_LightSource[i].linearAttenuation*len + gl_LightSource[i].quadraticAttenuation*len*len);	"	
		"		if (type == 2) {																		"
		"			float spotDot = max(dot(-light_VSp, gl_LightSource[i].spotDirection), 0.0);			"
		"			if (spotDot >= gl_LightSource[i].spotCosCutoff )	{								"
		"				spotDot = (spotDot - gl_LightSource[i].spotCosCutoff) / (1.0 - gl_LightSource[i].spotCosCutoff );"
		"				if (spotDot != 0 && gl_LightSource[i].spotExponent > 0.0 )						"
		"					spot_attenuation = mypow( spotDot, gl_LightSource[i].spotExponent );			"		
		"				else																			"
		"					spot_attenuation = spotDot;													"
		"			}																					"																				
		"			attenuation *= spot_attenuation;													"
		"		}																						"
		"	}																							"
		"   if (hasBumpMapping)	{																		" 
		"		ComputeTangentSP(light_VSp, view_VSp, L, E, N);											"
		"	}else{																						"
		"		L = light_VSp;																			"
		"		E = view_VSp;																			"
		"		N = normalize(normal_VSp);																"
		"   }																							"
		"	float NdotL = 0.0, NdotH = 0.0;																"
		"   NdotL = max(dot(N, L), 0.0);																"
		"	NdotH = max(dot(reflect(-L, N), E),  0.0);													"
		"   if ( NdotL >= 0.0 && NdotH != 0.0 && gl_FrontMaterial.shininess > 0.0)						"
		"		NdotH = mypow(NdotH, gl_FrontMaterial.shininess);											"
		"   ambient += gl_LightSource[i].ambient * attenuation;											"
		"	diffuse += gl_LightSource[i].diffuse * NdotL * attenuation;									"
		"   specular += gl_LightSource[i].specular * NdotH * attenuation;								" 
		"}																								"
		"void main(void)																			    "
		"{																							    "
		"	vec3 diffuse = vec3(DiffuseColor[0],DiffuseColor[1],DiffuseColor[2]);						"
		"	if (hasDiffuse)																				"
		"		diffuse = mypow(texture2D(DiffuseColorMap,  gl_TexCoord[0].st).rgb, 2.2);					"
		//"		diffuse = texture2D(DiffuseColorMap,  gl_TexCoord[0].st).rgb;							"
		"   vec4 ambientT = vec4(0.0);																	" 
		"   vec4 diffuseT = vec4(0.0);																	"
		"   vec4 specularT = vec4(0.0);																	"
		"   for(int i = 0; i < num_lights; i++)															"
		"	{																							"
		"       if (gl_LightSource[i].position.w == 0.0)												"
		"			ComputeLightSource(0, i, ambientT, diffuseT, specularT);							"
		"       else if ( gl_LightSource[i].spotCutoff == 180.0 )										"
		"			ComputeLightSource(1, i, ambientT, diffuseT, specularT);							"
		"       else																					"
		"			ComputeLightSource(2, i, ambientT, diffuseT, specularT);							"
		"	}																							"
		"	ambientT = ambientT * gl_FrontMaterial.ambient;												"
		"	diffuseT = diffuseT * vec4(diffuse, 1.0);													"
		"	specularT = specularT * gl_FrontMaterial.specular;											"
		"   vec3 gamma = vec3(1.0/2.2);																	"
		"	vec4 linear_color = gl_FrontLightModelProduct.sceneColor + ambientT + diffuseT + specularT; "
		//"   gl_FragColor = vec4(sqrt(linear_color.rgb), 1.0);											"
		"   gl_FragColor = vec4(mypow(linear_color.rgb, gamma), linear_color.w);							"
		//"	gl_FragColor = linear_color;																"
		"}\0";

	SetVertexShaderSource(vertex_prog);
	SetPixelShaderSource(pixel_prog);

	Shader::Initialize();
}

/*
// Single light version - completed!  10/7
void MaterialShader::Initialize()
{
	const char* vertex_prog = 
		"uniform bool hasBumpMapping;																	"
		"varying vec3 lightDir;																			"
		"varying vec3 normalDir;																		"
		"varying vec3 viewDir;																			"
		"varying double attenuation;																		"
		"varying double spoteffect;																		"
		"vec3 get_tangent(){						   												    "
		"	vec3 c1 = cross( gl_NormalMatrix * gl_Normal, vec3(0.0, 0.0, 1.0) );					    "
		"   vec3 c2 = cross( gl_NormalMatrix * gl_Normal, vec3(0.0, 1.0, 0.0) );						"
		"   vec3 t;																					    "
		"   if( length(c1)>length(c2) )																    "
		"		t = c1;																				    "
		"	else																					    "
		"		t = c2;																				    "
		"   return normalize(t);																	    "
		"}																							    "
		"void main(void)																			    "
		"{																							    "
		"	gl_TexCoord[0] = gl_MultiTexCoord0;														    "
		"   gl_Position = ftransform();																	"
		"   vec4 vertex_VSp4 = gl_ModelViewMatrix * gl_Vertex;											"
		"	vec3 vertex_VSp = (vec3(vertex_VSp4)) / vertex_VSp4.w;										"
		"	vec3 normal_VSp = normalize(gl_NormalMatrix * gl_Normal);									"
		"   vec3 light_VSp;																				"
		"	if ( gl_LightSource[0].position.w == 0.0 ){													"
		"		light_VSp = normalize(gl_LightSource[0].position.xyz);									"
		"		attenuation = 1;																		"	
		"		spoteffect = 1;																			"	
		"    }else {																					"
		"		light_VSp = gl_LightSource[0].position.xyz - vertex_VSp;								"
		"		double len = length(light_VSp);															"
		"		light_VSp = normalize(light_VSp);														"
		"		attenuation = 1.0 / (gl_LightSource[0].constantAttenuation + gl_LightSource[0].linearAttenuation*len + gl_LightSource[0].quadraticAttenuation*len*len);"		
		"		spoteffect = 1;																			"		
		"       if ( gl_LightSource[0].spotCutoff != 180.0 ){											"					
		"			float insideCone = max(dot(-light_VSp, normalize(gl_LightSource[0].spotDirection.xyz)), 0.0);"
		"			if (insideCone < cos(radians(gl_LightSource[0].spotCutoff))	)						"
		"				spoteffect = 0.0;																"
		"			else																				"
		"				spoteffect = pow( insideCone, (gl_LightSource[0].spotExponent<=0.0?1:gl_LightSource[0].spotExponent) );	"		
		"       }																						"
		"	}																							"
		"   if (hasBumpMapping==1)																		" 
		"   {																							"
		"		vec3 tangent = get_tangent();															"
		"		vec3 binormal = normalize(cross(gl_NormalMatrix * gl_Normal, tangent));					"
		"		vec3 normal = normalize(gl_NormalMatrix * gl_Normal);									"
		"		lightDir.x = dot(tangent,light_VSp);													"
		"		lightDir.y = dot(binormal,light_VSp);													"
		"		lightDir.z = dot(normal,light_VSp);											    		"
		"		lightDir = normalize(lightDir);															"
		"		vec3 view_VSp = -normalize(vertex_VSp);													"
		"		viewDir.x = dot(tangent, view_VSp);														"
		"		viewDir.y = dot(binormal, view_VSp);													"
		"		viewDir.z = dot(normal, view_VSp);														"
		"	} else {																					"
		"       lightDir = light_VSp;																	"
		//"       normalDir = normalize(normal_VSp - vertex_VSp);											"
		"       normalDir = normalize(normal_VSp);											"
		"		viewDir = -normalize(vertex_VSp);														"
		"   }																							"
		"}\0";

	const char* pixel_prog =  	
		"uniform bool hasDiffuse;																		"
		"uniform bool hasBumpMapping;																	"
		"uniform float[4] DiffuseColor;																	"
		"uniform sampler2D DiffuseColorMap;																"
		"uniform sampler2D BumpMap;																		"
		"varying vec3 lightDir;																			"
		"varying vec3 normalDir;																		"
		"varying vec3 viewDir;																			"
		"varying double attenuation;																	"
		"varying double spoteffect;																		"
		"void main(void)																			    "
		"{																							    "
		"	vec3 diffuse = vec3(DiffuseColor[0],DiffuseColor[1],DiffuseColor[2]);						"
		"	if (hasDiffuse)																				"
		"		diffuse = pow(texture2D(DiffuseColorMap,  gl_TexCoord[0].st).rgb, 2.2);					"
		"   float NdotL = 0.0, NdotH = 0.0;																"
		"	if (hasBumpMapping==1)																		"
		"		normalDir = texture2D(BumpMap,  gl_TexCoord[0].st).rgb * 2.0 - 1.0;						"
		"   else																						"
		"		normalDir = normalize(normalDir);														"
		"  lightDir = normalize(lightDir);																"
		"  viewDir = normalize(viewDir);																"
		"   NdotL = max(dot(normalDir, lightDir), 0.0);													"
		"   if ( NdotL > 0 )																			"
		"		NdotH = pow(max(dot(reflect(-lightDir, normalDir), viewDir),  0.0), (gl_FrontMaterial.shininess<=0.0?1:gl_FrontMaterial.shininess) );"
		"	vec4 ambientT = gl_LightSource[0].ambient * gl_FrontMaterial.ambient;						"
		"	vec4 diffuseT = gl_LightSource[0].diffuse * NdotL * vec4(diffuse.rgb, 1.0);				    "
		"	vec4 specularT = gl_LightSource[0].specular * NdotH * gl_FrontMaterial.specular;			"
		"   vec4 linear_color = gl_FrontLightModelProduct.sceneColor + attenuation * spoteffect * (diffuseT + ambientT + specularT); "
		"   vec3 gamma = vec3(1.0/2.2);																	"
		"   gl_FragColor = vec4( pow( linear_color.xyz, gamma), linear_color.w);						"
		//"	gl_FragColor = gl_FrontLightModelProduct.sceneColor + attenuation * spoteffect * (diffuseT + ambientT + specularT);"
		"}\0";

	SetVertexShaderSource(vertex_prog);
	SetPixelShaderSource(pixel_prog);

	Shader::Initialize();
}

*/
// bump-tangent space, else-view space 10/2
//void MaterialShader::Initialize()
//{
//	const char* vertex_prog = 
//		"uniform bool hasBumpMapping;																	"
//		"varying vec3 lightvec_TSp;																		"
//		"varying vec3 halfvec_TSp;														    			"
//		"varying vec3 light_VSp;																		"
//		"varying vec3 normal_VSp;																		"
//		"varying vec3 vertex_VSp;																		"
//		"varying float attenuation;																		"
//		"varying float spoteffect;																		"
//		"vec3 get_tangent(){						   												    "
//		"	vec3 c1 = cross( gl_NormalMatrix * gl_Normal, vec3(0.0, 0.0, 1.0) );					    "
//		"   vec3 c2 = cross( gl_NormalMatrix * gl_Normal, vec3(0.0, 1.0, 0.0) );						"
//		"   vec3 t;																					    "
//		"   if( length(c1)>length(c2) )																    "
//		"		t = c1;																				    "
//		"	else																					    "
//		"		t = c2;																				    "
//		"   return normalize(t);																	    "
//		"}																							    "
//		"void main(void)																			    "
//		"{																							    "
//		"	gl_TexCoord[0] = gl_MultiTexCoord0;														    "
//		"	gl_Position = gl_ModelViewProjectionMatrix * gl_Vertex;									    "
//		"	vertex_VSp = normalize(vec3(gl_ModelViewMatrix * gl_Vertex));								"
//		"	if ( gl_LightSource[0].position.w == 0.0 ){													"
//		"		light_VSp = normalize(gl_LightSource[0].position.xyz);									"
//		"		attenuation = 1;																		"	
//		"		spoteffect = 1;																			"	
//		"	}else{																						"
//		"		light_VSp = normalize(gl_LightSource[0].position.xyz - vertex_VSp);						"
//		"		float len = length(light_VSp);															"
//		"		attenuation = 1.0 / (gl_LightSource[0].constantAttenuation + gl_LightSource[0].linearAttenuation*len + gl_LightSource[0].quadraticAttenuation*len*len);"		
//		"       float insideCone = max(dot(-light_VSp, normalize(gl_LightSource[0].spotDirection.xyz)), 0.0);"
//		"       if (insideCone < cos(radians(gl_LightSource[0].spotCutoff))	)							"
//		"			spoteffect = 0.0;																	"
//		"		else																					"
//		"			spoteffect = pow( insideCone, gl_LightSource[0].spotExponent);						"		
//		"	}																							"
//		"   if (hasBumpMapping)																			" 
//		"   {																							"
//		"		vec3 tangent = get_tangent();															"
//		"		vec3 binormal = normalize(cross(gl_NormalMatrix * gl_Normal, tangent));					"
//		"		vec3 normal = normalize(gl_NormalMatrix * gl_Normal);									"
//		"		lightvec_TSp.x = dot(tangent,light_VSp);												"
//		"		lightvec_TSp.y = dot(binormal,light_VSp);												"
//		"		lightvec_TSp.z = dot(normal,light_VSp);										    		"
//		"		lightvec_TSp = normalize(lightvec_TSp);													"
//		"		vec3 halfDir_VSp = normalize(vertex_VSp + light_VSp);									"
//		"		halfvec_TSp.x = dot(tangent, halfDir_VSp);												"
//		"		halfvec_TSp.y = dot(binormal, halfDir_VSp);												"
//		"		halfvec_TSp.z = dot(normal, halfDir_VSp);												"
//		"	} else {																					"
//		"		normal_VSp = normalize(gl_NormalMatrix * gl_Normal);									"
//		"       normal_VSp = normalize(normal_VSp - vertex_VSp);										"
//		"   }																							"
//		"}\0";
//
//	const char* pixel_prog =  	
//		"uniform bool hasDiffuse;																		"
//		"uniform bool hasBumpMapping;																	"
//		"uniform float[4] DiffuseColor;																	"
//		"uniform sampler2D DiffuseColorMap;																"
//		"uniform sampler2D BumpMap;																		"
//		"varying vec3 lightvec_TSp;																		"
//		"varying vec3 halfvec_TSp;														    			"
//		"varying vec3 light_VSp;																		"
//		"varying vec3 normal_VSp;																		"
//		"varying vec3 vertex_VSp;																		"
//		"varying float attenuation;																		"
//		"varying float spoteffect;																		"
//		"void main(void)																			    "
//		"{																							    "
//		"	vec3 diffuse = vec3(DiffuseColor[0],DiffuseColor[1],DiffuseColor[2]);						"
//		"	if (hasDiffuse)																				"
//		"		diffuse = texture2D(DiffuseColorMap,  gl_TexCoord[0].st).rgb;							"
//		"	vec3 normalDir, viewDir, lightDir;															"
//		"   float NdotL = 0.0, NdotH = 0.0;																"
//		"	if (hasBumpMapping)																			"
//		"	{																							"
//		"		normalDir = texture2D(BumpMap,  gl_TexCoord[0].st).rgb * 2.0 - 1.0;						"
//		"		NdotL = max(dot(normalDir, lightvec_TSp), 0.0);											"
//		"		if (NdotL > 0)																			"
//		"			NdotH = pow(max(dot(normalDir, halfvec_TSp ), 0.0), gl_FrontMaterial.shininess);	"	
//		"   } else {																					"
//		"		normalDir = normalize(normal_VSp);														"
//		"		viewDir = -normalize(vertex_VSp);														"
//		"		lightDir = normalize(light_VSp);														"
//		"       NdotL = max(dot(normalDir, lightDir), 0.0);												"
//		"       if ( NdotL > 0 )																		"
//		"			NdotH = pow(max(dot(reflect(-lightDir, normalDir), viewDir),  0.0), gl_FrontMaterial.shininess);"
//		"   }																							"
//		"	vec4 ambientT = gl_LightSource[0].ambient * gl_FrontMaterial.ambient;						"
//		"	vec4 diffuseT = gl_LightSource[0].diffuse * NdotL * vec4(diffuse.rgb, 1.0);				    "
//		"	vec4 specularT = gl_LightSource[0].specular * NdotH * gl_FrontMaterial.specular;			"
//		"	gl_FragColor = gl_FrontLightModelProduct.sceneColor + attenuation * spoteffect * (diffuseT + ambientT + specularT);"
//		"}\0";
//
//	SetVertexShaderSource(vertex_prog);
//	SetPixelShaderSource(pixel_prog);
//
//	Shader::Initialize();
//}

/// world space version - 9/24

//void MaterialShader::Initialize()
//{
//	const char* vertex_prog = 
//		"uniform mat4 modelMatrix;																		"
//		"uniform mat4 modelMatrix_Inv;																	"
//		"uniform bool hasBumpMapping;																	"
//		"varying vec3 lightvec_TSp;																		"
//		"varying vec3 halfvec_TSp;														    			"
//		"varying vec4 vpos_WSp;																			"
//		"varying vec3 normal_WSp;																		"
//		"vec3 get_tangent(){						   												    "
//		"	vec3 c1 = cross( gl_NormalMatrix * gl_Normal, vec3(0.0, 0.0, 1.0) );					    "
//		"   vec3 c2 = cross( gl_NormalMatrix * gl_Normal, vec3(0.0, 1.0, 0.0) );						"
//		"   vec3 t;																					    "
//		"   if( length(c1)>length(c2) )																    "
//		"		t = c1;																				    "
//		"	else																					    "
//		"		t = c2;																				    "
//		"   return normalize(t);																	    "
//		"}																							    "
//		"void main(void)																			    "
//		"{																							    "
//		"	gl_TexCoord[0] = gl_MultiTexCoord0;														    "
//		"	gl_Position = gl_ModelViewProjectionMatrix * gl_Vertex;									    "
//		"   if (hasBumpMapping)																			" 
//		"   {																							"
//		"		vec3 tangent = get_tangent();															"
//		"		vec3 binormal = normalize(cross(gl_NormalMatrix * gl_Normal, tangent));					"
//		"		vec3 normal = normalize(gl_NormalMatrix * gl_Normal);									"
//		"		vec3 vertex_VSp = vec3(gl_ModelViewMatrix * gl_Vertex);									"
//		"		vec3 lightDir_VSp;																		"
//		"		if ( gl_LightSource[0].position.w == 0.0 )												"
//		"			lightDir_VSp = normalize(gl_LightSource[0].position.xyz);							"
//		"		else																					"
//		"			lightDir_VSp = normalize(gl_LightSource[0].position.xyz - vertex_VSp);				"
//		"		lightvec_TSp.x = dot(tangent,lightDir_VSp);												"
//		"		lightvec_TSp.y = dot(binormal,lightDir_VSp);											"
//		"		lightvec_TSp.z = dot(normal,lightDir_VSp);										    	"
//		"		lightvec_TSp = normalize(lightvec_TSp);													"
//		"		vec3 halfDir_VSp = normalize(vertex_VSp + lightDir_VSp);								"
//		"		halfvec_TSp.x = dot(tangent, halfDir_VSp);												"
//		"		halfvec_TSp.y = dot(binormal, halfDir_VSp);												"
//		"		halfvec_TSp.z = dot(normal, halfDir_VSp);												"
//		"	} else {																					"
//		"		vpos_WSp = normalize(modelMatrix * gl_Vertex);											"
//		"       normal_WSp = normalize(vec3(vec4(gl_Normal,0.0) * modelMatrix));						"
//		"   }																							"
//		"}\0";
//
//	const char* pixel_prog =  	
//		"uniform mat4 modelMatrix;																		"
//		"uniform vec4 lightpos_WSp;																		"
//		"uniform vec3 camerapos_WSp;																	"
//		"uniform bool hasDiffuse;																		"
//		"uniform bool hasBumpMapping;																	"
//		"uniform float[4] DiffuseColor;																	"
//		"uniform sampler2D DiffuseColorMap;																"
//		"uniform sampler2D BumpMap;																		"
//		"varying vec3 lightvec_TSp;																		"
//		"varying vec3 halfvec_TSp;														    			"
//		"varying vec4 vpos_WSp;																			"
//		"varying vec3 normal_WSp;																		"
//		"void main(void)																			    "
//		"{																							    "
//		"	vec3 diffuse = vec3(DiffuseColor[0],DiffuseColor[1],DiffuseColor[2]);						"
//		"	if (hasDiffuse)																				"
//		"		diffuse = texture2D(DiffuseColorMap,  gl_TexCoord[0].st).rgb;							"
//		"	vec3 normalDir;																				"
//		"   float NdotL = 0.0, NdotH = 0.0, attenuation = 1.0;											"
//		"		vec3 lightDir_WSp;																		"
//		"	if (hasBumpMapping)																			"
//		"	{																							"
//		"		normalDir = texture2D(BumpMap,  gl_TexCoord[0].st).rgb * 2.0 - 1.0;						"
//		"		NdotL = max(dot(normalDir, lightvec_TSp), 0.0);											"
//		"		if (NdotL > 0)																			"
//		"			NdotH = pow(max(dot(normalDir, halfvec_TSp ), 0.0), gl_FrontMaterial.shininess);	"	
//		"   } else {																					"
//		"		normalDir = normalize(normal_WSp);														"
//		"		vec3 viewDir_WSp = normalize(camerapos_WSp - vec3(vpos_WSp));							"
//		//"		vec3 lightDir_WSp;																		"
//		"		lightDir_WSp = normalize(modelMatrix * vec4(lightDir_WSp,1));							"
//		"		if ( lightpos_WSp.w == 0.0){															"
//		"			attenuation = 1.0;			                										"
//		"			lightDir_WSp = normalize(vec3(lightpos_WSp));										"
//		"		} else {																				"
//		"			vec3 vertexToLightSource = vec3(lightpos_WSp - vpos_WSp);							"
//		"			lightDir_WSp = normalize(vertexToLightSource);										"
//		"			attenuation = 1.0 / ( length(vertexToLightSource) );								"
//		"       }																						"
//		"       NdotL = max(dot(normalDir, lightDir_WSp), 0.0);											"
//		"       if ( NdotL > 0 )																		"
//		"			NdotH = pow(max(dot(reflect(-lightDir_WSp, normalDir),viewDir_WSp),  0.0), gl_FrontMaterial.shininess);"
//		"   }																							"
//		"	vec4 ambientT = gl_LightSource[0].ambient * gl_FrontMaterial.ambient;						"
//		"	vec4 diffuseT = gl_LightSource[0].diffuse * NdotL * vec4(diffuse.rgb, 1.0);				    "
//		"	vec4 specularT = gl_LightSource[0].specular * NdotH * gl_FrontMaterial.specular;			"
//		//"	gl_FragColor = gl_FrontLightModelProduct.sceneColor + attenuation * (diffuseT + ambientT + specularT);	"
//		"	gl_FragColor = vec4(lightDir_WSp,1.0);	"
//		"}\0";
//
//	SetVertexShaderSource(vertex_prog);
//	SetPixelShaderSource(pixel_prog);
//
//	Shader::Initialize();
//}


// version 2
//void MaterialShader::Initialize()
//{
//	const char* vertex_prog = 
//		"uniform bool hasBumpMapping;																	"
//		"varying vec3 lightvec;																		    "
//		"varying vec3 normalvec;																		"
//		"varying vec3 halfvec;														    			    "
//		"vec3 get_tangent(){						   												    "
//		"	vec3 c1 = cross( gl_NormalMatrix * gl_Normal, vec3(0.0, 0.0, 1.0) );					    "
//		"   vec3 c2 = cross( gl_NormalMatrix * gl_Normal, vec3(0.0, 1.0, 0.0) );						"
//		"   vec3 t;																					    "
//		"   if( length(c1)>length(c2) )																    "
//		"		t = c1;																				    "
//		"	else																					    "
//		"		t = c2;																				    "
//		"   return normalize(t);																	    "
//		"}																							    "
//		"void main(void)																			    "
//		"{																							    "
//		"	gl_TexCoord[0] = gl_MultiTexCoord0;														    "
//		"	gl_Position = gl_ModelViewProjectionMatrix * gl_Vertex;									    "
//		"   vec3 tangent = get_tangent();															    "
//		"   vec3 binormal = normalize(cross(gl_NormalMatrix * gl_Normal, tangent));					    "
//		"	vec3 normal = normalize(gl_NormalMatrix * gl_Normal);									    "
//		"   vec3 vertex = vec3(gl_ModelViewMatrix * gl_Vertex);										    "
//		"	vec3 lightDir;																			    "
//		"   if ( gl_LightSource[0].position.w == 0.0 )												    "
//		"		lightDir = normalize(gl_LightSource[0].position.xyz);								    "
//		"   else																					    "
//		"		lightDir = normalize(gl_LightSource[0].position.xyz - vertex);						    "
//		"   if (hasBumpMapping)	{																		"
//		"		lightvec.x = dot(tangent,lightDir);														"
//		"		lightvec.y = dot(binormal,lightDir);													"
//		"		lightvec.z = dot(normal,lightDir);										    			"
//		"		lightvec = normalize(lightvec);															"
//		"		vec3 halfDir = normalize(vertex + lightDir);											"
//		"		halfvec.x = dot(tangent, halfDir);														"
//		"		halfvec.y = dot(binormal, halfDir);														"
//		"		halfvec.z = dot(normal, halfDir);														"
//		"	} else {																					"
//		"		lightvec = normalize(lightDir);															"
//		"       halfvec = normalize(vertex + lightDir);															"
//		"       normalvec = normal;  																	"
//		"   }																							"
//		"}\0";
//
//	const char* pixel_prog = 	
//		"uniform bool hasDiffuse;																		"
//		//"uniform bool hasTransparency;																	"
//		//"uniform bool hasTransparentColor;																"
//		//"uniform bool hasAmbientColor;																	"
//		//"uniform bool hasEmissiveColor;																	"
//		"uniform bool hasBumpMapping;																	"
//		//"uniform bool hasShininess;																		"
//		//"uniform bool hasSpecularColor;																	"
//		//"uniform bool hasReflectivity;																	"
//		//"uniform bool hasReflectedColor;																"
//		"uniform float[4] DiffuseColor;																		"
//		//"uniform float Transparency;																	"
//		//"uniform vec4 TransparentColor;																	"
//		//"uniform vec4 AmbientColor;																		"
//		//"uniform vec4 EmissiveColor;																	"
//		//"uniform float Shininess;																		"
//		//"uniform vec4 SpecularColor;																	"
//		//"uniform float Reflectivity;																	"
//		//"uniform vec4 ReflectedColor;																	"
//		"uniform sampler2D DiffuseColorMap;																"
//		//"uniform sampler2D TransparencyMap;																"
//		//"uniform sampler2D TransparentColorMap;															"
//		//"uniform sampler2D AmbientColorMap;																"
//		//"uniform sampler2D EmissiveColorMap;															"
//		"uniform sampler2D BumpMap;																		"
//		//"uniform sampler2D ShininessMap;																"
//		//"uniform sampler2D SpecularColorMap;															"
//		//"uniform sampler2D ReflectivityMap;																"
//		//"uniform sampler2D ReflectedColorMap;															"
//		"varying vec3 lightvec;																		    "
//		"varying vec3 normalvec;																		"
//		"varying vec3 halfvec;														    			    "
//		"void main(void)																			    "
//		"{																							    "
//		" vec3 diffuse = vec3(DiffuseColor[0],DiffuseColor[1],DiffuseColor[2]);							"
//		" if (hasDiffuse)																				"
//		"    diffuse = texture2D(DiffuseColorMap,  gl_TexCoord[0].st).rgb;								"
//		" vec3 normal = normalize(normalvec);																		"
//		" if (hasBumpMapping)																			"
//		"	 normal = texture2D(BumpMap,  gl_TexCoord[0].st).rgb * 2.0 - 1.0;							"
//		" float NdotL = max(dot(normal, lightvec), 0.0);											    "
//		" float NdotH = 0.0;																		    "
//		" if (NdotL > 0)																			    "
//		"   NdotH = pow(max(dot(normal, halfvec ), 0.0), gl_FrontMaterial.shininess);					"	
//		" vec4 ambientT = gl_LightSource[0].ambient * gl_FrontMaterial.ambient;						    "
//		" vec4 diffuseT = gl_LightSource[0].diffuse * NdotL * vec4(diffuse.rgb, 1.0);				    "
//		" vec4 specularT = gl_LightSource[0].specular * NdotH * gl_FrontMaterial.specular;				"
//		" gl_FragColor = gl_FrontLightModelProduct.sceneColor + diffuseT + ambientT + specularT;		"
//		"}\0";
//
//	SetVertexShaderSource(vertex_prog);
//	SetPixelShaderSource(pixel_prog);
//
//	Shader::Initialize();
//}


// version 1
//void MaterialShader::Initialize()
//{
//	const char* vertex_prog = 
//		"varying vec3 lightvec;														    "
//		"vec3 get_tangent(){						   									"
//		"	vec3 c1 = cross( gl_NormalMatrix * gl_Normal, vec3(0.0, 0.0, 1.0) );						"
//		"   vec3 c2 = cross( gl_NormalMatrix * gl_Normal, vec3(0.0, 1.0, 0.0) );						"
//		"   vec3 t;																		"
//		"   if( length(c1)>length(c2) )													"
//		"		t = c1;																	"
//		"	else																		"
//		"		t = c2;																	"
//		"   return normalize(t);														"
//		"}																				"
//		"void main(void)																"
//		"{																				"
//		"	gl_TexCoord[0] = gl_MultiTexCoord0;											"
//		"	gl_Position = gl_ModelViewProjectionMatrix * gl_Vertex;						"
//		"   vec3 tangent = get_tangent();												"
//		"   vec3 binormal = normalize(cross(gl_NormalMatrix * gl_Normal, tangent));		"
//		"	vec3 normal = normalize(gl_NormalMatrix * gl_Normal);						"
//		"   vec3 vertex = vec3(gl_ModelViewMatrix * gl_Vertex);							"
//		"	vec3 lightDir;																"
//		"   if ( gl_LightSource[0].position.w == 0.0 )									"
//		"		lightDir = normalize(gl_LightSource[0].position.xyz);					"
//		"   else																		"
//		"		lightDir = normalize(gl_LightSource[0].position.xyz - vertex);			"
//		"   lightvec.x = dot(tangent,lightDir);											"
//		"   lightvec.y = dot(binormal,lightDir);										"
//		"   lightvec.z = dot(normal,lightDir);										"
//		"   lightvec = normalize(lightvec);												"
//		"}\0";
//
//	const char* pixel_prog = 	
//		"uniform sampler2D normalMap;													"
//		"uniform sampler2D colorMap;													"
//		"varying vec3 lightvec;															"
//		"void main(void)																"
//		"{																				"
//		" vec3 normal = texture2D(normalMap,  gl_TexCoord[0].st).rgb * 2.0 - 1.0;		"
//		" vec3 color = texture2D(colorMap,  gl_TexCoord[0].st).rgb;						"
//		" float nxDir = max(dot(normal, lightvec), 0.0);								"
//		" vec4 diffuse = gl_LightSource[0].diffuse * nxDir;								"
//		" gl_FragColor += (gl_LightSource[0].ambient + diffuse) * vec4(color.rgb, 1.0); "
//		"}\0";
//
//	SetVertexShaderSource(vertex_prog);
//	SetPixelShaderSource(pixel_prog);
//
//	Shader::Initialize();
//}

