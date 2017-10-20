#ifndef __material_shader_h
#define __material_shader_h

#include <graphics/vec.h>
#include "graphics/Shader.h"





class MaterialShader: public graphics::Shader {

public:

	MaterialShader();

	virtual void Initialize();

	virtual void BeginShader();
	virtual void EndShader();

	void SetHasBumpMap(bool val);
	void SetBumpMapTexture(unsigned int val);

	void SetHasDiffuse(bool val);
	void SetDiffuseColorTexture(unsigned int val);
	void SetDiffuseColor(float* val);
	void SetNumberofLights(int val);

protected:

	GLboolean   hasBumMap_;
	GLuint		BumpMap_Texture_;

	GLboolean	hasDiffuse_;
	GLfloat		DiffuseColor_[4];																		
	GLuint		DiffuseColor_Texture_;

	GLint		num_lights_;

};

extern MaterialShader* kMaterialShader;



#endif