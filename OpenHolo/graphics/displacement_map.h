#ifndef __displacement_map_h
#define __displacement_map_h


#include "graphics/Shader.h"

namespace graphics {

class DisplacementMap: public Shader {

public:

	DisplacementMap();

	virtual void Initialize();

	virtual void BeginShader();
	virtual void EndShader();

	void SetDisplacementTexture(GLint val);
	void SetColorTexture(GLint val);
	void SetMaxHeight(float val);

protected:

	GLint	texture_unit_id_;
	GLint	colormap_unit_id_;
	GLint	max_height_id_;
	float	max_height_;
	GLint	displacement_texture_;
	GLint	color_texture_;

};

};

#endif