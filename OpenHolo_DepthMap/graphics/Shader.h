#ifndef __Shader_h
#define __Shader_h

#include "graphics/sys.h"

namespace graphics {

class Shader {

public:

	Shader ();
	 
	virtual ~Shader();

	virtual void Initialize();

	virtual void BeginShader();

	virtual void EndShader();

	void SetVertexShaderSource(const char* source);

	void SetPixelShaderSource(const char* source);

protected:

	unsigned int LoadShader(const char* source, unsigned int type);

protected:

	unsigned int shader_id_;

	unsigned int vertex_shader_handle_;

	unsigned int fragment_shader_handle_;

	char* vertex_shader_src_;

	char* pixel_shader_src_;
};

};
#endif