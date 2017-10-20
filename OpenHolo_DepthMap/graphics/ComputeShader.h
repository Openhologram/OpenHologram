#ifndef __COMPUTE_SHADER_H
#define __COMPUTE_SHADER_H


#include "graphics/sys.h"

namespace graphics {

	class ComputeShader {

	public:

		ComputeShader();

		virtual ~ComputeShader();

		virtual void Initialize();

		virtual void BeginShader();

		virtual void EndShader();

		void SetShaderSource(const char* source);

		void SetShaderSourceWithFile(const char* fname);

	protected:

		unsigned int LoadShader(const char* source);

	protected:

		unsigned int program_id_;

		unsigned int shader_handle_;

		char* shader_src_;
	};


}

#endif