#ifndef YUV_TO_RGB_SHADER_H_
#define YUV_TO_RGB_SHADER_H_

#include "graphics/Shader.h"

namespace graphics
{
	class YUVtoRGB_Shader : public Shader
	{
	public:

		YUVtoRGB_Shader();
		void BeginShader();

		virtual void Initialize();

		int GetTextureHeight(void) const;
		void SetTextureHeight(int i);

		unsigned int GetUTex(void) const;
		void SetUTex(unsigned int);

		unsigned int GetVTex(void) const;
		void SetVTex(unsigned int);

		unsigned int GetYTex(void) const;
		void SetYTex(unsigned int);

		unsigned int GetShaderHandle(void) { return shader_id_; }

	protected:
		int texture_height_;
		unsigned int ytex_, utex_, vtex_;
	};
}

#endif