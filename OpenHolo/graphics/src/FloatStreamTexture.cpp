#include "graphics/FloatStreamTexture.h"

namespace graphics {

	FloatStreamTexture::FloatStreamTexture()
		: width_(640), height_(479),bound_(false)
	{
		glEnable(GL_TEXTURE_2D);
		glGenTextures(1, &texture_id_);
		Bind();
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP);
		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB32F, width_, height_, 0, GL_RGBA, GL_FLOAT, (void*)0);
		Unbind();
	}

	FloatStreamTexture::~FloatStreamTexture()
	{
		glDeleteTextures(1, &texture_id_);
	}

	int  FloatStreamTexture::TextureDataSize() const
	{
		return width_*height_*4*sizeof(float);
	}

	real FloatStreamTexture::AspectRatio() const
	{
		return (real)width_/(real)height_;
	}
	void FloatStreamTexture::Bind()
	{
		if (!bound_) glBindTexture(GL_TEXTURE_2D, texture_id_);
		bound_ = true;
	}

	void FloatStreamTexture::Unbind()
	{
		if (bound_) {
			glBindTexture(GL_TEXTURE_2D, 0);
			bound_ = false;
		}
	}

	void FloatStreamTexture::Resize(int w, int h)
	{
		width_ = w;
		height_ = h;

		Bind();
		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB32F, width_, height_, 0, GL_RGBA, GL_FLOAT, 0);
		Unbind();
	}

	void FloatStreamTexture::Copy(uchar* data)
	{
		if (!bound_) glBindTexture(GL_TEXTURE_2D, texture_id_);
		glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, width_, height_, GL_RGBA, GL_FLOAT, (void*)data);
	}

	void FloatStreamTexture::Connect(int pbo_position)
	{
		if (!bound_) glBindTexture(GL_TEXTURE_2D, texture_id_);
		glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, width_, height_, GL_RGBA, GL_FLOAT, (void*)pbo_position);
	}

};