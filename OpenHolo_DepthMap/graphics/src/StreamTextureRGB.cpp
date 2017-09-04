#include "graphics/StreamTextureRGB.h"

namespace graphics {

StreamTextureRGB::StreamTextureRGB()
	: width_(640), height_(479),bound_(false)
{
	glEnable(GL_TEXTURE_2D);
	glGenTextures(1, &texture_id_);
	Bind();
	glTexEnvf(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_REPLACE);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, width_, height_, 0, GL_RGBA, GL_UNSIGNED_BYTE, (void*)0);
	Unbind();
}

StreamTextureRGB::~StreamTextureRGB()
{
	glDeleteTextures(1, &texture_id_);
}

int  StreamTextureRGB::TextureDataSize() const
{
	return width_*height_*4;
}

real StreamTextureRGB::AspectRatio() const
{
	return (real)width_/(real)height_;
}
void StreamTextureRGB::Bind()
{
	glBindTexture(GL_TEXTURE_2D, texture_id_);
	bound_ = true;
}

void StreamTextureRGB::Unbind()
{

	glBindTexture(GL_TEXTURE_2D, 0);
	bound_ = false;

}

void StreamTextureRGB::Resize(int w, int h)
{
	if (width_ == w && height_ == h) return;

	width_ = w;
	height_ = h;

	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, width_, height_, 0, GL_RGBA, GL_UNSIGNED_BYTE, 0);
}

void StreamTextureRGB::Copy(uchar* data)
{
	glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, width_, height_, GL_RGBA, GL_UNSIGNED_BYTE, (void*)data);
}

void StreamTextureRGB::Connect(int pbo_position)
{
	glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, width_, height_, GL_RGBA, GL_UNSIGNED_BYTE, (void*)pbo_position);
}

};