#include	"graphics/texture_buffer.h"

namespace graphics {

TextureBuffer::TextureBuffer()
	: width_(100), height_(100),  rbo_id_(0), fbo_id_(0), texture_id_(0), pbo_id_(0)
{
}


TextureBuffer::~TextureBuffer()
{
	if (fbo_id_) {
		glDeleteFramebuffersEXT(1, &fbo_id_);
	}

	if (rbo_id_) {
		glDeleteRenderbuffersEXT(1, &rbo_id_);
	}

	if (texture_id_)
		glDeleteTextures(1, &texture_id_);

	if (pbo_id_) {
		glDeleteBuffersARB(1, &pbo_id_);
	}
}

void 
TextureBuffer::GenerateObjects(uint mag_filter, uint min_filter)
{
	glGenTextures(1, &texture_id_);

	glBindTexture(GL_TEXTURE_2D, texture_id_);
	glTexEnvf(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_REPLACE);
	glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, mag_filter);
	glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, min_filter);
	glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP);
	glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, width_, height_, 0,
		GL_RGBA, GL_UNSIGNED_BYTE, 0);
	glBindTexture(GL_TEXTURE_2D, 0);

	glGenRenderbuffersEXT(1, &rbo_id_);
	glBindRenderbufferEXT(GL_RENDERBUFFER_EXT, rbo_id_);
	glRenderbufferStorageEXT(GL_RENDERBUFFER, GL_DEPTH_COMPONENT,
		width_, height_);
	glBindRenderbufferEXT(GL_RENDERBUFFER, 0);

	glGenFramebuffersEXT(1, &fbo_id_);
	glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, fbo_id_);
	

	// attach the texture to FBO color attachment point
	glFramebufferTexture2DEXT(GL_FRAMEBUFFER_EXT, GL_COLOR_ATTACHMENT0_EXT,
							  GL_TEXTURE_2D, texture_id_, 0);
	glFramebufferRenderbufferEXT(GL_FRAMEBUFFER_EXT, GL_DEPTH_ATTACHMENT_EXT,
		GL_RENDERBUFFER_EXT, rbo_id_);


	glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, 0);
	glBindRenderbufferEXT(GL_RENDERBUFFER_EXT, 0);
	
	
	
}

void
TextureBuffer::Resize(int w, int h)
{
	if (width_ == w && height_ == h) return;

	width_ = w; height_ = h;

	glBindTexture(GL_TEXTURE_2D, texture_id_);
	glTexEnvf(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_REPLACE);
	glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP);
	glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, width_, height_, 0,
		GL_RGBA, GL_UNSIGNED_BYTE, 0);
	glBindTexture(GL_TEXTURE_2D, 0);

	glBindRenderbufferEXT(GL_RENDERBUFFER_EXT, rbo_id_);
	glRenderbufferStorageEXT(GL_RENDERBUFFER, GL_DEPTH_COMPONENT,
		width_, height_);
	glBindRenderbufferEXT(GL_RENDERBUFFER, 0);

	glBindRenderbufferEXT(GL_RENDERBUFFER_EXT, 0);
	glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, 0);
}

void TextureBuffer::ReadData(void* ptr, uint pbo)
{
	if (!pbo) glGenBuffersARB(1, &pbo);
	glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, fbo_id_);
	glBindBufferARB(GL_PIXEL_PACK_BUFFER_ARB, pbo);
	

	glBufferDataARB(GL_PIXEL_PACK_BUFFER_ARB, width_* height_*4, 0, GL_STREAM_READ_ARB);
	glReadPixels(0, 0, width_, height_, GL_RGBA, GL_UNSIGNED_BYTE, 0);

	GLubyte* src = (GLubyte*)glMapBufferARB(GL_PIXEL_PACK_BUFFER_ARB, GL_READ_ONLY_ARB);
	if(src)
	{
		memcpy(ptr, src, width_* height_*4);
		glUnmapBufferARB(GL_PIXEL_UNPACK_BUFFER_ARB); // release the mapped buffer
	}

	glBindBufferARB(GL_PIXEL_PACK_BUFFER_ARB, 0);
	glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, 0);
}

void TextureBuffer::ReadData(void* ptr)
{
	
	if (!pbo_id_) glGenBuffersARB(1, &pbo_id_);
	glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, fbo_id_);
	glBindBufferARB(GL_PIXEL_PACK_BUFFER_ARB, pbo_id_);
	

	glBufferDataARB(GL_PIXEL_PACK_BUFFER_ARB, width_* height_*4, 0, GL_STREAM_READ_ARB);
	glReadPixels(0, 0, width_, height_, GL_RGBA, GL_UNSIGNED_BYTE, 0);

	GLubyte* src = (GLubyte*)glMapBufferARB(GL_PIXEL_PACK_BUFFER_ARB, GL_READ_ONLY_ARB);
	if(src)
	{
		memcpy(ptr, src, width_* height_*4);
		glUnmapBufferARB(GL_PIXEL_UNPACK_BUFFER_ARB); // release the mapped buffer
	}

	glBindBufferARB(GL_PIXEL_PACK_BUFFER_ARB, 0);
	glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, 0);

}
void TextureBuffer::BeginDraw()
{
	glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, fbo_id_);

	// clear buffers
	//glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
}


void TextureBuffer::EndDraw()
{
	// unbind FBO
	glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, 0);
}


};