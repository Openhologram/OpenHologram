#include	"graphics/render_buffer.h"

namespace graphics {

RenderBuffer::RenderBuffer()
	: width_(0), height_(0),  dbo_id_(0), rbo_id_(0), fbo_id_(0), pbo_id_(0), depth_test_(true)
{
}

RenderIntensityBuffer::RenderIntensityBuffer()
	: RenderBuffer()
{

}

RenderIntensityBuffer::~RenderIntensityBuffer()
{

}

RenderBuffer::~RenderBuffer()
{
	if (fbo_id_) {
		glDeleteFramebuffersEXT(1, &fbo_id_);
	}

	if (rbo_id_) {
		glDeleteRenderbuffersEXT(1, &rbo_id_);
	}

	if (dbo_id_) {
		glDeleteRenderbuffersEXT(1, &dbo_id_);
	}

	if (pbo_id_) {
		glDeleteBuffersARB(1, &pbo_id_);
	}
}

void 
RenderBuffer::GenerateObjects()
{
	glGenRenderbuffersEXT(1, &rbo_id_);
	if (depth_test_) glGenRenderbuffersEXT(1, &dbo_id_);
	glGenFramebuffersEXT(1, &fbo_id_);

	glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, fbo_id_);
	glBindRenderbufferEXT(GL_RENDERBUFFER_EXT, rbo_id_);
	

	// attach the texture to FBO color attachment point
	glFramebufferRenderbufferEXT(GL_FRAMEBUFFER_EXT, GL_COLOR_ATTACHMENT0_EXT,
							  GL_RENDERBUFFER_EXT, rbo_id_);

	// attach the renderbuffer to depth attachment point
	if (depth_test_) {
		glBindRenderbufferEXT(GL_RENDERBUFFER_EXT, dbo_id_);
		glFramebufferRenderbufferEXT(GL_FRAMEBUFFER_EXT, GL_DEPTH_ATTACHMENT_EXT,
								 GL_RENDERBUFFER_EXT, dbo_id_);
	}

	glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, 0);
	glBindRenderbufferEXT(GL_RENDERBUFFER_EXT, 0);

	glGenBuffersARB(1, &pbo_id_);
}

void
RenderBuffer::Resize(int w, int h)
{
	if (width_ == w && height_ == h) return;

	width_ = w; height_ = h;
	glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, fbo_id_);
	glBindRenderbufferEXT(GL_RENDERBUFFER_EXT, rbo_id_);
	glRenderbufferStorageEXT(GL_RENDERBUFFER_EXT, GL_RGBA,
							 w, h);

	if (depth_test_) {
		glBindRenderbufferEXT(GL_RENDERBUFFER_EXT, dbo_id_);
		glRenderbufferStorageEXT(GL_RENDERBUFFER_EXT, GL_DEPTH_COMPONENT32,
								 w, h);
	}

	glBindRenderbufferEXT(GL_RENDERBUFFER_EXT, 0);
	glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, 0);
}

void
RenderIntensityBuffer::Resize(int w, int h)
{
	if (width_ == w && height_ == h) return;

	width_ = w; height_ = h;
	glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, fbo_id_);
	glBindRenderbufferEXT(GL_RENDERBUFFER_EXT, rbo_id_);
	glRenderbufferStorageEXT(GL_RENDERBUFFER_EXT, GL_RED,
		w, h);

	if (depth_test_) {
		glBindRenderbufferEXT(GL_RENDERBUFFER_EXT, dbo_id_);
		glRenderbufferStorageEXT(GL_RENDERBUFFER_EXT, GL_DEPTH_COMPONENT32,
			w, h);
	}

	glBindRenderbufferEXT(GL_RENDERBUFFER_EXT, 0);
	glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, 0);
}



void RenderBuffer::BeginDraw() const
{
	glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, fbo_id_);

	// clear buffers
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
}

void RenderBuffer::DisableDepthTest()
{
	depth_test_ = false;
}

void RenderBuffer::EnableDepthTest()
{
	depth_test_ = true;
}


void RenderBuffer::EndDraw()
{
	// unbind FBO
	glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, 0);
}

void RenderBuffer::ReadData(void* ptr)
{
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
void RenderIntensityBuffer::ReadData(void* ptr)
{
	glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, fbo_id_);
	glBindBufferARB(GL_PIXEL_PACK_BUFFER_ARB, pbo_id_);


	glBufferDataARB(GL_PIXEL_PACK_BUFFER_ARB, width_* height_, 0, GL_STREAM_READ_ARB);
	glReadPixels(0, 0, width_, height_, GL_RED, GL_UNSIGNED_BYTE, 0);

	GLubyte* src = (GLubyte*)glMapBufferARB(GL_PIXEL_PACK_BUFFER_ARB, GL_READ_ONLY_ARB);
	if(src)
	{
		memcpy(ptr, src, width_* height_);
		glUnmapBufferARB(GL_PIXEL_UNPACK_BUFFER_ARB); // release the mapped buffer
	}

	glBindBufferARB(GL_PIXEL_PACK_BUFFER_ARB, 0);
	glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, 0);

}

void RenderBuffer::ReadDepthBuffer(void* ptr)
{
	glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, fbo_id_);
	glBindBufferARB(GL_PIXEL_PACK_BUFFER_ARB, pbo_id_);
	glReadBuffer(GL_COLOR_ATTACHMENT0_EXT);

	glBufferDataARB(GL_PIXEL_PACK_BUFFER_ARB, width_* height_*sizeof(float), 0, GL_STREAM_READ_ARB);
	glReadPixels(0, 0, width_, height_, GL_DEPTH_COMPONENT, GL_FLOAT, 0);

	GLubyte* src = (GLubyte*)glMapBufferARB(GL_PIXEL_PACK_BUFFER_ARB, GL_READ_ONLY_ARB);
	if(src)
	{
		memcpy(ptr, src, width_* height_*sizeof(float));
		glUnmapBufferARB(GL_PIXEL_UNPACK_BUFFER_ARB); // release the mapped buffer
	}

	glBindBufferARB(GL_PIXEL_PACK_BUFFER_ARB, 0);
	glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, 0);

}


};