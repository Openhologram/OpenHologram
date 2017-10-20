#include "graphics/gl_extension.h"
#include "graphics/PixelBufferObj.h"

namespace graphics {

PixelBufferObj::PixelBufferObj(BufferMode mode)
	:mode_(mode), buffer_size_(0), bound_(false)
{
	glGenBuffersARB(1, &buffer_id_);
	glBindBufferARB(GL_PIXEL_UNPACK_BUFFER_ARB, buffer_id_);
	glBufferDataARB(GL_PIXEL_UNPACK_BUFFER_ARB, 1000*1000*4, 0, GL_STREAM_DRAW_ARB);
	glBindBufferARB(GL_PIXEL_UNPACK_BUFFER_ARB, 0);
}

PixelBufferObj::~PixelBufferObj()
{
	glDeleteBuffersARB(1, &buffer_id_);
}


void PixelBufferObj::Bind()
{
	glBindBufferARB(GL_PIXEL_UNPACK_BUFFER_ARB, buffer_id_);
	bound_ = true;
}

void PixelBufferObj::Unbind()
{
	glBindBufferARB(GL_PIXEL_UNPACK_BUFFER_ARB, 0);
	bound_ = false;
}
uint  PixelBufferObj::buffer_size() const
{
	return buffer_size_;
}

void  PixelBufferObj::Resize(uint size)
{
	if (size == buffer_size_) return;
	buffer_size_ = size;

	if (!bound_) Bind();
	glBufferDataARB(GL_PIXEL_UNPACK_BUFFER_ARB, size, 0, GL_STREAM_DRAW_ARB);
}

uchar* PixelBufferObj::MapBuffer()
{
	if (!bound_) Bind();
	return (GLubyte*)glMapBufferARB(GL_PIXEL_UNPACK_BUFFER_ARB, GL_WRITE_ONLY_ARB);
}

void PixelBufferObj::UnmapBuffer()
{
	if (bound_)	glUnmapBufferARB(GL_PIXEL_UNPACK_BUFFER_ARB);
}

};