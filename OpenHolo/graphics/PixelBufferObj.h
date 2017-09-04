#ifndef __PixelBufferObj_h
#define __PixelBufferObj_h

#include "graphics/unsigned.h"

namespace graphics {

class PixelBufferObj {

	enum BufferMode {UNPACK, PACK};

public:

	PixelBufferObj(BufferMode mode = UNPACK);
	virtual ~PixelBufferObj();

	void	Bind();

	void	Unbind();

	uint	buffer_size() const;

	void	Resize(uint size);

	uchar*	MapBuffer();

	void	UnmapBuffer();

	// Usage:
	// Create first stream textures with StreamTexture
	// 1. Bind
	// 2. StreamTexture.connect(pos); // stream texture가 pbo에서 어느 offset에 연결되는지 지정
	// 3. Resize(size) // resize the pbo
	// 4. addr = MapBuffer
	//    copy data to the addr
	// 5. UnmapBuffer
	//

protected:

	bool		bound_;
	uint		buffer_id_;
	BufferMode	mode_;
	uint		buffer_size_;
};

};

#endif