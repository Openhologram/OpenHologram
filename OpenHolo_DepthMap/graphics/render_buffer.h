#ifndef __render_buffer_h
#define __render_buffer_h

#include "graphics/sys.h"
#include "graphics/gl.h"
#include "graphics/gl_extension.h"
#include "graphics/unsigned.h"

namespace graphics {

class RenderBuffer{

public:

	RenderBuffer();
	virtual ~RenderBuffer();

	void GenerateObjects();
	virtual void Resize(int w, int h);

	void BeginDraw() const;
	void EndDraw();

	virtual void ReadData(void* ptr); // ptr should be memory-allocated before the call
	void ReadDepthBuffer(void* ptr);
	bool IsValid() const { return rbo_id_!= 0; }

	// these function should be called before GenerateObjects()!!
	void DisableDepthTest();
	void EnableDepthTest();

	int width() const { return width_; }
	int height() const { return height_; }


protected:

	bool depth_test_;
	int	 width_;
	int  height_;
	uint rbo_id_;
	uint dbo_id_;
	uint fbo_id_;
	uint pbo_id_;
};



class RenderIntensityBuffer: public RenderBuffer{

public:

	RenderIntensityBuffer();
	virtual ~RenderIntensityBuffer();

	virtual void Resize(int w, int h);
	virtual void ReadData(void* ptr); // ptr should be memory-allocated before the call
};


};

#endif