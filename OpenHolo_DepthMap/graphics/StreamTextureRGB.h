#ifndef __StreamTextureRGB_h
#define __StreamTextureRGB_h

#include "graphics/gl_extension.h"
#include "graphics/unsigned.h"
#include "graphics/real.h"

namespace graphics {

// stream texture rgba
class StreamTextureRGB {

public:
	StreamTextureRGB();
	virtual ~StreamTextureRGB();

	void Bind();
	void Unbind();
	virtual void Resize(int w, int h);
	void Connect(int pbo_position = 0);

	int  width() const { return width_; }
	int  height() const { return height_; }

	void Copy(uchar* data);

	//
	// return the texture data size in byte
	int  TextureDataSize() const;
	void SetTexture(uint id) { texture_id_ = id; }

	uint GetTexture() const { return texture_id_; }
	//
	// return width_/height_
	real AspectRatio() const;

protected:

	int		width_;
	int		height_;
	uint	texture_id_;
	bool    bound_;
};

};

#endif