#ifndef __FloatStreamTexture_h
#define __FloatStreamTexture_h


#include "graphics/unsigned.h"
#include "graphics/real.h"

namespace graphics {

	class FloatStreamTexture {

	public:
		FloatStreamTexture();
		virtual ~FloatStreamTexture();

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