#ifndef __rasterizer_h
#define __rasterizer_h

#include	"graphics/sys.h"
#include	"graphics/geom.h"
#include	"graphics/matrix.h"

namespace graphics {

class Rasterizer {
public:

	Rasterizer(int w = 100, int h = 100);

	void  Resize(int w, int h);
	void  DrawBox(const ivec2& bottom_left, const ivec2& top_right);
	real  ReadContent(const ivec2& bottom_left, const ivec2& top_right);

	void  Clear();

private:

	gmatrix<unsigned char> display_;
};

};	// namespace graphics

#endif