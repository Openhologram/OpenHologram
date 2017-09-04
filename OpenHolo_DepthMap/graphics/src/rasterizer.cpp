#include "graphics/rasterizer.h"

namespace graphics {

Rasterizer::Rasterizer(int w, int h)
	:display_(w, h)
{
}

void Rasterizer::Resize(int w, int h)
{
	display_.resize(w, h);
}

void Rasterizer::DrawBox(const ivec2& bottom_left, const ivec2& top_right)
{
	for (int i = bottom_left[0]; i < top_right[0] ;++i){
		for (int j = bottom_left[1] ; j < top_right[1] ; ++j) {
			if (i >= 0 && i < display_.n1 && j >= 0 && j < display_.n2)
				display_(i,j) =1;
		}
	}
}

real  Rasterizer::ReadContent(const ivec2& bottom_left, const ivec2& top_right)
{
	int ret = 0;
	for (int i = bottom_left[0]; i < top_right[0] ;++i){
		for (int j = bottom_left[1] ; j < top_right[1] ; ++j) {
			if (i >= 0 && i < display_.n1 && j >= 0 && j < display_.n2) {
				if (display_(i,j)) ret++;
			}
		}
	}
	ivec2 a = top_right-bottom_left;
	return ((real)ret/fabs((real)a[0] * (real)a[1])) * 100.0;
}

void Rasterizer::Clear()
{
	memset(display_.get_array(), 0, display_.n);
}

};