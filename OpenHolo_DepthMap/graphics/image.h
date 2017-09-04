#ifndef __image_h
#define __image_h

#include <stdlib.h>

#include "graphics/unsigned.h"

namespace graphics
{

	template <typename T>
	struct image {
		int w, h;				// width, height
		int ww;
		T* buf;
		T zero;

		image() {
			ww = w = h = 0;
			buf = 0;
			zero = 0;
		}

		image(int a, int b) {
			ww = w = h = 0;
			buf = 0;
			alloc(a, b);
		}

		image(image& a) {
			ww = w = h = 0;
			buf = 0;
			alloc(a.w, a.h);
		}

		~image() { free(buf); }

		void alloc(int a, int b);

		T& operator() (int x, int y) { 
			int idx = y * ww + x;
			if(idx >= 0 && idx < ww * h) return buf[idx]; else return zero; }
	};

	template <typename T>
	void image<T>::alloc(int a, int b)
	{
		ww = w = a;
		h = b;

		if(sizeof(T) < 4) {
			int k = 4 / sizeof(T);
			ww = (w + k - 1) / k * k;
		}

		buf = (T*) realloc(buf, ww * h * sizeof(T));
		for(int i = 0; i < ww * h;++i)
			buf[i] = 0;
	}

//	template struct image<uchar>;
//	template struct image<uint>;

}

#endif
