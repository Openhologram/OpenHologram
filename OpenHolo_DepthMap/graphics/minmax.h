#ifndef __minmax_h
#define __minmax_h

#include "graphics/sys.h"
#include "graphics/real.h"

namespace graphics {

int  bound(int x, int l, int u);
real bound(real x, real l, real u);

#ifdef _MAC_OS
template<class T>
T max(const T& a, const T& b) { return a>b?a:b; }
template<class T>
T min(const T& a, const T& b) { return a>b?b:a; }
template<class T>
T min(const T& a, const T& b, const T& c, const T& d) { return min(min(min(a,b),c),d); }
template<class T>
T max(const T& a, const T& b, const T& c, const T& d) { return max(max(max(a,b),c),d); }
#endif
	
}; // namespace graphics
#endif
