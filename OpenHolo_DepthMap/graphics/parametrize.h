#ifndef __parametrize_h
#define __parametrize_h

#include "graphics/sys.h"
#include "graphics/real.h"
#include "graphics/vec.h"
#include "graphics/vector.h"

namespace graphics {

template <class T>
vector<real> chord_length_parametrize(const vector<T>& a)
;

template <class T>
vector<real> normalized_chord_length_parametrize(const vector<T>& a)
;

template <class T>
vector<real> centripetal_parametrize(const vector<T>& d)
;

template <class T>
vector<real> normalized_centripetal_parametrize(const vector<T>& d)
;



void scale_parametrization(vector<real>& a, real tmin, real tmax)
;

}; // namespace graphics
#endif
