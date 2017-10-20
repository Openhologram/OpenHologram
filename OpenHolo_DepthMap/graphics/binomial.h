#ifndef __binomial_h
#define __binomial_h

#include "graphics/sys.h"
#include "graphics/log.h"
#include "graphics/ivec.h"
#include "graphics/vector.h"
#include "graphics/matrix.h"

namespace graphics {

int fact(int n);

int Binomial(int a, int b);




//| Binomial(2, 2)
void Binomial(vector<ivec2>& b, ivec2& a);


//| Binomial(3, 2)
void Binomial(vector<ivec2>& b, ivec3& a);


//| Binomial(4, 2)
void Binomial(vector<ivec2>& b, ivec4& a);


}; // namespace graphics 
#endif
