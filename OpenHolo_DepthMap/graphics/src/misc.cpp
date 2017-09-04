#include "graphics/misc.h"


#include <math.h>

namespace graphics {





void byte_copy(int n, char* a, char* b)
{
    while(n--) 
	*a++ = *b++;
}

void byte_swap(int n, char* a, char* b)
{
    while(n--) {
	char c = *a;
	*a++ = *b;
	*b++ = c;
    }
}
}; // namespace graphics