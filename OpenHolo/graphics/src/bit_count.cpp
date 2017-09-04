#include "graphics/bit_count.h"

namespace graphics {

int bit_count(int a)
{
    int c;
    c = 0;
    while(a != 0)
    {
	c++;
	a = a &~ -a;
    }
    return c;
}

}; // namespace graphics