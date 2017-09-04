#include "graphics/minmax.h"




namespace graphics {

int bound(int x, int l, int u)
{
    return max(min(x, u), l);
}

real bound(real x, real l, real u)
{
    return max(min(x, u), l);
}

}; // namespace graphics