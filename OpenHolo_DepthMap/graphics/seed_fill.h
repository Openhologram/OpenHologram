#ifndef __seed_fill_h
#define __seed_fill_h

#include "graphics/geom.h"

namespace graphics {

struct WinRect {		/* window: a discrete 2-D rectangle */
    int x0, y0;			/* xmin and ymin */
    int x1, y1;			/* xmax and ymax (inclusive) */
};


typedef struct {short y, xl, xr, dy;} Segment;

/*
 * A Seed Fill Algorithm
 * by Paul Heckbert
 * from "Graphics Gems", Academic Press, 1990
 *
 * user provides pixelread() and pixelwrite() routines
 */

/*
 * fill.c : simple seed fill program
 * Calls pixelread() to read pixels, pixelwrite() to write pixels.
 *
 * Paul Heckbert	13 Sept 1982, 28 Jan 1987
 */


/*
 * Filled horizontal segment of scanline y for xl<=x<=xr.
 * Parent segment was on line y-dy.  dy=1 or -1
 */

#define MAX 10000		/* max depth of stack */

#define PUSH(Y, XL, XR, DY)	/* push new segment on stack */ \
    if (sp<stack+MAX && Y+(DY)>=win->y0 && Y+(DY)<=win->y1) \
    {sp->y = Y; sp->xl = XL; sp->xr = XR; sp->dy = DY; sp++;}

#define POP(Y, XL, XR, DY)	/* pop segment off stack */ \
    {sp--; Y = sp->y+(DY = sp->dy); XL = sp->xl; XR = sp->xr;}

/*
 * fill: set the pixel at (x,y) and all of its 4-connected neighbors
 * with the same pixel value to the new pixel value nv.
 * A 4-connected neighbor is a pixel above, below, left, or right of a pixel.
 */
template <typename T, typename V>
bool under_fill(int x, int y, WinRect* win, T nv, V idx, T* ref,  V* wrt, T mask_val, T* mask, int w, int h, box2& bound)
{
	bool ret = false;
	bound.make_empty();
    int l, x1, x2, dy;
    T ov;	/* old pixel value */
    Segment stack[MAX], *sp = stack;	/* stack of filled segments */

    ov = *(ref + x + y *w);		/* read pv at seed point */
	T mv = *(mask + x + y*w);
    if (ov==nv || x<win->x0 || mv < mask_val || x>win->x1 || y<win->y0 || y>win->y1) return false;
    PUSH(y, x, x, 1);			/* needed in some cases */
    PUSH(y+1, x, x, -1);		/* seed segment (popped 1st) */

    while (sp>stack) {
	/* pop segment off stack and fill a neighboring scan line */
	POP(y, x1, x2, dy);
	/*
	 * segment of scan line y-dy for x1<=x<=x2 was previously filled,
	 * now explore adjacent pixels in scan line y
	 */
	for (x=x1; x>=win->x0 && *(ref + x + y *w) < nv && *(mask + x + y*w) > mask_val ; x--) {
	    *(wrt + x + y * w) = idx;
		*(ref + x + y *w) = nv;
		bound.extend(vec2(x, y));
		ret = true;
	}

	if (x>=x1) goto skip;
	l = x+1;
	if (l<x1) PUSH(y, l, x1-1, -dy);		/* leak on left? */
	x = x1+1;
	do {
	    for (; x<=win->x1 && *(ref + x + y *w)<nv && *(mask + x + y*w) > mask_val ; x++) {
		 *(wrt + x + y * w) = idx;
		 *(ref + x + y *w) = nv;
		 bound.extend(vec2(x, y));
		 ret = true;
		}
	    PUSH(y, l, x-1, dy);
	    if (x>x2+1) PUSH(y, x2+1, x-1, -dy);	/* leak on right? */
skip:	    for (x++; x<=x2 && *(ref + x + y *w)>=nv && *(mask + x + y*w) < mask_val; x++);
	    l = x;
	} while (x<=x2);
    }
	return ret;
}

}
#endif