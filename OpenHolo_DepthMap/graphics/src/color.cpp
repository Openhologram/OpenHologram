#include "graphics/real.h"
#include "graphics/color.h"
#include "graphics/quater.h"

namespace graphics {

//
// Sets value of vector from 3 individual hsv components
//
color &
color::set_HSV_value(real hue, real sat, real val)
{
    real f,q,t,p;
    int i;
    
    if (hue == 1.0)
	hue = 0.0;
    else
	hue *= 6.0;
    i = (int)(floor(hue));
    f = hue-i;
    p = val*(1.0-sat);
    q = val*(1.0-(sat*f));
    t = val*(1.0-(sat*(1.0-f)));
    switch (i) {
	case 0: v[0] = val; v[1] = t; v[2] = p; break;
	case 1: v[0] = q; v[1] = val; v[2] = p; break;
	case 2: v[0] = p; v[1] = val; v[2] = t; break;
	case 3: v[0] = p; v[1] = q; v[2] = val; break;
	case 4: v[0] = t; v[1] = p; v[2] = val; break;
	case 5: v[0] = val; v[1] = p; v[2] = q; break;
    }

    return (*this);
}

//
// Returns 3 individual hsv components
//
void
color::get_HSV_value(real &hue, real &sat, real &val) const
{
    real max,min;
    
    max = (v[0] > v[1]) ? 
	((v[0] > v[2]) ? v[0] : v[2]) : 
	((v[1] > v[2]) ? v[1] : v[2]);
    min = (v[0] < v[1]) ? 
	((v[0] < v[2]) ? v[0] : v[2]) : 
	((v[1] < v[2]) ? v[1] : v[2]);
    
    // brightness
    val = max;
    
    // saturation
    if (max != 0.0) 
	sat = (max-min)/max;
    else
	sat = 0.0;
    
    // finally the hue
    if (sat  !=  0.0) {
    	real h;
	
	if (v[0]  ==  max) 
	    h = (v[1] - v[2]) / (max-min);
	else if (v[1]  ==  max)
	    h = 2.0 + (v[2] - v[0]) / (max-min);
	else
	    h = 4.0 + (v[0] - v[1]) / (max-min);
	if (h < 0.0)
	    h += 6.0;
	hue = h/6.0;
    }
    else
    	hue = 0.0;
}

//
// Set value of vector from rgba color
//
color &
color::set_packed_value(unsigned int orderedRGBA)
{
    real f = 1.0 / 255.0;
    v[0] = ((orderedRGBA & 0xFF000000)>>24) * f;
    v[1] = ((orderedRGBA & 0xFF0000) >> 16) * f;
    v[2] = ((orderedRGBA & 0xFF00) >> 8) * f;
    v[3] = (orderedRGBA & 0xFF) * f;
    
    return (*this);
}

vec2    color::to_position() const
{
	real h, s, v;
	get_HSV_value(h, s, v);

	real ang = h* 2.0 * M_PI;
	quater q = orient(ang, vec3(0,0,1));
	vec3 ret = rot(q, vec3(1,0,0));
	s = min(1, max(s,0));
	return vec2(ret[0] * s, ret[1] * s);
}

// color wheel pos to color
void    color::set_with_position(const vec2& pos)
{
	real h, s, v;
	get_HSV_value(h, s, v);

	real len = norm(pos);	
	if (len > 1) return;
	if (apx_equal(len, 0)) { set_HSV_value(0,0,v); return; }
	vec2 pos1 = pos;		
	pos1.unit();
	vec2 base = vec2(1.0,0.0);		
	real vv = inner(pos1, base);	
	vv =	 acos(vv); 
	if (pos[1] < 0) vv = M_PI + (M_PI-vv); 
	base = vec2(vv/(2.0*M_PI), len);
	set_HSV_value(base[0], base[1], v);
}
//
// Returns orderedRGBA packed color format
//
unsigned int
color::get_packed_value() const
{
    return (
    	(((unsigned int) (v[0] * 255)) << 24) +
	(((unsigned int) (v[1] * 255)) << 16) +
	(((unsigned int) (v[2] * 255)) << 8) +
	((unsigned int)  (v[3] * 255)));
}

real color::trunc(real a) 
{
    if(a > 1)
	return 1.;
    else if(a <= 0) 
	return 0;
    else
	return a;
}

void color::trunc() 
{
    for(int i = 0; i < 3;++i){
	v[i] = trunc(v[i]);
    }
}

color operator + (color a, color b)
{
    color c = (vec4) a + (vec4) b;
    //c.trunc();
    return c;
}

color operator + (real a, color b)
{
    color c = (vec4) a + (vec4) b;
    //c.trunc();
    return c;
}

color operator + (color a, real b)
{
    color c = (vec4) a + (vec4) b;
    //c.trunc();
    return c;
}



color operator - (color a, color b)
{
    color c = (vec4) a - (vec4) b;
    //c.trunc();
    return c;
}

color operator - (real a, color b)
{
    color c = (vec4) a - (vec4) b;
    //c.trunc();
    return c;
}

color operator - (color a, real b)
{
    color c = (vec4) a - (vec4) b;
    //c.trunc();
    return c;
}



color operator * (color a, color b)
{
    color c = (vec4) a * (vec4) b;
    //c.trunc();
    return c;
}

color operator * (real a, color b)
{
    color c = (vec4) a * (vec4) b;
    //c.trunc();
    return c;
}

color operator * (color a, real b)
{
    color c = (vec4) a * (vec4) b;
    //c.trunc();
    return c;
}



color operator / (color a, color b)
{
    color c = (vec4) a / (vec4) b;
    //c.trunc();
    return c;
}

color operator / (real a, color b)
{
    color c = (vec4) a / (vec4) b;
    //c.trunc();
    return c;
}

color operator / (color a, real b)
{
    color c = (vec4) a / (vec4) b;
    //c.trunc();
    return c;
}



int gray(color& c)
{
    return (2 * c.ir() + 2* c.ig() + c.ib());
}

};