
#ifndef  __color_h_
#define  __color_h_

#include "graphics/vec.h"

namespace graphics {

class color;

class color : public vec4 {

public:


    //-------------------------------------------------------------------------
    // constructors: The signature is nearly same as class vec4
    //
    color() : vec4(1.0)	{ }

    color(const vec3& val)  { v[0] = val[0]; v[1] = val[1]; v[2] = val[2]; v[3] = 1.0; /*trunc();*/ }
    color(const vec4& val) : vec4(val) { /*trunc();*/ }
    color(const color& val) : vec4(val) {/* trunc(); */}

    color(real r, real g, real b, real a): vec4(r, g, b, a) { /*trunc();*/ }
    color(real r, real g, real b) { v[0] = r; v[1] = g; v[2] = b; v[3] = 1.0; /*trunc();*/ }
    color(int  r, int  g, int  b) { v[0] = r/255; v[1] = g/255; v[2] = g/255; v[3] = 1.0; /*trunc();*/ }
    
    color(real a): vec4(a) { }



    //
    // assignment
    inline color& operator=(const color& val) 
	{ vec4::operator =(val); return *this; } 


    //
    // HSV routines. Those are 3 reals containing the Hue, Saturation and
    // Value (same as brightness) of the color.
    //
    //
    // Sets value of color vector from 3 hsv components
    color &	set_HSV_value(real h, real s, real v);
    color &	set_HSV_value(const real hsv[3])
	{ return set_HSV_value(hsv[0], hsv[1], hsv[2]); }
    // Returns 3 individual hsv components
    void	get_HSV_value(real &h, real &s, real &v) const;

	void    set_V_value(real v) { real h,s,ov; get_HSV_value(h,s,ov); set_HSV_value(h,s,v); }
	real    get_V_value() const { real h,s,ov; get_HSV_value(h,s,ov); return ov; }

	real    get_H_value() const { real h,s,ov; get_HSV_value(h,s,ov); return h; }
	void    set_H_value(real v) { real h,s,ov; get_HSV_value(h,s,ov); set_HSV_value(v,s,ov); }

	real    get_S_value() const { real h,s,ov; get_HSV_value(h,s,ov); return s; }
	void    set_S_value(real v) { real h,s,ov; get_HSV_value(h,s,ov); set_HSV_value(h,v,ov); }

    // Returns an array of 3 hsv components
    void	get_HSV_value(real hsv[3]) const 
	{ get_HSV_value(hsv[0], hsv[1], hsv[2]); }

	// convert to color wheel position
    vec2    to_position() const;

	// color wheel pos to color
	void    set_with_position(const vec2& v); 
    //
    // RGBA Packed integer color routines. The color format expressed in 
    // hexadecimal is 0xrrggbbaa, where
    //	    aa 	is the alpha value
    //	    bb 	is the blue value
    //	    gg 	is the green value
    //	    rr 	is the red value
    // RGBA component values range from 0 to 0xFF (255).
    //
    
    // Sets value from ordered RGBA packed color. Alpha value is used for
    // transparency.
    color &	set_packed_value(unsigned int orderedRGBA);

    // Returns ordered RGBA packed color. Alpha is 1 - transparency, scaled
    // between 0 and 255 = 0xFF.
    unsigned int   get_packed_value() const;

    real trunc(real a);

    void trunc();

    int ir() { return (int) (v[0] * 255); }
    int ig() { return (int) (v[1] * 255); }
    int ib() { return (int) (v[2] * 255); }

    int gray() { return 2 * ir() + 2* ig() + ib(); }

};

color operator + (color a, color b);

color operator + (real a, color b);

color operator + (color a, real b);



color operator - (color a, color b);

color operator - (real a, color b);

color operator - (color a, real b);



color operator * (color a, color b);

color operator * (real a, color b);

color operator * (color a, real b);



color operator / (color a, color b);

color operator / (real a, color b);

color operator / (color a, real b);



int gray(color& c);

}; // namespace graphics
#endif
