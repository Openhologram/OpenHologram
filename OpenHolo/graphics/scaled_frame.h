#ifndef __scaled_frame_h
#define __scaled_frame_h
/*|~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*/
/*|    DEFINE CAMERA FRAME                                                   */
/*|__________________________________________________________________________*/

#include "graphics/sys.h"
#include "graphics/real.h"
#include "graphics/vec.h"
#include "graphics/matrix.h"
#include "graphics/quater.h"
#include "graphics/gl.h"
#include "graphics/geom.h"
#include "graphics/frame.h"


namespace graphics {

class scaled_frame: public frame {

public:

	vec3 scale;

	scaled_frame(): frame(), scale(1)
	{
	}


	scaled_frame( const vec3 &org, const vec3 &x, const vec3 &y, const vec3 &z)
	: frame(org, x, y, z), scale(1)
	{}

	scaled_frame(const vec3& eye, const vec3& dir, const vec3& up) 
		: frame(eye, dir, up), scale(1)
	{
	}

	scaled_frame(const vec3 &dir, const vec3 &up)
		: frame(dir, up), scale(1)
	{
	}

	scaled_frame(const scaled_frame &val)
		: frame(val), scale(val.scale)
	{
	}

	virtual vec3 get_scale() const { return scale; }

	void SetScale(vec3& s) { scale = s; }

	virtual frame& operator = (const frame& a) ;

	box3 Transform(const box3& input) const;

    virtual void push_to_world() const ;

    virtual void push_to_model() const;

    virtual vec3 to_model(const vec3& a) const;
    
    virtual vec4 to_model(const vec4& a) const;
    virtual vec3 to_model_normal(const vec3& a) const;

    virtual vec4 to_world(const vec4& a) const;

    virtual vec3 to_world(const vec3& a) const;
    virtual vec3 to_world_normal(const vec3& a) const;

    virtual line to_model(const line& a) const;

    virtual line to_world(const line& a) const;
};



};
#endif
