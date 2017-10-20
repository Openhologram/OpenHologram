#include "graphics/scaled_frame.h"
#include <graphics/gl.h>
namespace graphics {



frame& scaled_frame::operator = (const frame& a) {

	const scaled_frame& ff = dynamic_cast<const scaled_frame&>(a);
    for(int i = 0 ; i < 3; i++)
	   basis[i] = a.basis[i];

    eye_position = a.eye_position;
	scale = ff.scale;

	memcpy(worldMatrix, a.worldMatrix, sizeof(real)*16);
	memcpy(inverseWorldMatrix, a.inverseWorldMatrix, sizeof(real)*16);

    return *this;
}


box3 scaled_frame::Transform(const box3& input) const
{
	vec3 min_ = input.get_minimum();
	vec3 max_ = input.get_maximum();
	
	min_ = min_ * scale;
	max_ = max_ * scale;
	min_ = to_world(min_);
	max_ = to_world(max_);

	return box3(min_, max_);
}

void scaled_frame::push_to_world() const
{
	real mat[16];
	gl_identity(mat);
	mat[0] = scale[0];
	mat[5] = scale[1];
	mat[10] = scale[2];
    glPushMatrix();
    gl_multmatrix((real*)worldMatrix); //
	gl_multmatrix(mat);
}

void scaled_frame::push_to_model() const
{
	real mat[16];
	gl_identity(mat);
	mat[0] = 1.0/scale[0];
	mat[5] = 1.0/scale[1];
	mat[10] = 1.0/scale[2];

    glPushMatrix();
	gl_multmatrix(mat);
    gl_multmatrix((real*)inverseWorldMatrix); //
}

vec3 scaled_frame::to_model(const vec3& a) const
{
    vec4 ret(0);
    vec4 aa(a[0], a[1], a[2], 1.0);

    for (int i = 0 ; i < 4 ;++i)
	for (int j = 0 ; j < 4 ; ++j)
	    ret[i] += inverseWorldMatrix[j*4 + i] * aa[j];

    return vec3(ret[0]/ret[3]/scale[0], ret[1]/ret[3]/scale[1], ret[2]/ret[3]/scale[2]);
}

vec4 scaled_frame::to_model(const vec4& a) const
{
    vec4 ret(0);

    for (int i = 0 ; i < 4 ;++i)
	for (int j = 0 ; j < 4 ; ++j)
	    ret[i] += inverseWorldMatrix[j*4 + i] * a[j];
	ret[0]/=scale[0];
	ret[1]/=scale[1];
	ret[2]/=scale[2];
    return ret;
}

vec3 scaled_frame::to_world(const vec3& a) const
{
    vec4 ret(0);
	vec3 aa = a;
	aa *= scale;
    vec4 bb(aa[0], aa[1], aa[2], 1.0);

    for (int i = 0 ; i < 4 ;++i)
	for (int j = 0 ; j < 4 ; ++j)
	    ret[i] += worldMatrix[j*4 + i] * bb[j];

    return vec3(ret[0]/ret[3], ret[1]/ret[3], ret[2]/ret[3]);
}

vec4 scaled_frame::to_world(const vec4& a) const
{
    vec4 ret(0);
	vec4 aa(a[0]*scale[0], a[1]*scale[1], a[2]*scale[2], a[3]);
    for (int i = 0 ; i < 4 ;++i)
	for (int j = 0 ; j < 4 ; ++j)
	    ret[i] += worldMatrix[j*4 + i] * aa[j];

    return ret;
}

vec3 scaled_frame::to_model_normal(const vec3& a) const
{
    vec3 ret(0);

    for (int i = 0 ; i < 3 ;++i)
	for (int j = 0 ; j < 3 ; ++j)
	    ret[i] += inverseWorldMatrix[j*4 + i] * a[j];

	ret = ret;
    return ret;
}

vec3 scaled_frame::to_world_normal(const vec3& a) const
{
    vec3 ret(0);

    for (int i = 0 ; i < 3 ;++i)
	for (int j = 0 ; j < 3 ; ++j)
	    ret[i] += worldMatrix[j*4 + i] * a[j];

    return ret;
}

line scaled_frame::to_model(const line& a) const
{
	vec3 p1 = a.get_position();
	vec3 p2 = a.get_position() + a.get_direction();
	p1 = to_model(p1);
	p2 = to_model(p2);
	line new_l(p1,p2);

    return new_l;
}

line scaled_frame::to_world(const line& a) const
{
	vec3 p1 = a.get_position();
	vec3 p2 = a.get_position() + a.get_direction();
	p1 = to_world(p1);
	p2 = to_world(p2);
	line new_l(p1,p2);

    return new_l;
}


};