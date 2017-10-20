#ifndef __intersection_h
#define	__intersection_h

#include "graphics/frame.h"

namespace graphics {

bool TetrahedronPlaneIntersect(const frame& pl, // plane
							   const vec3&	a,	// tetrahedra vertex a
							   const vec3&	b,  // tetrahedra vertex b
							   const vec3&	c,  // tetrahedra vertex c
							   const vec3&	d,  // tetrahedra vertex d
							   const vec3&  av, // tetrahedra vertex data a
							   const vec3&  bv, // tetrahedra vertex data a
							   const vec3&  cv, // tetrahedra vertex data a
							   const vec3&  dv, // tetrahedra vertex data a
							   std::vector<vec3>& poly,
							   std::vector<vec3>& out_v);

};

#endif