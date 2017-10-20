#ifndef __quater_h
#define __quater_h
//|
//| rotation using unit quaternion
//|


#include "graphics/sys.h"
#include "graphics/real.h"
#include "graphics/vec.h"
#include "graphics/matrix3x3.h"

namespace graphics {

typedef vec4 quater;

quater _quater(real w, const vec3& v)
;

quater _quater(const vec3& v)
;

quater _zero_quater()
;

vec3 _vec3(const quater& a)
;

/*|
 * Rotation quaternion generator : "orient()"
 *   generates a quaternion via which we could do rotation
 *   around a direction 'v' by 'phi' angle.
 *
 *   This can be achieved like following :
 *
 *       quater a = orient(phi, v);
 *       quater b = orient(theta, u);
 *       quater c = a & b;            composition operator: b after a
 *       new_pt = rot(c, point);	

 *   A series of statements defined above mean that
 *   "rotate 'point' first by phi around an arbitrary axis(direction) v and
 *   rotate the once rotated point by theta around another axis u."
   
 *   Good things with quaternion is less computation compared
 *   to that of direct matrix.
 */

quater orient(real phi, const vec3& v)
;

/*|
 * In contrary to orient, get_rotation gets the rotation info.,
 * angle and axis corresponding to the quaternion a.
 * THIS IS NOT SAFE FOR THE TIME BEING
 */
void get_rotation(const quater& a, real& phi, vec3& v)
;

quater slerp(const quater& a, const quater& b, real lambda);
/*|
 * Compute a rotational quaternion which takes the vector 
 * uu onto the vector vv.
 * assumption : uu and vv are already unit vector
 * Originally programmed by Joran Popovic.
 * Edited by Francis to our end.
 */

quater u2v_quater(const vec3& uu, const vec3& vv)
;

//| when they are pointing in the opposite direction, it uses the given
//| default_axis as an rotational axis. default_axis is assumed to be
//| perpendicular to both uu and vv.
quater u2v_quater(const vec3& uu, const vec3& vv, const vec3& default_axis)
;

matrix3x3 to_matrix(const quater& q);

quater from_matrix(const matrix3x3& basis);

vec3 euler(const quater q);

#undef eps_eq

quater operator& (const quater& a, const quater& b)	//| rotational mult, b after a
;

quater inv(const quater& a)			//| mult inverse for unit quaternion
;

quater exp(const vec3& v)			//| rotational exp of w = log(q)
;

quater d_exp(const vec3& v, const vec3& dv)
;

quater pow(const quater& q, real a)		//| rotational pow
;

quater pow(const vec3& w, real a)		//| rotational pow
;

quater d_pow(const vec3& w, real a)		//| rotational pow
;

quater dd_pow(const vec3& w, real a)		//| rotational pow
;

vec3 log(const quater& q)			//| rotational log 
;

vec3 d_log(const quater& q, const quater& dq)		//| Jacobian of log(q), at q
;

vec3 rot(const quater& a, const vec3& v)		//| rotate vec3 v by quaternion a
;

};

#endif
