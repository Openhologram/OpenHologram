#include "graphics/quater.h"
//|
//| rotation using unit quaternion
//|




#include <math.h>
#include "graphics/epsilon.h"

namespace graphics {


typedef vec4 quater;

quater _quater(real w, const vec3& v)
{
    return vec4(w, v[0], v[1], v[2]);
}

quater _quater(const vec3& v)
{
    return vec4(0, v[0], v[1], v[2]);
}

quater _zero_quater()
{
    return vec4(1.0, 0.0, 0.0, 0.0);
}

vec3 _vec3(const quater& a)
{
    return vec3(a[1], a[2], a[3]);
}

matrix3x3 to_matrix(const quater& q)
{
	matrix3x3 basis;

	real X = q[1];
	real Y = q[2];
	real Z = q[3];
	real W = q[0];

    real xx      = X * X;
    real xy      = X * Y;
    real xz      = X * Z;
    real xw      = X * W;

    real yy      = Y * Y;
    real yz      = Y * Z;
    real yw      = Y * W;

    real zz      = Z * Z;
    real zw      = Z * W;

    basis[0][0]  = 1.0 - 2.0 * ( yy + zz );
    basis[0][1]  =     2.0 * ( xy - zw );
    basis[0][2]  =     2.0 * ( xz + yw );

    basis[1][0]  =     2.0 * ( xy + zw );
    basis[1][1]  = 1.0 - 2.0 * ( xx + zz );
    basis[1][2]  =     2.0 * ( yz - xw );

    basis[2][0]  =     2.0 * ( xz - yw );
    basis[2][1]  =     2.0 * ( yz + xw );
    basis[2][2]  = 1.0 - 2.0 * ( xx + yy );

	return basis;
}

quater from_matrix(const matrix3x3& basis)
{
	real a = basis[0][0];
	real b = basis[1][1];
	real c = basis[2][2];
	real t = a + b + c + 1.0;

	real qx, qy, qz, qw, s;
	if (t > 0.0) {
		s = 0.5/sqrt(t);
		qw = 0.25/s;
		qx = (basis[2][1] - basis[1][2]) * s;
		qy = (basis[0][2] - basis[2][0]) * s;
		qz = (basis[1][0] - basis[0][1]) * s;
	}
	else if (a >= b && a >= c) {
		s = sqrt(1.0 + a - b - c) * 2.0;
		qx = 0.5/s;
		qy = (basis[0][1] + basis[1][0])/s;
		qz = (basis[0][2] + basis[2][0])/s;
		qw = (basis[1][2] + basis[2][1])/s;
	}
	else if (b >= a && b >= c) {
		s = sqrt(1.0 + b - c - a) * 2.0;
		qx = (basis[0][1] + basis[1][0])/s;
		qy = 0.5/s;
		qz = (basis[1][2] + basis[2][1])/s;
		qw = (basis[0][2] + basis[2][0])/s;
	}
	else if (c >= a && c >= b) {
		s = sqrt(1.0 + c - a - b) * 2.0;
		qx = (basis[0][2] + basis[2][0])/s;
		qy = (basis[1][2] + basis[2][1])/s;
		qz = 0.5/s;
		qw = (basis[0][1] + basis[1][0])/s;
	}

	return quater(qw, qx, qy, qz);
}

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
{
    real w = cos(phi / 2);
    vec3 _v = unit(v) * sin(phi / 2);
    return _quater(w, _v);
}

quater slerp(const quater& q1, const quater& q2, real lambda)
{
	

	real dotproduct = inner(q1, q2); 
	real theta, st, sut, sout, coeff1, coeff2;

	if (dotproduct > 1.0) dotproduct = 1.0;
	if (dotproduct < -1.0) dotproduct = -1.0;

	theta = (real) acos(dotproduct);
	if (theta<0.0) theta=-theta;
	
	st = (real) sin(theta);
	sut = (real) sin(lambda*theta);
	sout = (real) sin((1-lambda)*theta);

	if (apx_equal(st, 0, zero_epsilon)) 
		return q2;

	coeff1 = sout/st;
	coeff2 = sut/st;

	quater qr;

	qr[1] = coeff1*q1[1] + coeff2*q2[1];
	qr[2] = coeff1*q1[2] + coeff2*q2[2];
	qr[3] = coeff1*q1[3] + coeff2*q2[3];
	qr[0] = coeff1*q1[0] + coeff2*q2[0];

	real n = norm(qr);
	qr[1]/=n;
	qr[2]/=n;
	qr[3]/=n;

	return qr;
}

/*|
 * In contrary to orient, get_rotation gets the rotation info.,
 * angle and axis corresponding to the quaternion a.
 * THIS IS NOT SAFE FOR THE TIME BEING
 */
void get_rotation(const quater& a, real& phi, vec3& v)
{
    //| get rotational info. from a quaternion

	if (a[0] < -1.0 || a[0] > 1.0) {
		phi = 0.0;
		v = vec3(0);
		return;
	}
    real t = acos(a[0]);

    phi = 2.0 * t;
    t = 1.0 / sin(t);

    v[0] = a[1] * t;
    v[1] = a[2] * t;
    v[2] = a[3] * t;
}

/*|
 * Compute a rotational quaternion which takes the vector 
 * u onto the vector v.
 * assumption : u and v are already unit vector
 * Originally programmed by Joran Popovic.
 * Edited by Francis to our end.
 */

quater u2v_quater(const vec3& uu, const vec3& vv)
{
    vec3 u = unit(uu);
    vec3 v = unit(vv);

    vec3 mid = u + v;
    if (apx_equal(norm(mid), 0)) {	   //| check for 180 degree rotation
	static vec3 axis(1, 0, 0);
	if (apx_equal(fabs(u[0]), 1.) || apx_equal(fabs(v[0]), 1.)) {
	    axis[0] = 0.0;
	    axis[1] = 1.0;
	}
	return _quater(0., cross(u, axis));
    } else {
	mid = unit(mid);
	return _quater(inner(u,mid), cross(u,mid));
    }
}

//| when they are pointing in the opposite direction, it uses the given
//| default_axis as an rotational axis. default_axis is assumed to be
//| perpendicular to both u and v.
quater u2v_quater(const vec3& uu, const vec3& vv, const vec3& default_axis)
{
    vec3 u = unit(uu);
    vec3 v = unit(vv);

    vec3 mid = u + v;

    if (apx_equal(norm(mid), 0)) {	   //| check for 180 degree rotation
	return orient(M_PI, default_axis);
    } else {	
	//| this includes the case where u and v conincide, which
	//| is automatically recognized as zero quaternion
	mid = unit(mid);
	return _quater(inner(u,mid), cross(u,mid));
    }
}

#undef eps_eq

quater operator& (const quater& a, const quater& b)	//| rotational mult, b after a
{
    quater c;
    c[0] = a[0] * b[0] - a[1] * b[1] - a[2] * b[2] - a[3] * b[3];
    c[1] = a[0] * b[1] + a[1] * b[0] + a[2] * b[3] - a[3] * b[2];
    c[2] = a[0] * b[2] + a[2] * b[0] + a[3] * b[1] - a[1] * b[3];
    c[3] = a[0] * b[3] + a[3] * b[0] + a[1] * b[2] - a[2] * b[1];
    return c;
}

quater inv(const quater& a)			//| mult inverse for unit quaternion
{
    return quater(a[0], -a[1], -a[2], -a[3]);
}

quater exp(const vec3& v)			//| rotational exp of w = log(q)
{
    real theta = norm(v);
    return _quater(cos(theta), sin(theta) * unit(v));
}

quater d_exp(const vec3& v, const vec3& dv)
{
    real n = norm(v);

    if(n < epsilon) {
	return _quater(dv);
    }

    real n_1 = 1 / n;
    real n_2 = n_1 / n;
    real n_3 = n_2 / n;
    real cosn = cos(n);
    real sinn = sin(n);

    real x = v[0];
    real y = v[1];
    real z = v[2];

    
    real w_x = -x * n_1 * sinn;
    real x_x = n_1 * sinn - x * x * n_3 * sinn + x * x * n_2 * cosn;
    
    real w_y = -y * n_1 * sinn;
    real y_y = n_1 * sinn - y * y * n_3 * sinn + y * y * n_2 * cosn;
    
    real w_z = -z * n_1 * sinn;
    real z_z = n_1 * sinn - z * z * n_3 * sinn + z * z * n_2 * cosn;
    

    
    real x_y = -x * y * n_3 * sinn + x * y * n_2 * cosn;
    real y_x = x_y;
    
    real y_z = -y * z * n_3 * sinn + y * z * n_2 * cosn;
    real z_y = y_z;
    
    real z_x = -z * x * n_3 * sinn + z * x * n_2 * cosn;
    real x_z = z_x;
    

    quater dexp;

    
    dexp[0] = inner(vec3(w_x, w_y, w_z), dv);
    
    dexp[1] = inner(vec3(x_x, x_y, x_z), dv);
    
    dexp[2] = inner(vec3(y_x, y_y, y_z), dv);
    
    dexp[3] = inner(vec3(z_x, z_y, z_z), dv);
    

    return dexp;
}

quater pow(const quater& q, real a)		//| rotational pow
{
    return exp(a * log(q));
}

quater pow(const vec3& w, real a)		//| rotational pow
{
    return exp(a * w);
}

quater d_pow(const vec3& w, real a)		//| rotational pow
{
    return pow(w, a) & _quater(w);
}

quater dd_pow(const vec3& w, real a)		//| rotational pow
{
    return pow(w, a) & _quater(w) & _quater(w);
}

vec3 log(const quater& q)			//| rotational log 
{
    vec3 v = _vec3(q);
    return atan2(norm(v), q[0]) * unit(v);
}

vec3 d_log(const quater& q, const quater& dq)		//| Jacobian of log(q), at q
{
    //| numerical solution

    quater dq_w = quater(epsilon, 0, 0, 0);
    quater dq_x = quater(0, epsilon, 0, 0);
    quater dq_y = quater(0, 0, epsilon, 0);
    quater dq_z = quater(0, 0, 0, epsilon);

    log(q + q);
    
    vec3 dv_w = ( log(q + dq_w) - log(q) ) / epsilon;
    
    vec3 dv_x = ( log(q + dq_x) - log(q) ) / epsilon;
    
    vec3 dv_y = ( log(q + dq_y) - log(q) ) / epsilon;
    
    vec3 dv_z = ( log(q + dq_z) - log(q) ) / epsilon;
    

    return dv_w * dq[0] + dv_x * dq[1] + dv_y * dq[2] + dv_z * dq[3];
}

vec3 rot(const quater& a, const vec3& v)		//| rotate vec3 v by quaternion a
{
    quater c = a & _quater(v) & inv(a);
    return _vec3(c);
}


vec3 euler(const quater q)
{
	real x, y, z, w;
	vec3 ret;

	x = q[1];
	y = q[2];
	z = q[3];
	w = q[0];

	real sqx = x*x;
	real sqy = y*y;
	real sqz = z*z;
	real sqw = w*w;

	ret[0] = (real) atan2(2.0 * (y*z + x*w),1-2*(sqx + sqy));  
	ret[1] = (real) asin(-2.0 * (x*z - y*w));
	ret[2] = (real) atan2(2.0 * (x*y + z*w),1-2*(sqy + sqz));
	return ret;
}

}; //namespace graphics