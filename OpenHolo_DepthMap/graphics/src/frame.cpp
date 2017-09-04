#include "graphics/frame.h"
#include "graphics/change_type.h"
#include "graphics/_limits.h"
#include "graphics/matrix.h"
/*|~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*/
/*|    DEFINE CAMERA FRAME                                                   */
/*|__________________________________________________________________________*/

namespace graphics {

void frame::test()
{
	matrix m(4,4);
	for (int i = 0 ; i < 4 ;++i){
		for (int j = 0 ; j < 4 ; ++j)  {
			m(i,j) = worldMatrix[i*4 + j];
		}
	}
	matrix inv = m.inverse();
	matrix iw(4,4);
	for (int i = 0 ; i < 4 ;++i){
		for (int j = 0 ; j < 4 ; ++j)  {
			iw(i,j) = inverseWorldMatrix[i*4 + j];
		}
	}
	LOG("begin\n");
	LOG("inv\n");
	for (int i = 0 ; i < 4 ;++i){
		for (int j = 0 ; j < 4 ; ++j)  {
			LOG("%f ", inv(i,j));
		}
		LOG("\n");
	}
	LOG("invworld\n");
	for (int i = 0 ; i < 4 ;++i){
		for (int j = 0 ; j < 4 ; ++j)  {
			LOG("%f ", iw(i,j));
		}
		LOG("\n");
	}
	LOG("end\n");
}

void frame::reset()
{
    gl_identity(worldMatrix);
    gl_identity(inverseWorldMatrix);
    reverse_update();
}

void frame::set_with(const plane& a)
{
    vec3 epos = a.n * a.d;
    create_from_normal(epos, a.n);
}

void frame::Transform(const frame& f)
{
	basis[0] = f.to_model_normal(basis[0]);
	basis[1] = f.to_model_normal(basis[1]);
	basis[2] = f.to_model_normal(basis[2]);
	eye_position = f.to_model(eye_position);

	update();
}

bool frame::create_from_points(
          const vec3& P,  // point on the plane
          const vec3& Q,  // point on the plane
          const vec3& R   // point on the plane
          )
{
  eye_position = P;
  bool rc = basis[Z].perpendicular(P, Q, R);
  basis[X] = unit(Q - P);
  basis[Y] = cross( basis[Z], basis[X] );
  basis[Y].unit();

  update();
  return rc;
}

bool frame::create_from_frame(
    const vec3&  p, // point on the plane
    const vec3&  x, // non-zero vector in plane
    const vec3&  y  // another non-zero normal in the plane
    )
{
  eye_position = p;

  basis[X] = unit(x);
  basis[Y] = y - (inner(y, basis[X])*basis[X]);
  basis[Y].unit();
  basis[Z] = cross( basis[X], basis[Y] );
  bool b = basis[Z].unit();

  update();

  return b;
}


bool frame::create_from_normal(
    const vec3& P,	// point on xy the plane
    const vec3& N	// normal
    )
{
  eye_position = P;
  basis[Z] = N;
  bool b = basis[Z].unit();
  basis[X].perpendicular(basis[Z]);
  basis[X].unit();
  basis[Y] = cross( basis[Z], basis[X] );
  basis[Y].unit();

  update();

  return b;
}

//
// global point returned
vec3 frame::point_at( real s, real t ) const
{
  return (get_origin() + (s*x_axis()) + (t*y_axis()));
}

//
// of course global point returned
vec3 frame::point_at( real s, real t, real c ) const
{
  return (get_origin() + (s*x_axis()) + (t*y_axis()) + (c*z_axis()));
}


line frame::iso_line( // iso parametric line
       int dir,              // 0 first parameter varies and second parameter is constant
                         //   e.g., line(t) = plane(t,c)
                         // 1 first parameter is constant and second parameter varies
                         //   e.g., line(t) = plane(c,t)
       real c           // c = value of constant parameter 
       ) const
{
  if ( dir ) {
    return line(point_at(0.0, c), point_at(1.0, c));
  }
  else {
    return line(point_at(c, 0.0), point_at(c, 1.0));
  }
}

// compute intersection between the input line and xy plane!
bool frame::intersect(const vec3 line_a, const vec3 line_b, real& t) const
{
    real a = distance_to(line_a);
    real b = distance_to(line_b);

    if ((a > 0 && b > 0) || (a < 0 && b < 0)) return false;

    real d = a-b;
    bool rc = false;
    
    if (d != 0.0) {
	d = 1.0 / d;
	real fd = fabs(d);
	if ( fd > 1.0 && (fabs(a) >= kMaxReal /fd || fabs(b) >= kMaxReal/fd ) ) {
	    // real overflow - line is probably parallel to plane
	    t = 0.5;
	}
	else {
	    t = a*d;
	    rc = true;
	}
    }

    return rc;
}

// compute intersection between the input line and xy plane!
bool frame::intersect(const vec3 line_a, const vec3 line_b, real& t, vec3& pnt) const
{
    bool ret = intersect(line_a, line_b, t);
    if (ret) {
	pnt = line_a + ((line_b - line_a) * t);
    }

    return ret;
}

bool frame::intersect(const box3& a,  vector<vec3>& pnts) const
{
    vec3 m1 = a.get_minimum();
    vec3 m2 = a.get_maximum();

    vec3 p1(m1);
    vec3 p2(m2[0], m1[1], m1[2]);
    vec3 p3(m2[0], m2[1], m1[2]);
    vec3 p4(m1[0], m2[1], m1[2]);
    vec3 p5(m1[0], m1[1], m2[2]);
    vec3 p6(m2[0], m1[1], m2[2]);
    vec3 p7(m2);
    vec3 p8(m1[0], m2[1], m2[2]);

    vec3 pnt;
    real t;
    if (intersect(p1, p2, t, pnt)) {
	pnts.add(pnt);
    }
    if (intersect(p2, p3, t, pnt)) {
	pnts.add(pnt);
    }
    if (intersect(p3, p4, t, pnt)) {
	pnts.add(pnt);
    }
    if (intersect(p4, p1, t, pnt)) {
	pnts.add(pnt);
    }
    if(intersect(p1, p5, t, pnt)) {
	pnts.add(pnt);
    }
    if (intersect(p2, p6, t, pnt)) {
	pnts.add(pnt);
    }
    if (intersect(p3, p7, t, pnt)) {
	pnts.add(pnt);
    }
    if (intersect(p4, p8, t, pnt)) {
	pnts.add(pnt);
    }
    if (intersect(p5, p6, t, pnt)) {
	pnts.add(pnt);
    }
    if (intersect(p6, p7, t, pnt)) {
	pnts.add(pnt);
    }
    if (intersect(p7, p8, t, pnt)) {
	pnts.add(pnt);
    }
    if (intersect(p8, p5, t, pnt)) {
	pnts.add(pnt);
    }

    if (pnts.size()) return true;
    return false;
}

real frame::equation_at( const vec3& p) const
{
    vec4 equation;
    get_equation(equation);
    return equation[0]*p[0]+ equation[1]*p[1] + equation[2]*p[2] + equation[3];
}

real frame::equation_at( const vec4& p) const
{
    vec4 equation;
    get_equation(equation);
    return equation[0]*p[0]+ equation[1]*p[1] + equation[2]*p[2] + equation[3]*p[3];
}

bool frame::get_equation(vec4& equation) const
{
    // computes equation[] from origin and zaxis.
    equation[0] = basis[Z][0];
    equation[1] = basis[Z][1];
    equation[2] = basis[Z][2];
    equation[3] = -inner(get_origin(), basis[Z] );
    return (equation[0] != 0.0 || equation[1] != 0.0 || equation[2] != 0.0)?true:false;
}

bool frame::closest_point_to( const vec3& p, real& s, real& t ) const
{
  const vec3 v = p - get_origin();
    s = inner(v, x_axis());
    t = inner(v, y_axis());
  return true;
}

vec3 frame::closest_point_to( const vec3& p ) const
{
  real s, t;
  closest_point_to( p, s, t );
  return point_at( s, t );
}

void frame::swap() 
{
	real temp[16];

	memcpy((void*)temp, (void*)worldMatrix, sizeof(real)*16);
	memcpy((void*)worldMatrix, (void*)inverseWorldMatrix, sizeof(real)*16);
	memcpy((void*)inverseWorldMatrix, (void*)temp, sizeof(real)*16);

    reverse_update();
}

const vec3& frame::get_origin() const
{
    return eye_position;
}

void frame::set_origin(const vec3& p)
{
    set_eye_position(p);
}

void frame::set_eye_position(const vec3& p)
{
    eye_position = p;
    update();
}

void frame::set_look(const vec3& dir, const vec3& up)
{
    //| no change eye_position

    basis[Z] = unit(dir);
    basis[Y] = unit(up - proj(basis[Z], up)); 
    basis[X] = cross(basis[Y], basis[Z]);

    update();
}

frame& frame::operator = (const frame& a) {

    for(int i = 0 ; i < 3;++i)
	basis[i] = a.basis[i];
    eye_position = a.eye_position;

	memcpy(worldMatrix, a.worldMatrix, sizeof(real)*16);
	memcpy(inverseWorldMatrix, a.inverseWorldMatrix, sizeof(real)*16);

    return *this;
}

void frame::translate_frame(const vec3& a)
{
    eye_position += a;	
    update();
}

void frame::set_look(real angle, const vec3& axis)
{
    quater qa = orient(angle, axis);

    for(int i = 0 ; i < 3 ;++i)
	basis[i] = unit(rot(qa, basis[i]));

    update();
}

void frame::rotate_frame(real angle, const vec3& axis)
{
    set_look(angle, axis);
}

void frame::rotate_frame(const quater& qa)
{
    for (int i = 0 ; i < 3 ;++i)
	basis[i] = unit(rot(qa, basis[i]));
    update();
}

void frame::rotate_frame_locally(const quater& qa)
{
    vec3 t_basis[3];

    t_basis[0] = vec3(1.0, 0.0, 0.0);
    t_basis[1] = vec3(0.0, 1.0, 0.0);
    t_basis[2] = vec3(0.0, 0.0, 1.0);

    int i;
    for (i = 0 ; i < 3 ;++i){
	t_basis[i] = unit(rot(qa, t_basis[i]));

	//| model -> world
	t_basis[i] = unit(to_world(t_basis[i]) - eye_position);
    }

    for (i = 0 ; i < 3 ;++i)
	basis[i] = t_basis[i];

    update();
}

quater frame::to_quater() const
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

vec3 frame::to_translation() const
{
	return vec3(-inner(eye_position, basis[0]),
			    -inner(eye_position, basis[1]),
				-inner(eye_position, basis[2]));
}

void frame::from_quater(const quater& q)
{
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

	update();
}

void frame::from_translation(const vec3& t)
{
	eye_position[0] = -1.0 * (basis[0][0]*t[0] + basis[1][0]*t[1] + basis[2][0] * t[2]);
	eye_position[1] = -1.0 * (basis[0][1]*t[0] + basis[1][1]*t[1] + basis[2][1] * t[2]);
	eye_position[2] = -1.0 * (basis[0][2]*t[0] + basis[1][2]*t[1] + basis[2][2] * t[2]);
	update();
}

void frame::update()
{
    gl_identity(worldMatrix);
    gl_identity(inverseWorldMatrix);
    for (int i = 0 ; i < 3 ;++i){
	for (int j = 0 ; j < 3 ; ++j) {
	    real a = basis[i][j];
	    worldMatrix[i*4 + j] = a;
	    inverseWorldMatrix[j*4 + i] = a;
	}
    }
    worldMatrix[12] = eye_position[0];
    worldMatrix[13] = eye_position[1];
    worldMatrix[14] = eye_position[2];
    
    inverseWorldMatrix[12] = -inner(eye_position, basis[0]);
    inverseWorldMatrix[13] = -inner(eye_position, basis[1]);
    inverseWorldMatrix[14] = -inner(eye_position, basis[2]);
}

void frame::reverse_update()
{
    for (int i = 0 ; i < 3 ;++i)
		eye_position[i] = worldMatrix[12+i]/worldMatrix[15];

    for (int i = 0 ; i < 3 ;++i){
		for(int j = 0 ; j < 3 ; ++j)
			basis[i][j] = worldMatrix[i*4 + j];
    }
}


void frame::push_to_world() const
{
    glPushMatrix();
    gl_multmatrix((real*)worldMatrix); //
}

void frame::push_to_model() const
{
    glPushMatrix();
    gl_multmatrix((real*)inverseWorldMatrix); //
}

vec3 frame::to_model(const vec3& a) const
{
    vec4 ret(0);
    vec4 aa(a[0], a[1], a[2], 1.0);

    for (int i = 0 ; i < 4 ;++i)
	for (int j = 0 ; j < 4 ; ++j)
	    ret[i] += inverseWorldMatrix[j*4 + i] * aa[j];

    return vec3(ret[0]/ret[3], ret[1]/ret[3], ret[2]/ret[3]);
}

vec4 frame::to_model(const vec4& a) const
{
    vec4 ret(0);

    for (int i = 0 ; i < 4 ;++i)
	for (int j = 0 ; j < 4 ; ++j)
	    ret[i] += inverseWorldMatrix[j*4 + i] * a[j];

    return ret;
}

vec3 frame::to_world(const vec3& a) const
{
    vec4 ret(0);
    vec4 aa(a[0], a[1], a[2], 1.0);

    for (int i = 0 ; i < 4 ;++i)
	for (int j = 0 ; j < 4 ; ++j)
	    ret[i] += worldMatrix[j*4 + i] * aa[j];

    return vec3(ret[0]/ret[3], ret[1]/ret[3], ret[2]/ret[3]);
}

vec4 frame::to_world(const vec4& a) const
{
    vec4 ret(0);

    for (int i = 0 ; i < 4 ;++i)
	for (int j = 0 ; j < 4 ; ++j)
	    ret[i] += worldMatrix[j*4 + i] * a[j];

    return ret;
}

vec3 frame::to_model_normal(const vec3& a) const
{
    vec3 ret(0);

    for (int i = 0 ; i < 3 ;++i)
	for (int j = 0 ; j < 3 ; ++j)
	    ret[i] += inverseWorldMatrix[j*4 + i] * a[j];

    return ret;
}

vec3 frame::to_world_normal(const vec3& a) const
{
    vec3 ret(0);

    for (int i = 0 ; i < 3 ;++i)
	for (int j = 0 ; j < 3 ; ++j)
	    ret[i] += worldMatrix[j*4 + i] * a[j];

    return ret;
}

line frame::to_model(const line& a) const
{
    line new_l;
    new_l.set_position(to_model(a.get_position()));
    new_l.set_direction(to_model_normal(a.get_direction()));

    return new_l;
}

line frame::to_world(const line& a) const
{
    line new_l;
    new_l.set_position(to_world(a.get_position()));
    new_l.set_direction(to_world_normal(a.get_direction()));

    return new_l;
}

void frame::pop() const
{
    glPopMatrix();
}

void frame::make_disc_space(const vec3& org, const vec3& z_dir)
{
    frame m_ls;

    plane pl;
    pl.reset(org, z_dir);

    m_ls.set_origin(org);
    vec3 x = m_ls.to_world(vec3(1.0, 0, 0));
    vec3 y = m_ls.to_world(vec3(0, 1.0, 0));
    vec3 z = m_ls.to_world(vec3(0, 0, 1.0));

    line lx, ly, lz;
    lx.set_position(x); lx.set_direction(pl.get_normal());
    ly.set_position(y); ly.set_direction(pl.get_normal());
    lz.set_position(z); lz.set_direction(pl.get_normal());


    pl.intersect(lx, x);
    pl.intersect(ly, y);
    pl.intersect(lz, z);

    real normx = norm(x-org), normy = norm(y-org), normz = norm(z-org);

    if (normx >= normy && normx >= normz) {
	set_look(z_dir, x-org);
    }
    else if (normy >= normz && normy >= normx) {
	set_look(z_dir, y-org);
    }
    else if (normz >= normx && normz >= normy) {
	set_look(z_dir, z-org);
    }

    set_origin(org);
}


}; //namespace
