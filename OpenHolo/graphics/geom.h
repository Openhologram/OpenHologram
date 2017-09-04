#ifndef __geom_h
#define __geom_h

#include "graphics/sys.h"
#include "graphics/real.h"
#include "graphics/vec.h"
#include "graphics/vector.h"
#include "graphics/matrix.h"
#include "graphics/epsilon.h"
#include "graphics/quater.h"

#include "graphics/minmax.h"


namespace graphics {


struct box2 {

    vec2	minimum; 
    vec2	maximum;

    //--------------------------------
    //  constructors                  
    //________________________________
    box2() 
    { 
	make_empty(); 
    }

    box2(real xmin, real ymin, real xmax, real ymax)
    { 
	minimum = vec2(xmin, ymin); 
	maximum = vec2(xmax, ymax); 
    }

    box2(const vec2& _min, const vec2& _max)
    { 
	minimum = _min; 
	maximum = _max; 
    }

    vec2& get_minimum() { return minimum; }
    const vec2& get_minimum() const { return minimum; }

    vec2& get_maximum() { return maximum; }
    const vec2& get_maximum() const { return maximum; }
    
    vec2 get_center()  const 
    { 
	return 
	vec2(0.5 * (minimum[0] + maximum[0]),0.5 * (minimum[1] + maximum[1]));
    }

    const box2& operator = (const box2& a)
    {
	minimum = a.minimum;
	maximum = a.maximum;

	return *this;
    }

    void extend(const vec2& pt);

    void extend(const box2& r);

	void extend_epsilon(real r);

    bool has_intersection(const vec2& pt) const;

    bool has_intersection(const box2& bb) const;

    void set_bounds(real xmin, real ymin, real xmax, real ymax)
    { 
	minimum = vec2(xmin, ymin); 
	maximum = vec2(xmax, ymax); 
    }

    void set_bounds(const vec2& _min, const vec2& _max)
    { 
	minimum = _min; 
	maximum = _max; 
    }

    void get_bounds(real &xmin, real &ymin, real &xmax, real &ymax) const
    { 
	xmin = minimum[0]; ymin = minimum[1]; 
	xmax = maximum[0]; ymax = maximum[1];
    }

    void get_bounds(vec2& _min, vec2& _max) const
    { 
	_min = minimum; 
	_max = maximum; 
    }

    vec2 get_closest_point(const vec2& point) const;
    
    void get_origin(real& orgx, real& orgy) const
    { 
	orgx = minimum[0]; 
	orgy = minimum[1]; 
    }

    void get_size(real& sizex, real& sizey) const
    { 
	sizex = maximum[0] - minimum[0]; 
	sizey = maximum[1] - minimum[1]; 
    }

    void make_empty();

    bool is_empty() const { return maximum[0] < minimum[0]; }
    bool has_area() const { return (maximum[0] > minimum[0] && maximum[1] > minimum[1]); }

	void translate (const vec2& val) { maximum+= val; minimum+= val; }
	void intersect(const box2& v) {
		if (v.is_empty()) {
			*this = v;
			return;
		}

		if (is_empty()) {
			return;
		}

		for (int i = 0 ; i < 2 ;++i) {
			if (v.minimum[i] > minimum[i]) minimum[i] = v.minimum[i];
			if (v.maximum[i] < maximum[i]) maximum[i] = v.maximum[i];
		}
	}
	// include circle 
	bool include(const vec2& pos, real rad) {

		if (!has_intersection(pos)) return false;
		vec2 p1 = pos;
		p1[0] -= rad;
		if (!has_intersection(p1)) return false;
		p1 = pos;
		p1[1] -= rad;
		if (!has_intersection(p1)) return false;
		p1 = pos;
		p1[0] += rad;
		if (!has_intersection(p1)) return false;
		p1 = pos;
		p1[1] += rad;
		if (!has_intersection(p1)) return false;
		return true;
	}
};

struct line;

struct box3 {

    vec3	minimum; 
    vec3	maximum;

    box3() 
    { 
	make_empty(); 
    }

    box3(real xmin, real ymin, real zmin,
	 real xmax, real ymax, real zmax)
    { 
	minimum = vec3(xmin, ymin, zmin); 
	maximum = vec3(xmax, ymax, zmax); 
    }

    box3(const vec3& _min, const vec3& _max) : minimum(_min), maximum(_max)
    { 
    }

    vec3& get_minimum() { return minimum; }
    const vec3& get_minimum() const { return minimum; }

    vec3& get_maximum() { return maximum; }
    const vec3& get_maximum() const { return maximum; }

    vec3 get_center() const
    {
	vec3 ret(0.5 * (minimum[0] + maximum[0]),
		 0.5 * (minimum[1] + maximum[1]),
		 0.5 * (minimum[2] + maximum[2]));
	return ret;
    }

    box3& operator = (const box3& a)
    {
		minimum = a.minimum;
		maximum = a.maximum;

		return *this;
    }

    void extend(const vec3& pt);

    void extend(const box3& a);

	// extend the box by the epsilon
	void extend_epsilon(real lval);

    // has intersection with pt??
    bool has_intersection(const vec3& pt) const;

    // has intersection with bb??  
    bool has_intersection(const box3& bb) const;

	bool intersect(const line& l, vec3& ret);

    void set_bounds(real xmin, real ymin, real zmin,
		real xmax, real ymax, real zmax)
    { 
	minimum = vec3(xmin, ymin, zmin); 
	maximum = vec3(xmax, ymax, zmax); 
    }

    void  set_bounds(const vec3& _min, const vec3& _max)
    { 
	minimum = _min; 
	maximum = _max; 
    }

    void  get_bounds(real &xmin, real &ymin, real &zmin,
		     real &xmax, real &ymax, real &zmax) const
    { 
	xmin = minimum[0]; ymin = minimum[1]; zmin = minimum[2]; 
	xmax = maximum[0]; ymax = maximum[1]; zmax = maximum[2]; 
    }

    void  get_bounds(vec3& _min, vec3& _max) const
    { 
	_min = minimum; 
	_max = maximum; 
    }

    // get point from this  closest to point	
    vec3 get_closest_point(const vec3& point) const;
    
    void  get_origin(real &originX, real &originY, real &originZ) const
    { 
	originX = minimum[0]; 
	originY = minimum[1]; 
	originZ = minimum[2]; 
    }

    void  get_size(real &sizeX, real &sizeY, real &sizeZ) const
    { 
	sizeX = maximum[0] - minimum[0];
	sizeY = maximum[1] - minimum[1];
        sizeZ = maximum[2] - minimum[2]; 
    }

    void  make_empty();

    bool is_empty() const { return maximum[0] < minimum[0]; }

    bool has_volume() const
    { 
	return (maximum[0] > minimum[0] && 
		maximum[1] > minimum[1] && 
		maximum[2] > minimum[2]); 
    }

    void transform(const matrix& m);

    real  get_volume() const
    {
	if (is_empty()) return 0.0;
	return (maximum[0] - minimum[0]) * 
	       (maximum[1] - minimum[1]) * 
	       (maximum[2] - minimum[2]);
    }

	void print() const;

	void translate (const vec3& val) { maximum+= val; minimum+= val; }

	void intersect(const box3& v) {
		if (v.is_empty()) {
			*this = v;
			return;
		}

		if (is_empty()) {
			return;
		}

		for (int i = 0 ; i < 3 ;++i) {
			if (v.minimum[i] > minimum[i]) minimum[i] = v.minimum[i];
			if (v.maximum[i] < maximum[i]) maximum[i] = v.maximum[i];
		}
	}
};



struct line {

  private :

    vec3	pos;
    vec3	dir;

  public :

    line () 
    { 
    }

    //-------------------------------------------------------------------------
    // As input, two points, p0 and p1, are given. The line will be formed
    // as it has direction, p1-p0, and position p0
    line (const vec3& p0, const vec3& p1) { set_value(p0, p1); }
    //__________________________________________________________


    //-------------------------------------------------------------------------
    // copy constructor
    line (const line& a)  { pos = a.pos; dir = a.dir;   }
    //___________________________________________________


    line& operator = (const line& a);

    void set_value(const vec3& p0, const vec3& p1);

    const vec3& get_position() const  { return pos;  }

    const vec3& get_direction() const { return dir;  }
    
    vec3& get_position() { return pos;  }

    vec3& get_direction() { return dir;  }

    void set_position(const vec3& a)  { pos = a; }

    void set_direction(const vec3& a)   { dir = a; }

    bool  get_closest_points
	(const line& l, vec3& ptOnThis, vec3& ptOnl) const;

    vec3 get_closest_point(const vec3& pnt) const;

    bool  intersect(const box3& bb, vec3& enter, vec3& exit) const;


    ///------------------------------------------------------------------------
    // Intersect the line with a 3D box.  The line is augmented with an  
    // angle to form a cone.  The box must lie within pickAngle of the  
    // line. If the angle is < 0.0, abs(angle) is the radius of a         
    // cylinder for an orthographic intersection. (OpenInventor)           
    //
    bool intersect(real ang, const box3& box) const;
    //______________________________________________


    bool intersect(real pickAngle,const vec3& point) const    ;

    ///------------------------------------------------------------------------
    // Intersection between line(v0, v1) and this 
    //
    bool intersect(real ang, const vec3& v0, const vec3& v1, vec3 &intersection) const;
    //_________________________________________________________________________________


    //-------------------------------------------------------------------------
    // Intersection with triangle (Open Inventor code at http://www.sgi.com)
    //
    int intersect(const vec3& v0, const vec3& v1, const vec3& v2,
		    vec3 &intersection,
		    vec3 &barycentric, 
		    int &front) const;
    //________________________________

    // Is the point rhs is within the distance eps from this line?
    bool is_on(const vec3& rhs, real eps) const;

};

struct plane {

    vec3 n;
    real d;				// n * p = d

    plane() : n(0), d(0) { }
    plane(const vec3& nn, real dd = 0) : n(nn), d(dd) { }
    plane(const vec3& dir, const vec3& pl_pnt) 
    { 
	reset(pl_pnt, dir);
    }

    plane(const plane& a) : n(a.n), d(a.d) { }

    //-------------------------------------------------------------------------
    // plane contains p, q, r
    plane(const vec3& p, const vec3& q, const vec3& r);
    //__________________________________________________

    void reset(const vec3& p, const vec3& q, const vec3& r) ;
    void reset(const vec3& p, const vec3& normal) ;
    void reset(const vec3& nn, real dd) ;

    plane& operator = (const plane& a) ;
    
    const vec3& get_normal() const;
    vec3& get_normal();

    real get_dist() const ;

	real signed_distance(const vec3& input) const { return inner(input, n) - d; }
    void offset(real a) ;
    bool intersect(const line& l, vec3& intersection) const ;

    //-------------------------------------------------------------------------
    // Is the input point on this plane?
    bool is_on(const vec3& rhs) const;
    //_________________________________

	bool is_same_plane(const plane &pl, real eps = 0.0000001) const;

	bool intersect(const plane& pl, line& is);
};

struct cylinder {
  
  private:

    line	axis;
    real	radius;

  public:

    
    cylinder()
    {
	axis.set_value(vec3(0.0, 0.0, 0.0), vec3(0.0, 1.0, 0.0));
	radius = 1.0;
    }

    // construct a cylinder given an axis and radius	
    cylinder(const line &a, real r)
    {
	axis = a;
	radius = r;
    }

    cylinder(const cylinder& a): axis(a.axis), radius(a.radius)
    {
    }

    // Change the axis and radius  
    void  set_value(const line& a, real r)
    {
	axis = a;
	radius = r;
    }

    // Set just the axis or radius 
    void    set_axis(const line& a)
    {
	axis = a;
    }
    void    set_radius(real r)
    {
	radius = r;
    }

    // Return the axis and radius  
    line    get_axis() const  { return axis; }
    real    get_radius() const { return radius; }


    bool    intersect(const line &l, vec3& isect) const
    {
	vec3 whoCares;
	return intersect(l, isect, whoCares);

    }

    bool    intersect(const line& l, vec3& enter, vec3& exit) const
    {
	quater	to_y_axis = u2v_quater(axis.get_direction(), vec3(0, 1, 0)); 

	vec3	scale(1);
	scale	= scale / radius;
	vec3	i_scale = 1.0 / scale;
	quater  i_to_y_axis = inv(to_y_axis);

	// find the given l un-translated  
	vec3	origin = l.get_position();
	origin	= origin - axis.get_position();
	vec3	dest = origin + l.get_direction();

	// ordering would be any problem.  
	//  1. rotate			    
	origin  = rot(to_y_axis, origin);
	dest = rot(to_y_axis, dest);

	// 2. scale			    
	origin  = scale * origin;
	dest = scale * dest;

	line cylLine(origin, dest);

	// find the intersection on the unit cylinder	
	vec3 cylEnter, cylExit;

	bool intersected = unit_cylinder_intersect(cylLine, cylEnter, cylExit);

	if (intersected) {

	    // transform back to original space    

	    // 1. iscale   
	    enter = i_scale * cylEnter;

	    //	 2. irot    
	    enter = rot(i_to_y_axis, enter);

	    enter += axis.get_position();

	    // 1. iscale   
	    exit = i_scale * cylExit;
	    //	 2. irot    
	    exit = rot(i_to_y_axis, exit);

	    exit += axis.get_position();
	}    

	return intersected;
    }

  private :


    //----------------------------------------------------------------------
    // Taken from Pat Hanrahan's chapter in Glassner's                      
    // _Intro to Ray Tracing_, page 91, and some code                       
    // stolen from Paul Strauss.					     
    //______________________________________________________________________

    static bool unit_cylinder_intersect(const line& l, vec3& enter, vec3& exit)
    {
	real	A, B, C, discr, sqroot, t0, t1;
	vec3	pos = l.get_position();
	vec3	dir = l.get_direction();
	bool	doesIntersect = true;

	A = dir[0] * dir[0] + dir[2] * dir[2];

	B = 2.0 * (pos[0] * dir[0] + pos[2] * dir[2]);

	C = pos[0] * pos[0] + pos[2] * pos[2] - 1;

	// discriminant = B^2 - 4AC	
	discr = B*B - 4.0*A*C;

	// if discriminant is negative, no intersection    
	if (discr < 0.0) {
	    doesIntersect = false;
	} else {
	    sqroot = sqrt(discr);

	    // magic to stabilize the answer	
	    if (B > 0.0) {
		t0 = -(2.0 * C) / (sqroot + B);
		t1 = -(sqroot + B) / (2.0 * A);
	    } else {
		t0 = (2.0 * C) / (sqroot - B);
		t1 = (sqroot - B) / (2.0 * A);
	    }	    

	    enter = pos + (dir * t0);
	    exit = pos + (dir * t1);
	}

	return doesIntersect;
    }
};

struct cone {
public:
    // An acute cone is Dot(A,X-V) = |X-V| cos(T) where V is the vertex, A
    // is the unit-length direction of the axis of the cone, and T is the
    // cone angle with 0 < T < PI/2.  The cone interior is defined by the
    // inequality Dot(A,X-V) >= |X-V| cos(T).  Since cos(T) > 0, we can avoid
    // computing square roots.  The solid cone is defined by the inequality
    // Dot(A,X-V)^2 >= Dot(X-V,X-V) cos(T)^2.

    // construction
    cone ();  // uninitialized
    cone (const vec3& rkVertex, const vec3& rkAxis,
        real fAngle);
    cone (const vec3& rkVertex, const vec3& rkAxis,
        real fCosAngle, real fSinAngle);

	bool intersect(const line& l, vec3& enter, vec3& exit) const;

	// make a cone with one axis and two surface points
	bool set_cone(const line& axis, const vec3& p1, const vec3& p2);

    vec3 Vertex;
    vec3 Axis;
    real CosAngle, SinAngle;  // cos(T), sin(T)
};


struct sphere {

    vec3	o;				// origin
    real 	r;				// radius

    sphere() : o(0), r(0) 
    {
    }

    sphere(vec3 oo, real rr) : o(oo), r(rr) 
    { 
    }

    sphere(const sphere& a) : o(a.o), r(a.r) 
    { 
    }

    inline void set_value(const vec3& _o, real _r)
    {
	r = _r;
	o = _o;
    }

    inline void set_center(const vec3& _o)
    {
	o = _o;
    }

    inline void set_radius(real _r)
    {
	r = _r;
    }

    inline vec3 get_center() const
    {
	return o;
    }

    inline real get_radius() const
    {
	return r;
    }


    //---------------------------------------------------------
    // sphere & line intersection : refer to Glassner Book,    
    //                              Introduction to RayTracing 
    //_________________________________________________________
    bool intersect(const line &l, vec3 &intersection) const ;
    bool intersect(const line& l, vec3& enter, vec3& exit) const;
};

struct plane2 {
    vec2 n;
    real d;					// n * p = d

    plane2() : n(0), d(0) { }
    plane2(const vec2& nn, real dd) : n(nn), d(dd) { }

    plane2(const vec2& p, const vec2& q);	// plane contains p, q, r
};

// functions

real 
diameter(const vector<vec3>& poly, int& i_dia, int& j_dia);

vec3 
mean_center(const vector<vec3>& poly);

vec3 
center(const vector<vec3>& poly);

real 
diameter(const vector<vec3>& poly);

sphere 
bsphere(const vector<vec3>& poly);	// bounding sphere

real 
dist(const plane& pl, const vec3& p);	// distance

real 
dist(const plane2& pl, const vec2& p);	// distance

real 
dist(const plane& pl, const vector<vec3>& poly); // distance

//						
// distance from line (p, q) to a point, a.	
//						
real 
dist(const vec2& p, const vec2& q, const vec2& a);

int 
angle_sign(const vec2& a, const vec2& b);

int 
is_convex(const vector<vec2>& input);

int 
is_pl_ccw(const std::vector<vec2>& input);

void 
bbox(const vector<vec3>& input, vec3& _min, vec3& _max);

void 
bbox(const vector<vec2>& input, vec2& _min, vec2& _max);

real 
min_dist(const ivec2& p, const ivec2& q, const ivec2& a);

real 
min_dist(const vec2 &p, const vec2 &q, const vec2 &a);

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// min_dist : computes distance between line(p,q) and                       
//           vertex a. Besides, this computes the parameter                 
//           value uu at which the shortest point is located.               
//                                                                          
//           0        uu     1.0                                            
//           p--------.------q                                              
//                                                                          
//                    a                                                     
// Return   : the distance between a and the point pq(uu)                   
//__________________________________________________________________________
real 
min_dist(const vec2 &p, const vec2 &q, const vec2 &a, real& uu);

real 
min_dist(const vec3 &p, const vec3 &q, const vec3 &a);

// min_dist : computes distance between line(p,q) and    
//           vertex a. Besides, this computes the parameter
//           value uu at which the shortest point is located.               
//                                                                          
//           0        uu     1.0                                            
//           p--------.------q                                              
//                                                                          
//                    a                                                     
// Return   : the distance between a and the point pq(uu)                   
//__________________________________________________________________________
real 
min_dist(const vec3 &p, const vec3 &q, const vec3 &a, real& uu);

real 
min_dist(const vec4 &p, const vec4 &q, const vec4 &a);

//
// min_dist : computes distance between line(p,q) and                       
//           vertex a. Besides, this computes the parameter                 
//           value uu at which the shortest point is located.               
//                                                                          
//           0        uu     1.0                                            
//           p--------.------q                                              
//                                                                          
//                    a                                                     
// Return   : the distance between a and the point pq(uu)                   
real 
min_dist(const vec4 &p, const vec4 &q, const vec4 &a, real& uu);

//
// Compute intersection of two 3D lines.
// return true if there is intersection. It does not process degenerate cases.
//  Just returns false if there occurs any degenerate case.
bool    intersect(const vec3& l1p, const vec3& l1q, // line l1
		  const vec3& l2p, const vec3& l2q, // line l2
		  vec2& ret, vec3& pnt);


struct line2 {

    vec2    p, q;
    vec2    dir;
    real    d;

    line2() { }

    line2(const vec2 &pp, const vec2 &qq)
    { 
	set_points(pp, qq);
    }

    line2& operator = (const line2& a);

    void set_points(const vec2 &pp, const vec2 &qq)
    {
	p = pp;
	q = qq;

	d = norm(q - p);
	dir = (q - p)/d;
    }

    bool on_half_open_edge(const vec2 &x, real& t) const;

    bool on_closed_edge(const vec2 &x) const;

    bool on_line(const line2& ll) const;

    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    // compute Line/line intersection :                                    
    // ret : 0 : intersection happened at out of domain of s1 or s2        
    //       1 : intersection was found                                    
    //       2 : intersection was found but degenertely parallel.          
    //           in this case further investigation is needed.             
    //       3 : intersection was found at the end point                   
    // var :                                                               
    //    t[0] : parameter value t, t <- [0, 1] of s1, where intersection  
    //           was found.                                                
    //    t[1] : parameter value t, t <- [0, 1] of s2, where intersection  
    //           was found.                                                
    //      p  : intersection point                                        
    //_____________________________________________________________________
    int intersect(const line2& ll, vec2& pp, vec2& t) const;

    inline vec2 get_closest_point(const vec2& pnt) const
    {
	vec2 prj = (inner((pnt - p), dir) * dir) + p;

	return prj;
    }
	
	inline real get_dist(const vec2& pnt) const
	{
		return norm(get_closest_point(pnt) - pnt);
	}

    vec2 get_closest_point(const vec2& pnt, real& t) const;

    int collect_coincidence(line2& ll, vec2& t) const;
};

struct param_line2 {
    vec2 p, q;
    real u_0, u_1;
    param_line2() { }
    param_line2(const vec2& pp, const vec2& qq) : p(pp), q(qq), u_0(0.0), u_1(1.0)
    { 
    }
    param_line2(const vec2& pp, const vec2& qq, real a, real b) 
    : p(pp), q(qq), u_0(a), u_1(b)
    {
    }
};


//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// returns twice the area of the oriented triangle (a, b, c).               
//__________________________________________________________________________
inline real tri_area(const vec2& a, const vec2& b, const vec2& c)
{
    return (b[0] - a[0])*(c[1] - a[1]) - (b[1] - a[1])*(c[0] - a[0]);
}
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// Returns 1    if the point d is inside the circle defined by the          
// points a, b, c. See Guibas and Stolfi (1985) p.107.                      
//__________________________________________________________________________
int in_circle(const vec2& a, const vec2& b, const vec2& c, const vec2& d);

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// Returns 1    if the points a, b, c are in a counterclockwise order       
//__________________________________________________________________________
int ccw(const vec2& a, const vec2& b, const vec2& c);

bool ccw(const vec3& a, const vec3& b, const vec3& c, const vec3& n);
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// Returns 1    if the points a, b, c are in a clockwise order              
//__________________________________________________________________________
int cw(const vec2& a, const vec2& b, const vec2& c) ;

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// Returns 1    if the points a, b, c are in a counterclockwise order       
//__________________________________________________________________________
int epsilon_ccw(const vec2& a, const vec2& b, const vec2& c);

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// Returns 1    if the points a, b, c are in a clockwise order              
//__________________________________________________________________________
int epsilon_cw(const vec2& a, const vec2& b, const vec2& c) ;


//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// Returns the center of the circle through points a, b, c.                 
// From Graphics Gems I, p.22                                               
//__________________________________________________________________________
vec2 circumcenter(const vec2& a, const vec2& b, const vec2& c) 
;

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// check if x is right of edge(org, dest)                                   
//__________________________________________________________________________
int is_right_point(const vec2& x, const vec2& org, const vec2& dest);

int is_left_point(const vec2& x, const vec2& org, const vec2& dest);

int on_edge(const vec2& x, const vec2& org, const vec2& dest);
bool on_edge(const vec3& x, const vec3& org, const vec3& dest);

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// snap x towards the line segment ab                                       
//__________________________________________________________________________
vec2 snap(const vec2 &x, const vec2& a, const vec2& b);

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// check if there is overlap of bboxes of s1 and s2.                        
// But, now it does not work correctly.                                     
//__________________________________________________________________________
int bbox_test(const line2& s1, const line2& s2);

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// compute Line/line intersection :                                    
// ret : 0 : intersection happened at out of domain of s1 or s2        
//       1 : intersection was found                                    
//       2 : intersection was found but degenertely parallel.          
//           in this case further investigation is needed.             
//       3 : intersection happend at the end point                     
// var :                                                               
//    t[0] : parameter value t, t <- [0, 1], of s1, where intersection 
//           was found.                                                
//    t[1] : parameter value t, t <- [0, 1], of s2, where intersection 
//           was found.                                                
//      p  : intersection point                                        
//_____________________________________________________________________

int intersect(const line2& s1, const line2& s2, vec2& p, vec2& t);

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// compute Line/line intersection :                                         
// ret : 0 : intersection was not found                                     
//       1 : intersection was found                                         
// var : ret : intersection point                                           
//__________________________________________________________________________
int intersect(const line2& s1, const line2& s2, vec2& ret, bool degen = true);

int intersect(const vec2& v1, const vec2& v2,
			  const vec2& u1, const vec2& u2, 
			  vec2& ret);

int contain(const std::vector<vec2>& pl, const vec2& p);

//--------------------------------------------------------------------------
// point containment test using Hormann winding number test                 
// see the paper, "The Point-in-polygon problem for arbitrary polygons",    
// Journal computational geometry, 1999.                                    
//--------------------------------------------------------------------------
int kai_contain(const std::vector<vec2>& P, const vec2& R);
int kai_contain(const graphics::vector<vec2>& P, const vec2& R);
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// compute least square fit to the sample data points and return the error. 
//__________________________________________________________________________
real least_square_line(const std::vector<vec2>& points, line2& l);


inline bool same_side(const vec3& p1, const vec3& p2, const vec3& a, const vec3& b)
{
	vec3 cp1 = cross(b-a, p1-a);
	vec3 cp2 = cross(b-a, p2-a);
	return (inner(cp1, cp2) >= 0.0);
}

inline bool point_in_triangle(const vec3& p, const vec3& a, const vec3& b, const vec3& c)
{
	line aa(a, b);
	if (aa.is_on(c, user_epsilon)) return false;
	if (aa.is_on(p, user_epsilon)) return true;
	line bc(b, c);
	if (bc.is_on(p, user_epsilon)) return true;
	line ca(c, a);
	if (ca.is_on(p, user_epsilon)) return true;
	return same_side(p, a, b, c) && same_side(p, b, a, c) && same_side(p, c, a, b);
}


inline bool point_in_triangle(const vec3& p, const vec3& a, const vec3& b, const vec3& c, real eps)
{
	line aa(a, b);
	if (aa.is_on(c, eps)) return false;
	if (aa.is_on(p, eps)) return true;
	line bc(b, c);
	if (bc.is_on(p, eps)) return true;
	line ca(c, a);
	if (ca.is_on(p, eps)) return true;
	return same_side(p, a, b, c) && same_side(p, b, a, c) && same_side(p, c, a, b);
}
const double kTempEpsilon = 0.000000000001f;
#define inline_cross(dest,v1,v2)	dest[0]=v1[1]*v2[2]-v1[2]*v2[1]; dest[1]=v1[2]*v2[0]-v1[0]*v2[2];   dest[2]=v1[0]*v2[1]-v1[1]*v2[0];
#define inline_dot(v1,v2)			(v1[0]*v2[0]+v1[1]*v2[1]+v1[2]*v2[2])
#define inline_sub(dest,v1,v2)		dest[0]=v1[0]-v2[0]; dest[1]=v1[1]-v2[1];  dest[2]=v1[2]-v2[2]; 

inline bool
intersect_triangle(double orig[3], double dir[3], // ray origin and ray dir
                   double vert0[3], double vert1[3], double vert2[3], // triangle
                   double *t, double *u, double *v) // t: distance, u,v
{
   double edge1[3], edge2[3], tvec[3], pvec[3], qvec[3];
   double det,inv_det;

   /* find vectors for two edges sharing vert0 */
   inline_sub(edge1, vert1, vert0);
   inline_sub(edge2, vert2, vert0);

   /* begin calculating determinant - also used to calculate U parameter */
   inline_cross(pvec, dir, edge2);

   /* if determinant is near zero, ray lies in plane of triangle */
   det = inline_dot(edge1, pvec);

#ifdef TEST_CULL           /* define TEST_CULL if culling is desired */
   if (det < kTempEpsilon)
      return false;

   /* calculate distance from vert0 to ray origin */
   inline_sub(tvec, orig, vert0);

   /* calculate U parameter and test bounds */
   *u = inline_dot(tvec, pvec);
   if (*u < 0.0 || *u > det)
      return false;

   /* prepare to test V parameter */
   inline_cross(qvec, tvec, edge1);

    /* calculate V parameter and test bounds */
   *v = inline_dot(dir, qvec);
   if (*v < 0.0 || *u + *v > det)
      return false;

   /* calculate t, scale parameters, ray intersects triangle */
   *t = inline_dot(edge2, qvec);
   inv_det = 1.0 / det;
   *t *= inv_det;
   *u *= inv_det;
   *v *= inv_det;
#else                    /* the non-culling branch */
   if (det > -kTempEpsilon && det < kTempEpsilon)
     return false;
   inv_det = 1.0 / det;

   /* calculate distance from vert0 to ray origin */
   inline_sub(tvec, orig, vert0);

   /* calculate U parameter and test bounds */
   *u = inline_dot(tvec, pvec) * inv_det;
   if (*u < 0.0 || *u > 1.0)
     return false;

   /* prepare to test V parameter */
   inline_cross(qvec, tvec, edge1);

   /* calculate V parameter and test bounds */
   *v = inline_dot(dir, qvec) * inv_det;
   if (*v < 0.0 || *u + *v > 1.0)
     return false;

   /* calculate t, ray intersects triangle */
   *t = inline_dot(edge2, qvec) * inv_det;
#endif
   return true;
}


struct implicit_line {

    real a, b, c;
    vec2 e1, e2;

    implicit_line() : a(0), b(0), c(0), e1(0), e2(0)
    {
    }

    implicit_line(const vec2& _e1, const vec2& _e2) ;


    void set(const vec2& aa, const vec2& bb);


    inline real dist(const vec2& x) const
    {
	return (a*x[0] + b*x[1] + c);
    }

    inline vec2 coord(const vec2& x) const
    {
	vec2 ret(inner(unit(e2-e1), x-e1), dist(x));
	return ret;
    }

    inline vec2 eval(const vec2& x) const
    {
	vec2 ret(inner(unit(e2-e1), x-e1), dist(x));
	return ret;
    }

    implicit_line& operator = (const implicit_line& _a)
    {
	a  = _a.a;
	b  = _a.b;
	c  = _a.c;

	e1 = _a.e1;
	e2 = _a.e2;

	return *this;
    }

};

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// these operators show if the point is above or below the line input       
//__________________________________________________________________________

int operator == (const vec2& point, const implicit_line& l);

int operator < (const vec2& point, const implicit_line& l);

int operator > (const vec2& point, const implicit_line& l);


plane least_square_plane(const std::vector<vec3>& points, real max_error = 1.0e-8);
line2 least_square_line(const std::vector<vec2>& points, real max_error = 2.0);
vec3 least_square_fit(const std::vector<vec3>& points);

void LineRectClip(double& x0, double& y0,double& x1, double& y1, const box2& box);
}; //namespace graphics
#endif
