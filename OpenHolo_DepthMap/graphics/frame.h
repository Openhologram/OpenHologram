#ifndef __frame_h
#define __frame_h
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


namespace graphics {

class frame {

public:

    vec3 basis[3];
    vec3 eye_position;
    
	real worldMatrix[16];
	real inverseWorldMatrix[16];

    //matrix mat_1; 			//| transform to world coordinates
    //matrix mat_2; 			//| transform to model coordinates

    enum axis {
	X = 0,
	Y = 1,
	Z = 2
    };

    frame(int a = 0) 
    {
	//| eye : eye position, origin of new frame

	reset();
	reverse_update();
    }

    frame(const frame& val)
	: eye_position(val.eye_position)
    {
	basis[X] = val.basis[X];
	basis[Y] = val.basis[Y];
	basis[Z] = val.basis[Z];
	memcpy(worldMatrix, val.worldMatrix, sizeof(real)*16);
	memcpy(inverseWorldMatrix, val.inverseWorldMatrix, sizeof(real)*16);
    }

    frame(const vec3& dir, const vec3& up)
    {
	eye_position = vec3(0);

	basis[Z] = unit(dir);
	basis[Y] = unit(up - proj(basis[Z], up)); 
	basis[X] = cross(basis[Y], basis[Z]);

	update();
    }

    frame(const vec3& eye, const vec3& dir, const vec3& up) 
    {
	eye_position = eye;

	basis[Z] = unit(dir);
	basis[Y] = unit(up - proj(basis[Z], up)); 
	basis[X] = cross(basis[Y], basis[Z]);

	update();
    }

    frame(const vec3& eye, const vec3& x, const vec3& y, const vec3& z) 
    {
	eye_position = eye;

	basis[Z] = unit(z);
	basis[Y] = unit(y);
	basis[X] = unit(x);

	update();
    }

    const vec3& x_axis() const { return basis[X]; }
    vec3& x_axis() { return basis[X]; }
    const vec3& y_axis() const { return basis[Y]; }
    vec3& y_axis() { return basis[Y]; }
    const vec3& z_axis() const { return basis[Z]; }
    vec3& z_axis() { return basis[Z]; }

	void Transform(const frame& f);

    bool create_from_normal(
	const vec3& P,	// point on xy the plane
	const vec3& N	// normal
	);

    bool create_from_frame(
	const vec3&, // point on the plane
	const vec3&, // non-zero vector in plane
	const vec3&  // another non-zero normal in the plane
	);

    bool create_from_points(
	const vec3& P,  // point on the plane
	const vec3& Q,  // point on the plane
	const vec3& R   // point on the plane
	);

    /*
    Description:
    Evaluate a point on the plane
    Parameters:
    u - [in]
    v - [in] evaulation parameters
    Returns:
    plane.origin + u*plane.xaxis + v*plane.yaxis
    */
    vec3 point_at(
	real u,
	real v
    ) const;

    /*
    Description:
    Evaluate a point on the plane
    Parameters:
    u - [in]
    v - [in] evaluation parameters
    w - [in] elevation parameter
    Returns:
    plane.origin + u*plane.xaxis + v*plane.yaxis + z*plane.zaxis
    */
    vec3 point_at(
	real u,
	real v,
	real w
    ) const;

    /*
    Description:
    Get an isoparameteric line on the plane.
    Parameters:
    dir - [in] direction of iso-parametric line
	0: first parameter varies and second parameter is constant
	   e.g., line(t) = plane(t,c)
	1: first parameter is constant and second parameter varies
	   e.g., line(t) = plane(c,t)
    c - [in] value of constant parameter 
    Returns:
    iso-parametric line
    */
    line iso_line(
	 int dir,
	 real c
	 ) const;

    /*
    Description:
	Get signed distance from the plane to a point.
    Parameters:
	point - [in]
    Returns:
	Signed distance from a point to a plane.
    Remarks:
	If the point is on the plane, the distance is 0.
	If the point is above the plane, the distance is > 0.
	If the point is below the plane the distance is < 0.
	The zaxis determines the plane's orientation.
    */
    real distance_to( 
	const vec3& point
	) const {  return inner(( point - get_origin()), z_axis()); }

    // compute intersection between the input line and xy plane!
    bool intersect(const vec3 line_a, const vec3 line_b, real& t) const;


    // compute intersection between the input line and xy plane!
    bool intersect(const vec3 line_a, const vec3 line_b, real& t, vec3& pnt) const;

    // compute intersection between input box and xy plane!
    bool intersect(const box3& a,  vector<vec3>& pnts) const;

    /*
    Description:
	Evaluate the plane's equation at a euclidean 3d point.
    Parameters:
	point - [in] 3d point
    Returns:
	eqn[0]*point.x + eqn[1]*point.y + eqn[2]*point.z + eqn[3]
    */
    real equation_at( 
	const vec3& point
	) const;

    /*
    Description:
	Evaluate the plane's equation at a homogeneous 4d point.
    Parameters:
	point - [in] homogeneous 4d point
    Returns:
	eqn[0]*point.x + eqn[1]*point.y + eqn[2]*point.z + eqn[3]*point.w
    */
    real equation_at( 
	const vec4& point
	) const;

    /*
    Description:
	Get the xy-plane equation based on the current values
	of the origin and zaxis.
    Returns:
	true if successful.  false if zaxis is zero.
    */
    bool get_equation(vec4& equation) const;

    /*
    Description:
    Get point on plane that is closest to a given point.
    Parameters:
    world_point - [in] 3d point
    u - [out] 
    v - [out] The point ON_Plane::PointAt(*u,*v) is the point
	      on the plane that is closest to world_point.
    Returns:
    true if successful.
    */
    bool closest_point_to( 
	 const vec3& world_point,
	 real& u,
	 real& v
	 ) const;

    /*
    Description:
    Get point on plane that is closest to a given point.
    Parameters:
    point - [in]
    Returns:
    A 3d point on the plane that is closest to world_point.
    */
    vec3 closest_point_to( 
	 const vec3& point
	 ) const;

    //
    // return xy plane : this is not equation
    //
    plane get_xy_plane() const { return plane(basis[Z], eye_position); } 

    void reset();

    void set_with(const plane& a);

    void swap();

    const vec3& get_origin() const;

    void set_origin(const vec3& p);

    void set_eye_position(const vec3& p);


    void set_look(const vec3& dir, const vec3& up);

    virtual frame& operator = (const frame& a) ;

    /*|----------------------------------------------------------------------*/
    /*|  translate frame by a vector                                         */
    /*|______________________________________________________________________*/
    void translate_frame(const vec3& a);


    /*|----------------------------------------------------------------------*/
    /*| make a frame which has z_dir vector as a z-axis.		     */
    /*|----------------------------------------------------------------------*/
    void make_disc_space(const vec3& org, const vec3& z_dir);

    /*|----------------------------------------------------------------------*/
    /*| rotate frame                                                         */
    /*|______________________________________________________________________*/
    void set_look(real angle, const vec3& axis) ;

 
    /*|----------------------------------------------------------------------*/
    /*| rotate frame with quaternion                                         */
    /*|______________________________________________________________________*/
    void rotate_frame(real angle, const vec3& axis);

    /*|----------------------------------------------------------------------*/
    /*| rotate frame with quaternion                                         */
    /*|______________________________________________________________________*/

    void rotate_frame(const quater& qa) ;

    /*|----------------------------------------------------------------------*/
    /*| rotate frame with quaternion : but this quaternion has been taken    */
    /*| from local space defined by this frame.                              */
    /*|______________________________________________________________________*/
    void rotate_frame_locally(const quater& qa);

    /*|----------------------------------------------------------------------*/
    /*|Construct matrix from orthogonal bases                                */
    /*|______________________________________________________________________*/
    void update();

    /*|----------------------------------------------------------------------*/
    /*| update viewing parameter information from the matrix                 */
    /*|______________________________________________________________________*/
    void reverse_update();


    /*|----------------------------------------------------------------------*/
    /*| Notice : Matrix Order of OpenGL differs from that of our math        */
    /*|          Column Major : OPENGL                                       */
    /*|          Row    Major : Our Scheme                                   */
    /*|______________________________________________________________________*/
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

	virtual vec3 get_scale() const { return vec3(1); }
    void pop() const;

	// from basis vectors, compute the quaternion
	quater to_quater() const;

	// from quaternion, define the basis vectors; this is used for vision applications
	void from_quater(const quater& q);

	// translation vector from matrix
	vec3 to_translation() const;

	// from translation, define eye_position
	void from_translation(const vec3& t);

	void test();
};



/*|~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*/
/*|  Note : Matrix Order of OpenGL differs from that of our math             */
/*|__________________________________________________________________________*/
vec3 operator * (real a[], const vec4& b);

};
#endif
