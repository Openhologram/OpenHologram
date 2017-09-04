// xform.h: interface for the xform class.
//
//////////////////////////////////////////////////////////////////////
/* $Header: /src3/opennurbs/opennurbs_xform.h 14    9/19/02 2:01p Dalelear $ */
/* $NoKeywords: $ */
/*
//
// Copyright (c) 1993-2001 Robert McNeel & Associates. All rights reserved.
// Rhinoceros is a registered trademark of Robert McNeel & Assoicates.
//
// THIS SOFTWARE IS PROVIDED "AS IS" WITHOUT EXPRESS OR IMPLIED WARRANTY.
// ALL IMPLIED WARRANTIES OF FITNESS FOR ANY PARTICULAR PURPOSE AND OF
// MERCHANTABILITY ARE HEREBY DISCLAIMED.
//				
// For complete openNURBS copyright information see <http://www.opennurbs.org>.
//
////////////////////////////////////////////////////////////////
*/
#if !defined(__xform_h)
#define __xform_h

#include "graphics/vec.h"
#include "graphics/frame.h"

namespace graphics {

class matrix;

class xform  
{
public:
    xform();

    xform( int );		    // diagonal matrix (d,d,d,1)
    xform( real );		    // diagonal matrix (d,d,d,1)
  
    xform( const real[4][4] );	    // from standard real m[4][4]
    xform( const float[4][4] );	    // from standard float m[4][4]
  
    xform( const real* );	    // from array of 16 reals (row0,row1,row2,row3)
    xform( const float* );	    // from array of 16 floats (row0,row1,row2,row3)
  
    xform( const matrix& );	    // from upper left 4x4 of an
                                    // arbitrary matrix.  Any missing
                                    // rows/columns are set to identity. 
    xform(  const vec3& P,	    // as a frame. 
	    const vec3& X,	
	    const vec3& Y,	
	    const vec3& Z); 

    // use implicit operator=(const vec3&), operator==

    real* operator[](int);
    const real* operator[](int) const;

    // xform = scalar results in a diagonal 3x3 with bottom row = 0,0,0,1
    xform& operator=( int );
    xform& operator=( float );
    xform& operator=( real );
    xform& operator=( const matrix& );   // from upper left 4x4 of an
					    // arbitrary matrix.  Any missing
					    // rows/columns are set to identity.

    // All non-commutative operations have "this" as left hand side and
    // argument as right hand side.
    vec2 operator*( const vec2& ) const;
    vec3 operator*( const vec3& ) const;
    vec4 operator*( const vec4& ) const;

    xform operator*( const xform& /*rhs*/ ) const;
    xform operator+( const xform& ) const;
    xform operator-( const xform& /*rhs*/ ) const;


    /*
    Returns:
    true if matrix is the identity transformation

	  1 0 0 0
	  0 1 0 0
	  0 0 1 0
	  0 0 0 1
    Remarks:
    The test for zero is fabs(x) <= zero_tolerance.
    The test for one is fabs(x-1) <= zero_tolerance.
    */
    bool IsIdentity( real zero_tolerance = 0.0) const;

    /*
    Returns:
    true if matrix is the zero transformation

	  0 0 0 0
	  0 0 0 0
	  0 0 0 0
	  0 0 0 *
    */
    bool IsZero() const;

    /*
    Description:
    A similarity transformation can be broken into a sequence
    of dialations, translations, rotations, and reflections.
    Returns:
    +1: This transformation is an orientation preserving similarity.
    -1: This transformation is an orientation preserving similarity.
     0: This transformation is not a similarity.
    */
    int IsSimilarity() const;

    // matrix operations
    void Transpose(); // transposes 4x4 matrix

    int 
    Rank( // returns 0 to 4
	real* = NULL // If not NULL, returns minimum pivot
    ) const;

    real
    Determinant( // returns determinant of 4x4 matrix
	real* = NULL // If not NULL, returns minimum pivot
    ) const;

    bool
    Invert( // If matrix is non-singular, returns true,
	  // otherwise returns false and sets matrix to 
	  // pseudo inverse.
	real* = NULL // If not NULL, returns minimum pivot
    );

    xform
    Inverse(  // If matrix is non-singular, returns inverse,
	    // otherwise returns pseudo inverse.
	real* = NULL // If not NULL, returns minimum pivot
    ) const;

    // Description:
    //   Computes matrix * transpose([x,y,z,w]).
    //
    // Parameters:
    //   x - [in]
    //   y - [in]
    //   z - [in]
    //   z - [in]
    //   ans - [out] = matrix * transpose([x,y,z,w])
    void ActOnLeft(
	 real, // x
	 real, // y
	 real, // z
	 real, // w
	 real[4] // ans
	 ) const;

    // Description:
    //   Computes [x,y,z,w] * matrix.
    //
    // Parameters:
    //   x - [in]
    //   y - [in]
    //   z - [in]
    //   z - [in]
    //   ans - [out] = [x,y,z,w] * matrix
    void ActOnRight(
	 real, // x
	 real, // y
	 real, // z
	 real, // w
	 real[4] // ans
	 ) const;

    ////////////////////////////////////////////////////////////////
    // standard transformations

    // All zeros including the bottom row.
    void Zero();

    // diagonal is (1,1,1,1)
    void Identity();

    // diagonal 3x3 with bottom row = 0,0,0,1
    void Diagonal(real); 

    /*
    Description:
    Create non-uniform scale transformation with the origin as
    a fixed point.
    Parameters:
    fixed_point - [in]
    x_scale_factor - [in]
    y_scale_factor - [in]
    z_scale_factor - [in]
    Remarks:
    The diagonal is (x_scale_factor, y_scale_factor, z_scale_factor, 1)
    */
    void Scale( 
	real x_scale_factor,
	real y_scale_factor,
	real z_scale_factor
    );

    /*
    Description:
    Create non-uniform scale transformation with the origin as
    a fixed point.
    Parameters:
    fixed_point - [in]
    scale_vector - [in]
    Remarks:
    The diagonal is (scale_vector.x, scale_vector.y, scale_vector.z, 1)
    */
    void Scale( 
	const vec3& scale_vector
    );

    /*
    Description:
    Create uniform scale transformation with a specified
    fixed point.
    Parameters:
    fixed_point - [in]
    scale_factor - [in]
    */
    void Scale
    (
	vec3 fixed_point,
	real scale_factor
    );

    /*
    Description:
    Create non-uniform scale transformation with a specified
    fixed point.
    Parameters:
    plane - [in] plane.origin is the fixed point
    x_scale_factor - [in] plane.xaxis scale factor
    y_scale_factor - [in] plane.yaxis scale factor
    z_scale_factor - [in] plane.zaxis scale factor
    */
    void Scale
    (
	const frame& plane,
	real x_scale_factor,
	real y_scale_factor,
	real z_scale_factor
    );

    /*
    Description:
    Create shear transformation.
    Parameters:
    plane - [in] plane.origin is the fixed point
    x1 - [in] plane.xaxis scale factor
    y1 - [in] plane.yaxis scale factor
    z1 - [in] plane.zaxis scale factor
    */
    void Shear
    (
	const frame& plane,
	const vec3& x1,
	const vec3& y1,
	const vec3& z1
    );

    // Right column is (d.x, d.y,d.z, 1).
    void Translation( 
	const vec3& // d
    );

    // Right column is (dx, dy, dz, 1).
    void Translation( 
	real, // dx
	real, // dy
	real  // dz
    );

    // Description:
    //   Get transformation that projects to a plane
    // Parameters:
    //   plane - [in] plane to project to
    // Remarks:
    //   This transformaton maps a 3d point P to the
    //   point plane.ClosestPointTo(Q).
    void PlanarProjection(
	const frame& plane
    );

    // Description: 
    //   The Rotation() function is overloaded and provides several
    //   ways to compute a rotation transformation.  A positive
    //   rotation angle indicates a counter-clockwise (right hand rule)
    //   rotation about the axis of rotation.
    //
    // Parameters:
    //   sin_angle - sin(rotation angle)
    //   cos_angle - cos(rotation angle)
    //   rotation_axis - 3d unit axis of rotation
    //   rotation_center - 3d center of rotation
    //
    // Remarks: 
    //   In the overloads that take frames, the frames should 
    //   be right hand orthonormal frames 
    //   (unit vectors with Z = X x Y).  
    //   The resulting rotation fixes
    //   the origin (0,0,0), maps initial X to 
    //   final X, initial Y to final Y, and initial Z to final Z.
    //  
    //   In the overload that takes frames with center points, 
    //   if the initial and final center are equal, then that 
    //   center point is the fixed point of the rotation.  If 
    //   the initial and final point differ, then the resulting
    //   transform is the composition of a rotation fixing P0
    //   and translation from P0 to P1.  The resulting 
    //   transformation maps P0 to P1, P0+X0 to P1+X1, ...
    //
    //   The rotation transformations that map frames to frames
    //   are not the same as the change of basis transformations
    //   for those frames.  See xform::ChangeBasis().
    //   
    void Rotation(
	real sin_angle,
	real cos_angle,
	vec3 rotation_axis,
	vec3 rotation_center
    );

    // Parameters:
    //   angle - rotation angle in radians
    //   rotation_axis - 3d unit axis of rotation
    //   rotation_center - 3d center of rotation
    void Rotation(
	real angle_radians,
	vec3 rotation_axis,
	vec3 rotation_center
    );

    // Parameters:
    //   X0 - initial frame X
    //   Y0 - initial frame Y
    //   Z0 - initial frame Z
    //   X1 - final frame X
    //   Y1 - final frame Y
    //   Z1 - final frame Z
    //
    void Rotation( 
	const vec3& X0,
	const vec3& Y0,
	const vec3& Z0,
	const vec3& X1,
	const vec3& Y1,
	const vec3& Z1
    );

    // Parameters:
    //   P0 - initial frame center
    //   X0 - initial frame X
    //   Y0 - initial frame Y
    //   Z0 - initial frame Z
    //   P1 - initial frame center
    //   X1 - final frame X
    //   Y1 - final frame Y
    //   Z1 - final frame Z
    void Rotation( 
	const vec3& P0,
	const vec3& X0,
	const vec3& Y0,
	const vec3& Z0,
	const vec3& P1,
	const vec3& X1,
	const vec3& Y1,
	const vec3& Z1
    );

    /*
    Description:
    Create rotation transformation that maps plane0 to plane1.
    Parameters:
    plane0 - [in]
    plane1 - [in]
    */
    void Rotation( 
	const frame& plane0,
	const frame& plane1
    );

    /*
    Description:
    Create mirror transformation matrix.
    Parameters:
    point_on_mirror_plane - [in] point on mirror plane
    normal_to_mirror_plane - [in] normal to mirror plane
    Remarks:
    The mirror transform maps a point Q to
    Q - (2*(Q-P)oN)*N, where
    P = point_on_mirror_plane and N = normal_to_mirror_plane.
    */
    void Mirror(
	vec3 point_on_mirror_plane,
	vec3 normal_to_mirror_plane
    );

    // Description: The ChangeBasis() function is overloaded 
    //   and provides several
    //   ways to compute a change of basis transformation.
    //
    // Parameters:
    //   plane0 - inital plane
    //   plane1 - final plane
    //
    // Returns:
    //   @untitled table
    //   true    success
    //   false   vectors for initial frame are not a basis
    //
    // Remarks: 
    //   If you have points defined with respect to planes, the
    //   version of ChangeBasis() that takes two planes computes
    //   the transformation to change coordinates from one plane to 
    //   another.  The predefined world plane ON_world_plane can
    //   be used as an argument.
    //
    //   If P = plane0.Evaluate( a0,b0,c0 ) and 
    //
    //   (a1,b1,c1) = ChangeBasis(plane0,plane1)*vec3(a0,b0,c0),
    //
    //   then P = plane1.Evaluate( a1, b1, c1 )
    //          
    //   The version of ChangeBasis() that takes six vectors
    //   maps (a0,b0,c0) to (a1,b1,c1) where
    //   a0*X0 + b0*Y0 + c0*Z0 = a1*X1 + b1*Y1 + c1*Z1
    //
    //   The version of ChangeBasis() that takes six vectors
    //   with center points
    //   maps (a0,b0,c0) to (a1,b1,c1) where
    //   P0 + a0*X0 + b0*Y0 + c0*Z0 = P1 + a1*X1 + b1*Y1 + c1*Z1
    //
    //   The change of basis transformation is not the same as
    //   the rotation transformation that rotates one orthonormal
    //   frame to another.  See xform::Rotation().
    bool ChangeBasis( 
	const frame& plane0,
	const frame& plane1
    );

    // Description:
    //   Get a change of basis transformation.
    // Parameters:
    //   X0 - initial basis X (X0,Y0,Z0 can be any 3d basis)
    //   Y0 - initial basis Y
    //   Z0 - initial basis Z
    //   X1 - final basis X (X1,Y1,Z1 can be any 3d basis)
    //   Y1 - final basis Y
    //   Z1 - final basis Z
    // Remarks:
    //   Change of basis transformations and rotation transformations
    //   are often confused.  This is a change of basis transformation.
    //   If Q = a0*X0 + b0*Y0 + c0*Z0 = a1*X1 + b1*Y1 + c1*Z1
    //   then this transform will map the point (a0,b0,c0) to (a1,b1,c1)
    bool ChangeBasis( 
	const vec3& X0,
	const vec3& Y0,
	const vec3& Z0,
	const vec3& X1,
	const vec3& Y1,
	const vec3& Z1
    );

    // Parameters:
    //   P0 - initial center
    //   X0 - initial basis X (X0,Y0,Z0 can be any 3d basis)
    //   Y0 - initial basis Y
    //   Z0 - initial basis Z
    //   P1 - final center
    //   X1 - final basis X (X1,Y1,Z1 can be any 3d basis)
    //   Y1 - final basis Y
    //   Z1 - final basis Z
    // Remarks:
    //   Change of basis transformations and rotation transformations
    //   are often confused.  This is a change of basis transformation.
    //   If Q = P0 + a0*X0 + b0*Y0 + c0*Z0 = P1 + a1*X1 + b1*Y1 + c1*Z1
    //   then this transform will map the point (a0,b0,c0) to (a1,b1,c1)
    bool ChangeBasis( 
	const vec3& P0,
	const vec3& X0,
	const vec3& Y0,
	const vec3& Z0,
	const vec3& P1,
	const vec3& X1,
	const vec3& Y1,
	const vec3& Z1
    );


    real m_xform[4][4]; // [i][j] = row i, column j.  I.e., 
                        //
                        //           [0][0] [0][1] [0][2] [0][3]
                        //           [1][0] [1][1] [1][2] [1][3]
                        //           [2][0] [2][1] [2][2] [2][3]
                        //           [3][0] [3][1] [3][2] [3][3]
};

};  // namespace graphics

#endif
