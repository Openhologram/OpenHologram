#ifndef __projector_h
#define __projector_h

#include "graphics/sys.h"
#include "graphics/quater.h"
#include "graphics/geom.h"
#include "graphics/gl.h"
#include "graphics/Camera.h"

namespace graphics {

struct projector {

private:
 
    frame   local_space;

    /*| last picked point     */
    vec3    last_point;	

public:

	Camera *vp;

    /////////////////////////////////////////////////////////////////////////|
    virtual  vec3 project(const vec2& point);
    //|
    //| compute projected point on the cylinder in model space
    /////////////////////////////////////////////////////////////////////////|

    virtual  void set_local_space(const frame &g);
    virtual  const frame& get_local_space() const;
	virtual  frame& get_local_space();

    const vec3& get_last_point() const
    {
	return last_point;
    }

    vec3& get_last_point()
    {
	return last_point;
    } 

    void set_last_point(const vec2& mousePosition)
    { 
	last_point = project(mousePosition); 
    }

    void set_last_point(const vec3& point)
    {
	last_point = point;
    }

    vec3 get_vector(const vec2& m1, const vec2& m2) ;

    vec3 get_vector(const vec2& m) ;

  protected:

    projector(Camera *pp) 
	: local_space(), vp(pp)
    {
    }

    projector() : local_space(), vp(0)
    {
    }

    virtual ~projector() { }

    /*|----------------------------------------------------------------------*/
    /*| line 'a' is a line in world space, which comes from image point to   */
    /*| inside the screen.                                                   */
    /*|______________________________________________________________________*/
    line get_model_line(const line& world_line) const ;

    line get_world_line(const line& model_line) const;

    plane get_world_plane(const plane& model_plane) const  ;

    plane get_model_plane(const plane& world_plane) const ;
};


struct line_projector : projector {

    /////////////////////////////////////////////////////////////////////////|
    line 	l;				
    //|
    //| This is a line with respect to space defined in projector
    /////////////////////////////////////////////////////////////////////////|

    line_projector (Camera *pp)  
	: projector(pp)
    {
		line ll(vec3(0), vec3(0.0,1.0,0.0));
		set_line(ll);
    }

    line_projector (const line& _ll, Camera *pp)  
	: projector(pp)
    {
		set_line(_ll);
    }

    line_projector () : projector()
    {
		fatal("line_projector :: no projection call back\n");
    }

    void set_line(const line& a);

    line_projector& operator = (const line_projector& a) ;

    virtual  vec3 project(const vec2& point)    ;

    line  get_line() const { return l; }


};


struct plane_projector: projector {

    /////////////////////////////////////////////////////////////////////////|
    plane 	_plane;
    //|
    //| This is a plane with respect to space defined in projector
    /////////////////////////////////////////////////////////////////////////|

    plane_projector (Camera *pp)
      : projector(pp)
    {
		plane pl(vec3(0,0,1), 0);
		set_plane(pl);
    }
    
    plane_projector(const plane& p,
		    Camera *pp)
	: projector(pp)
    {
		set_plane(p);
    }

    void  set_plane(const plane& p) ;

    plane_projector& operator = (const plane_projector& a)  ;

    /////////////////////////////////////////////////////////////////////////|
    vec3 project(const vec2& point);
    //|
    //| compute projected point on the cylinder in model space
    /////////////////////////////////////////////////////////////////////////|


    plane get_plane() const
    { 
	return _plane; 
    }

};

struct cylinder_projector : projector {

  protected:

    /////////////////////////////////////////////////////////////////////////|
    cylinder	my_cylinder;		//| Cylinder for this projector.
    //|
    //| This is a cylinder with respect to local 'space' defined in projector
    /////////////////////////////////////////////////////////////////////////|
    int		need_setup;

    //////|
    //| Constructors
    //| The default cylinder to is centered about the Y axis and
    //| has a radius of 1.0.
    cylinder_projector(Camera *pp) 
	: projector(pp)
    {
		set_cylinder(line(vec3(0), vec3(0,1,0)), 1.0);
    }

    cylinder_projector(const cylinder& cyl, Camera *pp)
	: projector(pp)
    {
		set_cylinder(cyl);
    }

  public:

    virtual vec3 project(const vec2& point)
    {
		return vec3(0);
    }

    vec3 project_and_get_rot(const vec2& point, quater &rot)  ;

    ///| abstract
    virtual  quater get_rot(const vec3& point1,  const vec3& point2)   ;

    void    set_cylinder(const cylinder& c)
    {
		my_cylinder = c;
		need_setup = 1;
    }

    void    set_cylinder(const line& a, real rad)
    {
		my_cylinder = cylinder(a, rad);
		need_setup = 1;
    }

    //////|
    //| note that return value is reference type.
    cylinder&	get_cylinder() { return my_cylinder; }

    
    bool is_point_in_front(const vec3& point) const  ;

    virtual void set_local_space(const frame &a);
};


struct sphere_projector : projector {

  protected:

    /////////////////////////////////////////////////////////////////////////|
    sphere  my_sphere;
    //|
    //| This is a sphere with respect to local 'space' defined in projector
    /////////////////////////////////////////////////////////////////////////|

    int	    need_setup;

    sphere_projector(Camera *pp)
	: projector(pp)
    {
		vec3 org(0); 
		real r = 0.0;
		sphere tmp(org, r);

	set_sphere(tmp);
    }
	
    sphere_projector(const sphere& sph, Camera *pp)
	: projector(pp)
    {
		set_sphere(sph);
    }
    
  public :

    virtual vec3 project(const vec2& point)
    {
		vec3 ret(0);
		return ret;
    }

    vec3 project_and_get_rot(const vec2& point, quater &rot)  ;

	///| abstract
    virtual  quater get_rot(const vec3& point1,  const vec3& point2)    ;

    void    set_sphere(const sphere& sph)
    {
		need_setup  = 1;
		my_sphere   = sph;
    }

    sphere  get_sphere() const
    { 
	return my_sphere; 
    }


    bool is_point_in_front (const vec3& point) const    ;

    virtual void set_local_space(const frame &a) ;
    
};

struct cylinder_section_projector : cylinder_projector {

  protected:

    //| Information about the slice tolerance.
    real    tolerance;		//| the edge tolerance
    real    tol_dist;		//| dist from planePoint to tolerance slice
  
    //| Information about the plane used for intersection testing.
    vec3    plane_point;	//| point on plane
    vec3    plane_dir;		//| point on plane
    line    plane_line;		//| line parallel to axis, but in plane
    real    plane_dist;		//| distance from sphere center
    plane   tol_plane;		//| the plane itself

  public:
    
    cylinder_section_projector(Camera *pp)
	: cylinder_projector(pp)
    {
		set_tolerance(0.9);
    }

    cylinder_section_projector(const cylinder &cyl, 
	Camera *pp)
	    : cylinder_projector(cyl, pp)
    {
		set_tolerance(0.9);
    }


    virtual  cylinder_section_projector& operator = (const cylinder_section_projector& a)    ;
    
    //////////////////////////////////////////////////////////////////////////|
    vec3  project(const vec2& point);
    //|
    //| return vector is always local to the space defined at projector
    //////////////////////////////////////////////////////////////////////////|

    ///////////////////////////////////////////////////////////////////////////|
    virtual  quater get_rot(const vec3& p1, const vec3& p2);
    //|
    //| Return quaternion is always applied and computed with respect to the
    //| local space defined by space declared at the super class, projector
    ///////////////////////////////////////////////////////////////////////////|


    void    set_tolerance(real t)
    {
		tolerance = t;
		need_setup = 1;
    }

    real    get_tolerance() const { return tolerance; }

    bool is_within_tolerance(const vec3& point) ;
    
  protected:


    //////|
    //| This part was where I could not implement by myself.
    virtual  void setup_tolerance(const vec2& point) ;

};

struct cylinder_plane_projector : cylinder_section_projector {
  
  public:
  
    cylinder_plane_projector(Camera *pp) 
	: cylinder_section_projector(pp)
    {
    }

    cylinder_plane_projector
	(const cylinder &cyl, Camera *pp)
	    : cylinder_section_projector(cyl, pp)
    {
    }

    //////////////////////////////////////////////////////////////////////////|
    virtual  vec3 project(const vec2& point);
    //|
    //| compute projected point on the cylinder in model space
    /////////////////////////////////////////////////////////////////////////|

    ///////////////////////////////////////////////////////////////////////////|
    virtual  quater get_rot(const vec3& p1, const vec3& p2);
    //|
    //| Return quaternion is always applied and computed with respect to the
    //| local space defined by space declared at the super class, projector
    ///////////////////////////////////////////////////////////////////////////|


  protected:
    
    quater get_rot(const vec3& p1, const vec3& p2, bool tol1, bool tol2)    ;
};

struct sphere_section_projector : sphere_projector {

  public:

    sphere_section_projector(Camera *pp)
	: sphere_projector(pp)
    {
		set_tolerance(0.9);
		set_radial_factor(0.0);
    }

    sphere_section_projector
	(const sphere& sph, Camera *pp)
	    : sphere_projector(sph, pp)
    {
		set_tolerance(0.9);
		set_radial_factor(0.0);
    }

    sphere_section_projector& operator = 
	(const sphere_section_projector& a)    ;

    /////////////////////////////////////////////////////////////////////////|
     vec3 project(const vec2& point);
    //|
    //| compute projected point on the cylinder in model space
    /////////////////////////////////////////////////////////////////////////|

    virtual  bool project(const vec2& point, vec3& ret);
    //|
    //| compute projected point on the cylinder in model space
    /////////////////////////////////////////////////////////////////////////|

    ///////////////////////////////////////////////////////////////////////////|
    virtual  quater get_rot(const vec3& p1, const vec3& p2);
    //|
    //| Return quaternion is always applied and computed with respect to the
    //| local space defined by space declared at the super class, projector
    ///////////////////////////////////////////////////////////////////////////|

    
    void  set_tolerance(real t) ;

    
    real get_tolerance() const ;

    void    set_radial_factor(real rad = 0.0) { radial_factor = rad; }
    real    get_radial_factor() const { return radial_factor; }

    
    bool is_within_tolerance(const vec3& point);
    
  protected:

    //////|
    //| This part was where I could not implement by myself.
    virtual  void   setup_tolerance(const vec2& point);

    //////|
    //| Information about the slice tolerance.
    real    tolerance;		//| the edge tolerance
    real    tol_dist;		//| dist from planePoint to tolerance slice

    real    radial_factor;
  
    //////|
    //| Information about the plane used for intersection testing.
    vec3    plane_point;	//| point on plane
    vec3    plane_dir;		//| normal direction
    real    plane_dist;		//| distance from sphere center
    plane   tol_plane;		//| the plane itself
};



struct sphere_plane_projector : sphere_section_projector {

  protected:

    /////////////////////////////////////////////////////////////////////////|
    plane   my_plane;
    //|
    //| This is a plane with respect to space defined in projector
    /////////////////////////////////////////////////////////////////////////|

  public :

    sphere_plane_projector(Camera *pp)
	: sphere_section_projector(pp)
    {
		plane pl(vec3(0,0,1), 0);
		set_plane(pl);
    }
	
    sphere_plane_projector(const sphere& sph, const plane& pl, Camera *pp)
	: sphere_section_projector(sph, pp)
    {
		set_plane(pl);
    }


    virtual vec3 project(const vec2& point)
    {
		vec3 result = sphere_section_projector::project(point);
		vec3 plane_dir = my_plane.get_normal();
		line prj_line(result, result + plane_dir);

		vec3 isect;
		int is_isct = my_plane.intersect(prj_line, isect);

		if (is_isct) {
			set_last_point(isect);
			return get_last_point();
		}
		else {	//| should be fatal error
			set_last_point(vec3(0));
			return get_last_point();
		}
    }

    virtual bool project(const vec2& point, vec3& hit)
    {
		vec3 result;

		line _ray;
		vp->getViewRay(point, _ray);
		line working_line = get_model_line(_ray);

		vec3 sphereIntersection, dontCare;

		int hitSphere =
    			my_sphere.intersect(working_line, sphereIntersection, dontCare);

		if (hitSphere) {
			hit = sphereIntersection;
			return true;
		} else {
			return false;
		}

    }

    vec3 project_and_get_rot(const vec2& point, quater &rot);

    virtual  quater get_rot(const vec3& point1, const vec3& point2) ;

    void    set_sphere(const sphere& sph)
    {
		sphere_section_projector::set_sphere(sph);
    }

    sphere  get_sphere() const
    { 
		return my_sphere; 
    }

    void    set_plane(const plane& pl)
    {
		my_plane = pl;
    }

    plane   get_plane() const
    {
		return my_plane;
    }
};

}; 	// namespace
#endif
