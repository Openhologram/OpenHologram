#include "graphics/projector.h"
/*|~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*/
/*| Open Source Project, Open Inventer were heavily referenced.              */
/*| Thus this work can not be used for commercial purpose.                   */
/*| When you want to use this for that, you need your own idea on            */
/*| tolerancing in *_section_projectors.                                     */
/*|                                                                          */
/*| In fact I also tried to make my own projectors for sphere and            */
/*| cylinder, this tolerancing was major difficulty.                         */
/*|__________________________________________________________________________*/


namespace graphics {

vec3 projector::project(const vec2& point)
//|
//| compute projected point on the cylinder in model space
/////////////////////////////////////////////////////////////////////////|
{
    return vec3(0);
}

void projector::set_local_space(const frame &g)
{	   
    local_space = g;
}

const frame& projector::get_local_space() const
{
    return local_space;
}

 frame& projector::get_local_space()
{
    return local_space;
}

vec3 projector::get_vector(const vec2& m1, const vec2& m2)
{
    vec3 p1 = project(m1);
    vec3 p2 = project(m2);

    last_point = p2;
    return p2 - p1;
}

vec3 projector::get_vector(const vec2& m)
{
    vec3 p = project(m);
    vec3 result = p - last_point;
    last_point = p;
    return result;
}

line projector::get_model_line(const line& world_line) const
{
    vec3 pos1 = world_line.get_position();
    vec3 dir = world_line.get_direction();

    pos1 = local_space.to_model(pos1);
    dir = local_space.to_model_normal(dir);

    line ret;
	ret.set_position(pos1);
	ret.set_direction(dir);
    return ret;
}

line projector::get_world_line(const line& model_line) const
{
    vec3 pos1 = model_line.get_position();
    vec3 dir  = model_line.get_direction();

    pos1 = local_space.to_world(pos1);
    dir = local_space.to_world_normal(dir);

    line ret;
	ret.set_position(pos1);
	ret.set_direction(dir);
    return ret;
}

plane projector::get_world_plane(const plane& model_plane) const
{
    vec3 normal = model_plane.get_normal();
    vec3 pos1   = normal * model_plane.get_dist();

    pos1 = local_space.to_world(pos1);
    normal = local_space.to_world_normal(normal);

    plane ret;

    ret.reset(pos1, normal);

    return ret;
}

plane projector::get_model_plane(const plane& world_plane) const
{
    vec3 normal = world_plane.get_normal();
    vec3 pos1   = normal * world_plane.get_dist();

    pos1 = local_space.to_model(pos1);
    normal = local_space.to_model_normal(normal);

    plane ret;

    ret.reset(pos1, normal);

    return ret;
}




void line_projector::set_line(const line& a)
{
    l = a;
}

line_projector& line_projector::operator = (const line_projector& a)
{
    l = a.l;
    set_last_point(a.get_last_point());
    set_local_space(a.get_local_space());

    return *this;
}

vec3 line_projector::project(const vec2& point)
{
    /*|________________________________________*/
    /*| convert two line points to world space */
    /*|________________________________________*/
    line world_line = get_world_line(l);

    vec3 a = world_line.get_position();
    vec3 b = world_line.get_direction() + a;

    vec2 na, nb;

    na = vp->projectToImagePlane(a); 
    nb = vp->projectToImagePlane(b);

    vec3 sna(na[0], na[1], 0);
    vec3 snb(nb[0], nb[1], 0);

    vec3 spoint(point[0], point[1], 0);

    vec3 s_inter;
    if (norm(sna-snb) > epsilon) {
	line ll(sna, snb);
	s_inter= ll.get_closest_point(spoint);
    } else {
	s_inter = sna;
    }

    vec2 result(s_inter[0], s_inter[1]);

    line _ray;
    vp->getViewRay(result, _ray);

    _ray = get_model_line(_ray);

    vec3 res1, res2;
    l.get_closest_points(_ray, res1, res2);

    set_last_point(res1);

    return res1;
}




void  plane_projector::set_plane(const plane& p)
{
    _plane = p;
}

plane_projector& plane_projector::operator = (const plane_projector& a)
{
    _plane = a._plane;
    set_last_point(a.get_last_point());
    set_local_space(a.get_local_space());

    return *this;
}

vec3 plane_projector::project(const vec2& point)
//|
//| compute projected point on the cylinder in model space
/////////////////////////////////////////////////////////////////////////|
{
    vec3 result;

    plane pp = get_world_plane(_plane);

    line ll;
    vp->getViewRay(point, ll);

    if (apx_equal(inner(ll.get_direction(), pp.get_normal()), 0.0))
	result = _plane.get_dist() * _plane.get_normal();
    else {
	int res = pp.intersect(ll, result);

	if (res)  result = get_local_space().to_model(result);
	else      result = _plane.get_dist() * _plane.get_normal();
    }

    set_last_point(result);

    return result;
}



vec3 cylinder_projector::project_and_get_rot(const vec2& point, quater &rot)
{
    vec3	old_point = get_last_point();
    vec3	new_point = project(point);
    rot	= get_rot(old_point, new_point);
    return	new_point;
}

quater cylinder_projector::get_rot(const vec3& point1,  const vec3& point2)
{
    return quater(0);
}

bool cylinder_projector::is_point_in_front(const vec3& point) const
{
    vec3	closest_pnt_on_axis, axis_pnt_to_input;

    closest_pnt_on_axis = my_cylinder.get_axis().get_closest_point( point );
    axis_pnt_to_input = point - closest_pnt_on_axis;

    vec2	image_p = vp->projectToImagePlane(point);
    vec3	image_p3(image_p[0], image_p[1], 0);
    vec3	input_to_image_pnt = image_p3 - point; 
    
    if (inner(input_to_image_pnt, axis_pnt_to_input) < 0.0)
	return false;

    return true;
}

void cylinder_projector::set_local_space(const frame &a)
{
    projector::set_local_space(a);
    need_setup = 1;
}


vec3 sphere_projector::project_and_get_rot(const vec2& point, quater &rot)
{
    vec3	old_point = get_last_point();
    vec3	new_point = project(point);
    rot	= get_rot(old_point, new_point);
    return	new_point;
}

quater sphere_projector::get_rot(const vec3& point1,  const vec3& point2)
{
    return quater(0);
}

bool sphere_projector::is_point_in_front (const vec3& point) const
{
    vec2 image_p = vp->projectToImagePlane(point);
    vec3 image_p3(image_p[0], image_p[1], 0);

    vec3 cntrToProj = image_p3 - my_sphere.get_center();
    vec3 cntrToInput = point - my_sphere.get_center();

    if (inner(cntrToProj, cntrToInput) < 0.0)
	return false;

    return true;
}

void sphere_projector::set_local_space(const frame &a)
{
    projector::set_local_space(a);
    need_setup = 1;
}



cylinder_section_projector& cylinder_section_projector::operator = (const cylinder_section_projector& a)
{
    my_cylinder = a.my_cylinder;
    need_setup = a.need_setup;
    set_local_space(a.get_local_space());
    set_last_point(a.get_last_point());

    tolerance = a.tolerance;
    tol_dist = a.tol_dist;

    plane_point = a.plane_point;
    plane_dist = a.plane_dist;
    plane_line = a.plane_line;
    tol_plane = a.tol_plane;

    return *this;
}

vec3  cylinder_section_projector::project(const vec2& point)
//|
//| return vector is always local to the space defined at projector
//////////////////////////////////////////////////////////////////////////|
{
    vec3 result;

    line _ray;
    vp->getViewRay(point, _ray);
    line working_line = get_model_line(_ray);

    if (need_setup)
		setup_tolerance(point);

    vec3 plane_intersection;
    vec3 cylinderIntersection, dontCare;
    int hitCylinder;

    hitCylinder = 
	my_cylinder.intersect(working_line, cylinderIntersection, dontCare);

    if (hitCylinder) {
		line projectLine(cylinderIntersection, cylinderIntersection + plane_dir);
		tol_plane.intersect(projectLine, plane_intersection);
    } else if (!tol_plane.intersect(working_line, plane_intersection))
	; //| do nothing
    
    vec3 vecToPoint 
	= plane_intersection - plane_line.get_closest_point(plane_intersection);

    real dist = norm(vecToPoint);

    if (dist < tol_dist) {
		result = cylinderIntersection;
    } else {
		vec3 tolVec = vecToPoint;
		vec3 axisPoint = plane_intersection - tolVec;
		tolVec = unit(tolVec);
		tolVec *= tol_dist;
		result = axisPoint + tolVec;
    }

    set_last_point(result);

    return result;
}

quater cylinder_section_projector::get_rot(const vec3& p1, const vec3& p2)
//|
//| Return quaternion is always applied and computed with respect to the
//| local space defined by space declared at the super class, projector
///////////////////////////////////////////////////////////////////////////|

{
    vec3 v1 = p1 - my_cylinder.get_axis().get_closest_point(p1);
    vec3 v2 = p2 - my_cylinder.get_axis().get_closest_point(p2);
	
    real cosAngle = inner(v1, v2)/(norm(v1)*norm(v2));
	
    if ((cosAngle > 1.0) || (cosAngle < -1.0))
		return _zero_quater();
	    
    real ang = acosf(cosAngle);

    vec3 rotAxis = cross(v1,v2);
	
    return orient(ang, rotAxis);
}

bool cylinder_section_projector::is_within_tolerance(const vec3& point)
{
    if (need_setup) {
		vec2 img_pnt = vp->projectToImagePlane(point);
		setup_tolerance(img_pnt);
    }

    vec3	plane_intersection;
    line	project_line(point, point + plane_dir);

    tol_plane.intersect(project_line, plane_intersection);

    vec3	vecToPoint = plane_intersection - 
			plane_line.get_closest_point(plane_intersection);

    real dist = norm(vecToPoint);

    if (dist < tol_dist)
	return true;
    return false;
}

void cylinder_section_projector::setup_tolerance(const vec2& point)
{
    vec3 perpDir, eyeDir, local_prj_pnt;

    line tmp;
    vp->getViewRay(point, tmp);

    local_prj_pnt = get_local_space().to_model(tmp.get_position());

    eyeDir = local_prj_pnt - my_cylinder.get_axis().get_position();
    perpDir = cross(my_cylinder.get_axis().get_direction(), eyeDir);
    
    plane_dir = cross(perpDir, my_cylinder.get_axis().get_direction());
    plane_dir = unit(plane_dir);

    //| distance from planePoint to edge of tolerance
    tol_dist = my_cylinder.get_radius() * tolerance;

    //| find disntance from the center of the cylinder to the tolerance
    //| plane
    plane_dist = sqrtf((my_cylinder.get_radius()*my_cylinder.get_radius()) - 
			(tol_dist * tol_dist));

    //| plane given direction and distance to origin
    vec3 planePoint = plane_dist*plane_dir + my_cylinder.get_axis().get_position();
    tol_plane = plane(plane_dir, planePoint);
    plane_line.set_value(planePoint, 
	planePoint + my_cylinder.get_axis().get_direction());

    need_setup = 0;
}



vec3 cylinder_plane_projector::project(const vec2& point)
//|
//| compute projected point on the cylinder in model space
/////////////////////////////////////////////////////////////////////////|
{
    vec3 result;

    line _ray;
    vp->getViewRay(point, _ray);
    line working_line = get_model_line(_ray);

    if (need_setup)
		setup_tolerance(point);

    vec3 plane_intersection;
    tol_plane.intersect(working_line, plane_intersection);

    vec3 cylIntersection, dontCare;
    int hitCyl;
    hitCyl = my_cylinder.intersect(working_line, cylIntersection, dontCare);
    
    if (!hitCyl) {
		result = plane_intersection;
    } else {
		line projectLine(cylIntersection, cylIntersection + plane_dir);
		vec3 projectIntersection;
		tol_plane.intersect(projectLine, projectIntersection);

		vec3 vecToPoint = projectIntersection - 
			plane_line.get_closest_point(projectIntersection);
		real dist = norm(vecToPoint);
		
		if (dist < tol_dist)
			result = cylIntersection;
		else
			result = plane_intersection;
    }

    set_last_point(result);
    return result;
}

quater cylinder_plane_projector::get_rot(const vec3& p1, const vec3& p2)
//|
//| Return quaternion is always applied and computed with respect to the
//| local space defined by space declared at the super class, projector
///////////////////////////////////////////////////////////////////////////|
{
    int tol1 = is_within_tolerance(p1);
    int tol2 = is_within_tolerance(p2);

    return get_rot(p1, p2, tol1, tol2);
}

quater cylinder_plane_projector::get_rot(const vec3& p1, const vec3& p2, bool tol1, bool tol2)
{
    if (tol1 && tol2) {

		vec3 v1 = p1 - my_cylinder.get_axis().get_closest_point(p1);
		vec3 v2 = p2 - my_cylinder.get_axis().get_closest_point(p2);
		
		real cosAngle = inner(v1,v2)/(norm(v1)*norm(v2));
		
		//| prevent numerical instability problems
		if ((cosAngle > 1.0) || (cosAngle < -1.0))
			return _zero_quater();
		    
		real angle = acosf(cosAngle);

		vec3 rotAxis = cross(v1, v2);
		
		return orient(angle, rotAxis);

    } else if (!tol1 && !tol2) {
		vec3 v1 = p1 - plane_line.get_closest_point(p1);
		vec3 v2 = p2 - plane_line.get_closest_point(p2);
		if ( inner(v1, v2) < 0.0 ) {
			vec3 linePtNearestP1 = plane_line.get_closest_point(p1);
			vec3 linePtNearestP2 = plane_line.get_closest_point(p2);

			vec3 dirToP1 = p1 - linePtNearestP1;
			vec3 dirToP2 = p2 - linePtNearestP2;
			dirToP1 = unit(dirToP1);
			dirToP2 = unit(dirToP2);

			vec3 ptOnCylP1Side = linePtNearestP1 + (dirToP1 * tol_dist);
			vec3 ptOnCylP2Side = linePtNearestP2 + (dirToP2 * tol_dist);

			return  get_rot(ptOnCylP2Side, p2, 0, 0) & 
				get_rot(ptOnCylP1Side, ptOnCylP2Side, 1, 1) &
				get_rot(p1, ptOnCylP1Side, 0, 0);
		} else {

			vec3 diff = v2 - v1;
		    
			real d = norm(diff);

			real angle = (my_cylinder.get_radius()==0.0) 
				  ? 0 : (d / my_cylinder.get_radius());

			vec3 rotAxis = cross(plane_dir, v1);
		    
			if (norm(v2) > norm(v1))
			return orient(angle, rotAxis);
			else
			return orient(-angle, rotAxis);
		}
    } else {
		vec3 offCylinderPt = (tol1) ? p2 : p1;

		vec3 linePtNearest = plane_line.get_closest_point(offCylinderPt);

		vec3 dirToOffCylinderPt = offCylinderPt - linePtNearest;
		dirToOffCylinderPt = unit(dirToOffCylinderPt);

		vec3 ptOnCylinder = linePtNearest + (dirToOffCylinderPt * tol_dist);

		if (tol1) {

			//| p1 is on cyl, p2 off - went off cylinder

			return  get_rot(ptOnCylinder, p2, 0, 0) &
				get_rot(p1, ptOnCylinder, 1, 1);
		} else {

			//| p1 is off cyl, p2 on - came on to cylinder
		    
			return  get_rot(ptOnCylinder, p2, 1, 1) &
				get_rot(p1, ptOnCylinder, 0, 0);
		}
    }
}



sphere_section_projector& sphere_section_projector::operator = 
    (const sphere_section_projector& a)
{
    my_sphere = a.my_sphere;
    need_setup = a.need_setup;
    set_local_space(a.get_local_space());
    set_last_point(a.get_last_point());

    tolerance = a.tolerance;
    tol_dist = a.tol_dist;
    plane_point = a.plane_point;
    radial_factor = a.radial_factor;
    plane_dist = a.plane_dist;
    plane_dir = a.plane_dir;
    tol_plane = a.tol_plane;

    return *this;
}

vec3 sphere_section_projector::project(const vec2& point)
//|
//| compute projected point on the cylinder in model space
/////////////////////////////////////////////////////////////////////////|
{
    vec3 result;

    line _ray;
    vp->getViewRay(point, _ray);
    line working_line = get_model_line(_ray);

    if (need_setup)
		setup_tolerance(point);

    vec3 plane_intersection;
    vec3 sphereIntersection, dontCare;

    int hitSphere;

    hitSphere = 
	    my_sphere.intersect(working_line, sphereIntersection, dontCare);

    if (hitSphere) {
		line projectLine(sphereIntersection, sphereIntersection + plane_dir);
		tol_plane.intersect(projectLine, plane_intersection);
    } else if (! tol_plane.intersect(working_line, plane_intersection))
	    ;
    
    real dist = norm(plane_intersection - plane_point);

    if (dist < tol_dist) {
		result = sphereIntersection;
    } else {
		result = plane_intersection;
    }

    set_last_point(result);

    return result;
}

bool sphere_section_projector::project(const vec2& point, vec3& ret)
//|
//| compute projected point on the cylinder in model space
/////////////////////////////////////////////////////////////////////////|
{
    vec3 result;

    line _ray;
    vp->getViewRay(point, _ray);
    line working_line = get_model_line(_ray);

    if (need_setup)
		setup_tolerance(point);

    vec3 plane_intersection;
    vec3 sphereIntersection, dontCare;

    int hitSphere;

    hitSphere = 
	    my_sphere.intersect(working_line, sphereIntersection, dontCare);

    if (hitSphere) {
		ret = sphereIntersection;
	return true;
    } else {
		return false;
    }
}

quater sphere_section_projector::get_rot(const vec3& p1, const vec3& p2)
//|
//| Return quaternion is always applied and computed with respect to the
//| local space defined by space declared at the super class, projector
///////////////////////////////////////////////////////////////////////////|
{
    bool tol1 = is_within_tolerance(p1);
    bool tol2 = is_within_tolerance(p2);


    quater ret;

    if (tol1 && tol2) {

		vec3 before = unit(p1 - my_sphere.get_center());
		vec3 after  = unit(p2 - my_sphere.get_center());

		ret = u2v_quater(before, after);
		return ret;
    }  else if (!tol1 && !tol2) {
	
		vec3 before = unit(p1 - plane_point);
		vec3 after  = unit(p2 - plane_point);

		quater ret = u2v_quater(before, after);

		return ret;

    } else {

		line plane_line;
		vec3 intersection;

		if (tol1) {
			plane_line.set_value(plane_point, p2);
		} else {
			plane_line.set_value(plane_point, p1);
		}

		my_sphere.intersect(plane_line, intersection);

		if (tol1) {
			vec3 before = unit(p1 - my_sphere.get_center());
			vec3 after  = unit(intersection - my_sphere.get_center());
			ret = u2v_quater(before, after);
			return ret;
		} else {
			vec3 before = unit(intersection - my_sphere.get_center());
			vec3 after  = unit(p2 - my_sphere.get_center());
			ret = u2v_quater(before, after);
			return ret;
		}
    }
}

void  sphere_section_projector::set_tolerance(real t)
{
    tolerance = t;
    need_setup = 1;
}

real sphere_section_projector::get_tolerance() const
{ 
    return tolerance; 
}

bool sphere_section_projector::is_within_tolerance(const vec3& point)
{
   if (need_setup) {
		vec2 img_pnt = vp->projectToImagePlane(point);
		setup_tolerance(img_pnt);
    }

    vec3 plane_intersection;
    line ll(point, point + plane_dir);

    if (!tol_plane.intersect(ll, plane_intersection)) {
	    return false;
    }

    real dist = norm(plane_intersection - plane_point);

    return (dist < (tol_dist - .001));
}

void   sphere_section_projector::setup_tolerance(const vec2& point)
{
    vec3 local_prj_pnt;

    line tmp;
    vp->getViewRay(point, tmp);

    local_prj_pnt = get_local_space().to_model(tmp.get_position());

    plane_dir = local_prj_pnt - my_sphere.get_center();
    plane_dir = unit(plane_dir);
    tol_dist  = my_sphere.get_radius() * tolerance;

    plane_dist = sqrtf((my_sphere.get_radius() * my_sphere.get_radius()) -
		       (tol_dist * tol_dist));

    //////|
    //| plane given direction and point to pass through
    plane_point = my_sphere.get_center() + (plane_dist*plane_dir);

    tol_plane = plane(plane_dir, plane_point);

    need_setup = 0;

}


vec3 sphere_plane_projector::project_and_get_rot(const vec2& point, quater &rot)
{
    vec3	old_point = get_last_point();
    vec3	new_point = project(point);
	    
    rot	= get_rot(old_point, new_point);

    set_last_point(new_point);

    return	new_point;
}

quater sphere_plane_projector::get_rot(const vec3& point1, const vec3& point2)
{
    vec3 center = my_sphere.get_center();

    vec3 u = unit(point1 - center);
    vec3 v = unit(point2 - center);

    quater ret = u2v_quater(u, v);

    return ret;
}

}; 	// namespace
