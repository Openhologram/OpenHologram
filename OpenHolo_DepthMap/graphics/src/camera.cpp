//
// Camera.cpp: implementation of the Camera class.
//
//////////////////////////////////////////////////////////////////////

#include "graphics/Camera.h"

#ifdef __APPLE__
#include <Opengl/glext.h>
#else
#include <gl/glext.h>
#endif
#include "graphics/geom.h"
#include "graphics/frame.h"
#include "graphics/_limits.h"
#include "GL/glext.h"
#include "graphics/projector.h"

//////////////////////////////////////////////////////////////////////
// Construction/Destruction
//////////////////////////////////////////////////////////////////////

namespace graphics {

Camera::Camera( int w, int h) 
    : 
    grid_size_(250),
    const_far_dist_(100000.0),
    angle(45),
    scale_(1),
    camera_pose_(),
    need_update_(true),
	selection_buffer(),
	viewport_(0,0, w, h),
	camera_mode_(kPerspective),
	center_of_look_(vec3(0.01,0,0)),
	model_associated_(0),
	center_of_model_(0),
	near_(0.1),
	far_(10000.0),
	image_depth_(30.0),
	camera_modeling_tool_(0),
	selection_buffer_enable_(true)
{

	model_associated_ = default_model_associated_ = new VirtualModel();
	rotation_radius_ = grid_size_;
	real const_focal_length_ = grid_size_ / tan(radian(angle/2.0));
	aspect_ratio = viewport_.GetAspectRatio();

	// setup initial camera position
	vec3 dir = vec3(0, const_focal_length_, -const_focal_length_*3.0);

	frame f(unit(dir), vec3(0,0,1));
	f.set_eye_position(-dir);

	SetCameraPose(f);
	

	need_update(true);
}


void 
Camera::reset()
{
	grid_size_ = 250;
	const_far_dist_ = 100000.0; 
	angle = 45; 
	scale_ = 1;
	camera_pose_ = frame();
	need_update_ = true;

	camera_mode_ = kPerspective;
	center_of_look_ = vec3(0.01, 0, 0);
	center_of_model_ = vec3(0);
	near_ = 0.1;
	far_ = 10000.0;
	rotation_radius_ = grid_size_;
	real const_focal_length_ = grid_size_ / tan(radian(angle / 2.0));
	aspect_ratio = viewport_.GetAspectRatio();

	// setup initial camera position
	vec3 dir = vec3(0, const_focal_length_, -const_focal_length_*3.0);

	frame f(unit(dir), vec3(0, 0, 1));
	f.set_eye_position(-dir * 1000);

	SetCameraPose(f);


	need_update(true);
}
Camera::Camera(const Camera& val) 
    :
    grid_size_(val.grid_size_),
    const_far_dist_(val.const_far_dist_),
    angle(val.angle),
    scale_(val.scale_),
    camera_pose_(val.camera_pose_),
	viewport_(val.viewport_),
	selection_buffer(val.selection_buffer),
	camera_mode_(val.camera_mode_),
	center_of_look_(val.center_of_look_),
	model_associated_(val.model_associated_),
	center_of_model_(val.center_of_model_),
	camera_modeling_tool_(val.camera_modeling_tool_),
	selection_buffer_enable_(val.selection_buffer_enable_)
{
}

Camera::~Camera()
{
	delete default_model_associated_;
}

void Camera::set_width(int a) { 
	viewport_.Resize(a, viewport_.GetHeight());
	aspect_ratio = viewport_.GetAspectRatio();
}

void Camera::set_height(int a) 
{ 
	viewport_.Resize(viewport_.GetWidth(), a);
	aspect_ratio = viewport_.GetAspectRatio();
}

void Camera::Transform(const frame& f)
{
	camera_pose_.Transform(f);

	if (camera_modeling_tool_) camera_modeling_tool_->Transform(f);
}

void Camera::set_camera_mode(CameraMode mode)
{
	if (camera_mode_ == kPerspective && mode == kParallel) {
		box3 b = model_associated_->GetBoundingBox();
		if (!b.is_empty()) {
			vec3 size;
			b.get_size(size[0], size[1], size[2]);
			real rsize = norm(size);
			grid_size_ = rsize * 10;
			scale_ = 1.0;
		}
	}

	camera_mode_ = mode;
}

Camera::CameraMode 
Camera::camera_mode() const
{
	return camera_mode_;
}

int Camera::GetWidth() const
{
	return viewport_.GetWidth();
}

int Camera::GetHeight() const
{
	return viewport_.GetHeight();
}

// Set it to need update selection buffer formed by this camera
void Camera::need_update(bool rhs) 
{ 
	//LOG("set need_update selection buffer\n");
	need_update_ = rhs; 
}

// Need to update the selection buffer?
bool Camera::need_update() const 
{ 
	return need_update_; 
}

void Camera::update_center_of_look()
{
	center_of_model_ = model_associated_->ComputeCenterOfModel();
	real dist = norm(camera_pose_.get_origin() - center_of_model_);

	if (apx_equal(dist, 0.0)) {
		center_of_look_ = camera_pose_.get_origin() + camera_pose_.basis[2];
	}
	else {
		center_of_look_ = camera_pose_.get_origin() + 
		(norm(camera_pose_.get_origin() - center_of_model_) * camera_pose_.basis[2]);
	}

	
	if (camera_mode_ != kImage) {
		near_ = norm(camera_pose_.get_origin()-center_of_look_) * 0.001;

	}
	
	real cam_dist = norm(camera_pose_.get_origin());
	if (cam_dist > 10.0) {
		far_ = const_far_dist_ + cam_dist * 10.0;
		return;
	}

	far_ = const_far_dist_;
}

color_buffer64& Camera::get_buffer() 
{ 
	return selection_buffer; 
}



void Camera::SetView(const box3& bbox, const vec3& dir)
{
	real size = norm(bbox.get_maximum() - bbox.get_minimum());
	const vec3 center = bbox.get_center();	
	const real dist = size / tan(radian(angle));
	center_of_look_ = center;
	camera_pose_.create_from_normal(center + dist * dir, -dir);
	set_view();
	need_update(true);
	FitViewToBoundingBox(bbox);
}

void Camera::ResizeViewport(int w, int h)
{
	if (camera_mode_ == kImage) {

		LOG("resize view port\n");
		viewport_.Resize(w, h);

		if (selection_buffer_enable_)
			selection_buffer.resize(
				viewport_.GetX(), 
				viewport_.GetY(),
				viewport_.GetWidth(),
				viewport_.GetHeight());

		aspect_ratio = viewport_.GetAspectRatio();

		vec2 cen = (bottom_left_ + top_right_)/2.0;
		vec2 diff = top_right_ - cen;
		diff[0] = diff[1] * aspect_ratio;
		bottom_left_ = cen - diff;
		top_right_ = cen + diff;

		return;
	}
	
	viewport_.Resize(w, h);

	if (selection_buffer_enable_)
		selection_buffer.resize(
			viewport_.GetX(), 
			viewport_.GetY(),
			viewport_.GetWidth(),
			viewport_.GetHeight());

	aspect_ratio = viewport_.GetAspectRatio();
}


void Camera::ResizeViewport(int x, int y, int w, int h)
{
	if (camera_mode_ == kImage) {

		LOG("resize view port\n");
		viewport_.Resize(w, h);
		viewport_.SetPosition(ivec2(x,y));

		if (selection_buffer_enable_)
			selection_buffer.resize(
				viewport_.GetX(), 
				viewport_.GetY(),
				viewport_.GetWidth(),
				viewport_.GetHeight());

		aspect_ratio = viewport_.GetAspectRatio();

		vec2 cen = (bottom_left_ + top_right_)/2.0;
		vec2 diff = top_right_ - cen;
		diff[0] = diff[1] * aspect_ratio;
		bottom_left_ = cen - diff;
		top_right_ = cen + diff;

		return;
	}
	
	viewport_.Resize(w, h);
	viewport_.SetPosition(ivec2(x,y));

	if (selection_buffer_enable_)
		selection_buffer.resize(
			viewport_.GetX(), 
			viewport_.GetY(),
			viewport_.GetWidth(),
			viewport_.GetHeight());

	aspect_ratio = viewport_.GetAspectRatio();
}


void Camera::SetTopView(const box3& model_box)
{
	if (!model_box.is_empty()) {
		real dist = norm(center_of_look_-camera_pose_.get_origin());
		vec3 cen = model_box.get_center() + (dist * vec3(0,0,1));

		camera_pose_.create_from_normal(cen, vec3(0,0,-1));
		center_of_look_ = vec3(0);
		FitViewToBoundingBox(model_box);
	}
	else {
		center_of_look_ = vec3(0);
		real dist = grid_size_ / tan(radian(angle/2.0));
		vec3 cen = dist * vec3(0,0,1);
		camera_pose_.create_from_normal(cen, vec3(0,0,-1));
		FitViewToBoundingBox(model_box);
	}
}

void Camera::SetObliqueView(const box3& model_box)
{
	if (!model_box.is_empty()) {
		SetView(model_box, unit(vec3(1,1,0.5)));
	}
	else {
		center_of_look_ = vec3(0);
		real dist = grid_size_ / tan(radian(angle/2.0));
		vec3 cen = (dist * unit(vec3(1,1,1)));
		frame f(cen,unit(vec3(-1,-1,-1)),vec3(0,0,1));
		camera_pose_ = f;
		FitViewToBoundingBox(model_box);
	}
}
void Camera::FitViewToBoundingBox(const box3& model_box)
{
	if (model_box.is_empty()) return;

	//log();
	vec3 cen = model_box.get_center();
	line l(camera_pose_.get_origin(), camera_pose_.get_origin() + camera_pose_.basis[2]);
	vec3 cen_move = l.get_closest_point(cen);
	cen_move = cen - cen_move;
	camera_pose_.set_origin(camera_pose_.get_origin() + cen_move);
	real xs, ys, zs;
	model_box.get_size(xs,ys,zs);
	real size = sqrt(xs*xs + ys*ys + zs*zs) / 2.0;
	real len = size / tan(radian(angle/2.0));
	vec3 cam_dir = camera_pose_.basis[2];

	real mult = 1.0;
	if (inner(cam_dir, -unit(camera_pose_.get_origin() - cen)) < 0.0) {
		mult = -1.0;
	}
	camera_pose_.set_origin(cen + (mult*len*unit(camera_pose_.get_origin() - cen)));

	center_of_look_ = cen;

	if (camera_mode_ == kParallel) {
		scale_ = grid_size_/(size*1.3);
	}

	update_center_of_look();
	//log();
}

void Camera::drawCameraImage() const
{
	if (camera_mode_ != kImage) return;

	glPushAttrib(GL_ALL_ATTRIB_BITS);
	glMatrixMode(GL_MODELVIEW);
	glPushMatrix();
	glLoadIdentity();

	camera_pose_.push_to_world();

	real y = (near_ + 0.1) * tan(radian(angle/2.0));
	real x = y * image_aspect_ratio_;
	vec3 a(x, y,(near_ + 0.1));
	vec3 b(-x, y,(near_ + 0.1));
	vec3 c(-x, -y, (near_ + 0.1));
	vec3 d(x, -y, (near_ + 0.1));

	gl_color(vec4(1.0,1.0,1.0,0.1));
	glEnable(GL_TEXTURE_2D);
	glBindTexture(GL_TEXTURE_2D, texture_id_);
	glBegin(GL_QUADS);
	
	glTexCoord2f(0, 1);
	gl_vertex(a);

	glTexCoord2f(1, 1);
	gl_vertex(b);

	glTexCoord2f(1, 0);
	gl_vertex(c);

	glTexCoord2f(0, 0);
	gl_vertex(d);
	glEnd();

	glBindTexture(GL_TEXTURE_2D, 0);
	camera_pose_.pop();

	glPopMatrix();
	glPopAttrib();
}

void  Camera::DrawCameraGeometry(bool full) const
{
	glPushAttrib(GL_ALL_ATTRIB_BITS);
	glMatrixMode(GL_MODELVIEW);
	glPushMatrix();
	glLoadIdentity();

	camera_pose_.push_to_world();

	real y = image_depth_ * tan(radian(angle/2.0));
	real x = y * aspect_ratio;
	vec3 a(x,y,image_depth_);
	vec3 b(-x,y,image_depth_);
	vec3 c(-x,-y, image_depth_);
	vec3 d(x,-y, image_depth_);
	
	gl_color(vec3(1,0,0));
	glLineWidth(1.0);

	glDisable(GL_LIGHTING);
	glBegin(GL_LINES);
	gl_vertex(vec3(0));
	gl_vertex(a);
	gl_vertex(vec3(0));
	gl_vertex(b);
	gl_vertex(vec3(0));
	gl_vertex(c);
	gl_vertex(vec3(0));
	gl_vertex(d);
	gl_vertex(a);
	gl_vertex(b);
	gl_vertex(b);
	gl_vertex(c);
	gl_vertex(c);
	gl_vertex(d);
	gl_vertex(d);
	gl_vertex(a);
	gl_color(vec3(1,0,0));
	gl_vertex(vec3(0));
	gl_vertex(vec3(10,0,0));
	gl_color(vec3(0,1,0));
	gl_vertex(vec3(0));
	gl_vertex(vec3(0,10,0));
	gl_color(vec3(0,0,1));
	gl_vertex(vec3(0));
	gl_vertex(vec3(0,0,10));
	glEnd();


	camera_pose_.pop();

	glPopMatrix();
	glPopAttrib();

}

void  Camera::DrawSkyView() const
{

	if (camera_mode_ == kImage) {
		drawCameraImage();
		return;
	}

	int w = viewport_.GetWidth();
	int h = viewport_.GetHeight();

	glViewport(viewport_.GetX(),viewport_.GetY(),w,h);
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity( );

    gluOrtho2D(0, w, h, 0);

    glMatrixMode(GL_MODELVIEW);
    glPushMatrix();
    glLoadIdentity( );

    glPushAttrib(GL_ALL_ATTRIB_BITS);   
    glDisable( GL_DEPTH_TEST );   
    glDisable( GL_LIGHTING );   
	glDisable(GL_CULL_FACE);
	glDisable(GL_TEXTURE_2D);

    glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);


	gl_color(vec3(1,0,0));
	glLineWidth(3.0);

	if (!horizon_reversed_) {

		// earth
		gl_color(vec4(0.875, 0.875, 0.875, 1.0));
		glBegin(GL_QUADS);
		glVertex2i(0,h-horizon_line_y_ < 0 ? 0 : h-horizon_line_y_);
		glVertex2i(0,h);
		glVertex2i(w,h);
		glVertex2i(w,h-horizon_line_y_ < 0 ? 0 : h-horizon_line_y_);
		
		// heaven
		gl_color(vec4(0.7,0.7,1.0,1.0));
		glVertex2i(0,0);
		gl_color(vec4(1.0,1.0,1.0,1.0));
		glVertex2i(0,h-horizon_line_y_ > h ? h : h-horizon_line_y_);
		gl_color(vec4(1.0,1.0,1.0,1.0));
		glVertex2i(w,h-horizon_line_y_ > h ? h : h-horizon_line_y_);
		gl_color(vec4(0.7,0.7,1.0,1.0));
		glVertex2i(w,0);
		glEnd();
	}
	else {

		// earth
		gl_color(vec4(0.875, 0.875, 0.875,1.0));
		glBegin(GL_QUADS);
		glVertex2i(0,0);
		glVertex2i(0,h-horizon_line_y_ > h ? h : h-horizon_line_y_);
		glVertex2i(w,h-horizon_line_y_ > h ? h : h-horizon_line_y_);
		glVertex2i(w,0);

		// heaven
		gl_color(vec4(1.0,1.0,1.0,1.0));
		glVertex2i(0,h-horizon_line_y_ < 0 ? 0 : h-horizon_line_y_);
		gl_color(vec4(0.7,0.7,1.0,1.0));
		glVertex2i(0,h);
		gl_color(vec4(0.7,0.7,1.0,1.0));
		glVertex2i(w,h);
		gl_color(vec4(1.0,1.0,1.0,1.0));
		glVertex2i(w,h-horizon_line_y_ < 0 ? 0 : h-horizon_line_y_);
		glEnd();
	}
	
	glPopAttrib();
	glLineWidth(1.0);

    glPopMatrix();   

	glEnable(GL_LIGHTING);	
}

void Camera::computeVanishingLine()
{

	if (camera_mode_ == kParallel) {

		if (inner(camera_pose_.basis[2],vec3(0,0,1)) < 0.0) {
			horizon_reversed_ = true;
			horizon_line_y_ = 0;
		}
		else {
			horizon_reversed_ = false;
			horizon_line_y_ = 0;
		}
		return;
	}

	vec2 p1 = projectToImagePlane(vec3(0,0,0));
	vec2 p2 = projectToImagePlane(vec3(1,0,0));
	vec2 p3 = projectToImagePlane(vec3(0,1,0));
	vec2 p4 = projectToImagePlane(vec3(1,1,0));

	vec2 p5 = projectToImagePlane(vec3(0, 0, 1));
	vec2 its;
	int ret = intersect(p1, p2, p3, p4, its);
	
	if (ret) {
		horizon_line_y_ = its[1];
	}
	else {
		intersect(p1,p3,p2,p4,its);
		horizon_line_y_ = its[1];
	}

	if (p5[1] >= p1[1]) horizon_reversed_ = false;
	else horizon_reversed_ = true;

}

void Camera::SetCameraPose(const frame& cam)
{
	camera_pose_ = cam;
	need_update(true);
}

real Camera::far_dist() const
{
	real cam_dist = norm(camera_pose_.get_origin());
	if (cam_dist > 10.0) {
		return const_far_dist_ + cam_dist * 10.0;
	}

	return const_far_dist_;
}


void Camera::set_view()
{
	glViewport(viewport_.GetX(),viewport_.GetY(),viewport_.GetWidth(),viewport_.GetHeight());
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();

	if (camera_mode_ == kPerspective) {
		gluPerspective(angle, aspect_ratio, near_, far_);
	}
	else if (camera_mode_ == kParallel) {
		real left = -(grid_size_ * aspect_ratio)/2.0;
		real right = -left;
		real bottom = -grid_size_/2.0;
		real top = -bottom;
		glOrtho(left / scale_, right / scale_, bottom / scale_, top / scale_, near_, far_);
	}
	else if (camera_mode_ == kImage) {
		glFrustum( bottom_left_[0], top_right_[0], bottom_left_[1], top_right_[1], near_, far_);
	}

	vec3 org = camera_pose_.get_origin();
	vec3 ref = org + camera_pose_.basis[2];
	vec3 up = camera_pose_.basis[1];

	gluLookAt(org[0], org[1], org[2], ref[0], ref[1], ref[2], up[0], up[1], up[2]);	

	
	if(camera_mode_ != kImage)
		computeVanishingLine();

	glMatrixMode(GL_MODELVIEW);

	
}

void Camera::set_render_view()
{
	glViewport(viewport_.GetX(),viewport_.GetY(),viewport_.GetWidth(),viewport_.GetHeight());
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();

	if (camera_mode_ == kPerspective) {
		gluPerspective(angle, aspect_ratio, near_, far_);
	}
	else if (camera_mode_ == kParallel) {
		real left = -(grid_size_ * aspect_ratio)/2.0;
		real right = -left;
		real bottom = -grid_size_/2.0;
		real top = -bottom;
		glOrtho(left / scale_, right / scale_, bottom / scale_, top / scale_, near_, far_);
	}
	else if (camera_mode_ == kImage) {
		glFrustum( bottom_left_[0], top_right_[0], bottom_left_[1], top_right_[1], near_, far_);
	}

	vec3 org = camera_pose_.get_origin();
	vec3 ref = org + camera_pose_.basis[2];
	vec3 up = camera_pose_.basis[1];

	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
	gluLookAt(org[0], org[1], org[2], ref[0], ref[1], ref[2], up[0], up[1], up[2]);	

	if(camera_mode_ != kImage)
		computeVanishingLine();
	
}

vec3  Camera::active_plane_normal(const frame& local, const vec3& center) const
{
	return active_plane_normal(local, projectToImagePlane(local.to_world(center)), center);
}

vec3 Camera::active_plane_normal(const frame& local, const vec2& image_pnt, const vec3& local_org) const
{

	if (camera_mode_ == kParallel) {
		real a = fabs(inner(camera_pose_.z_axis(), local.to_world_normal(vec3(1,0,0))));
		real b = fabs(inner(camera_pose_.z_axis(), local.to_world_normal(vec3(0,1,0))));
		real c = fabs(inner(camera_pose_.z_axis(), local.to_world_normal(vec3(0,0,1))));

		//LOG("%f %f %f\n", a, b, c);

		if (a>b && a>c) return vec3(1,0,0);
		if (b>a && b>c) return vec3(0,1,0);
		if (c>a && c>b) return vec3(0,0,1);


		return vec3(0,0,1);
	}

	real x = image_pnt[0], y = image_pnt[1];

	line ray1;
	getViewRay(local, vec2(x,y), ray1);
	vec3 z = local.to_model_normal(camera_pose_.basis[2]);

	plane floor_(vec3(0,0,1), local_org);

	vec3 pp1, pp2;

	if (floor_.intersect(ray1,pp1)) {

		vec3 c1 = pp1-ray1.get_position();

		if (inner(z, c1) > 0.0) {
			real dist1 = norm(pp1-ray1.get_position());
			getViewRay(local, vec2(x,y-1.0), ray1);
			if (floor_.intersect(ray1, pp1)) {
				vec3 c2 = pp1 - ray1.get_position();
				if (inner(z, c2) > 0.0) {
					real dist0 = norm(pp1-ray1.get_position());
					getViewRay(local, vec2(x,y+1.0), ray1);
					if (floor_.intersect(ray1, pp1)) {
						vec3 c3 = pp1 - ray1.get_position();
						if (inner(z, c3) > 0.0) {
							real dist2 = norm(pp1-ray1.get_position());
							real v1, v2;
							if (dist0 <= dist1) {
								v1 = dist1 - dist0;
								v2 = dist2 - dist1;
							}
							else {
								v1 = dist1 - dist2;
								v2 = dist0 - dist1;
							}

							//LOG("velocity 1 %f, velocity 2 %f, accel %f\n", v1, v2, v2/v1);

							if (!apx_equal(v1,0.0) && !apx_equal(v2,0.0) && v2/v1 <= 1.03) {
								return vec3(0,0,1.0);
							}
						}
					}
				}
			}
		}
	}


	plane pl;

    vec3 org = getCameraPose().get_origin();
    vec3 z_dir = getCameraPose().basis[2];
    real n = (near_dist() + far_dist())/2.0;

    vec3 pnt = org + ((z_dir * n) + 3.0);
    
    pl.reset(pnt, z_dir);

	vec3 O = projectToPlane(image_pnt, pl);
    vec3  A(1.0,0.0,0.0), B(0.0,1.0,0.0);
	A = O + local.to_world_normal(A);
	B = O + local.to_world_normal(B);


    vec2 wo = projectToImagePlane(O);
    vec2 wa = projectToImagePlane(A);
    vec2 wb = projectToImagePlane(B);


	real yz = norm(wa-wo);
	real zx = norm(wb-wo);


    if (yz <= zx)
		return vec3(1,0,0);

    if (zx <= yz)
		return vec3(0,1,0);

	return vec3(0,0,1);
	//LOG("fault\n");
}

vec3  Camera::active_plane_normal(const vec3& center) const
{
	return active_plane_normal(projectToImagePlane(center), center);
}

vec3 Camera::active_plane_normal(const vec2& image_pnt, const vec3& world_org) const
{

	if (camera_mode_ == kParallel) {
		real a = fabs(inner(camera_pose_.z_axis(), vec3(1,0,0)));
		real b = fabs(inner(camera_pose_.z_axis(), vec3(0,1,0)));
		real c = fabs(inner(camera_pose_.z_axis(), vec3(0,0,1)));

		//LOG("%f %f %f\n", a, b, c);

		if (a>b && a>c) return vec3(1,0,0);
		if (b>a && b>c) return vec3(0,1,0);
		if (c>a && c>b) return vec3(0,0,1);


		return vec3(0,0,1);
	}

	real x = image_pnt[0], y = image_pnt[1];

	line ray1;
	getViewRay(vec2(x,y), ray1);
	vec3 z =camera_pose_.basis[2];

	plane floor_(vec3(0,0,1), world_org);

	vec3 pp1, pp2;

	if (floor_.intersect(ray1,pp1)) {

		vec3 c1 = pp1-ray1.get_position();

		if (inner(z, c1) > 0.0) {
			real dist1 = norm(pp1-ray1.get_position());
			getViewRay(vec2(x,y-1.0), ray1);
			if (floor_.intersect(ray1, pp1)) {
				vec3 c2 = pp1 - ray1.get_position();
				if (inner(z, c2) > 0.0) {
					real dist0 = norm(pp1-ray1.get_position());
					getViewRay(vec2(x,y+1.0), ray1);
					if (floor_.intersect(ray1, pp1)) {
						vec3 c3 = pp1 - ray1.get_position();
						if (inner(z, c3) > 0.0) {
							real dist2 = norm(pp1-ray1.get_position());
							real v1, v2;
							if (dist0 <= dist1) {
								v1 = dist1 - dist0;
								v2 = dist2 - dist1;
							}
							else {
								v1 = dist1 - dist2;
								v2 = dist0 - dist1;
							}

							//LOG("velocity 1 %f, velocity 2 %f, accel %f\n", v1, v2, v2/v1);

							if (!apx_equal(v1,0.0) && !apx_equal(v2,0.0) && v2/v1 <= 1.03) {
								return vec3(0,0,1.0);
							}
						}
					}
				}
			}
		}
	}


	plane pl;

    vec3 org = getCameraPose().get_origin();
    vec3 z_dir = getCameraPose().basis[2];
    real n = (near_dist() + far_dist())/2.0;

    vec3 pnt = org + ((z_dir * n) + 3.0);
    
    pl.reset(pnt, z_dir);

	vec3 O = projectToPlane(image_pnt, pl);
    vec3  A(1.0,0.0,0.0), B(0.0,1.0,0.0);
	A = O + A;
	B = O + B;


    vec2 wo = projectToImagePlane(O);
    vec2 wa = projectToImagePlane(A);
    vec2 wb = projectToImagePlane(B);


	real yz = norm(wa-wo);
	real zx = norm(wb-wo);


    if (yz <= zx)
		return vec3(1,0,0);

    if (zx <= yz)
		return vec3(0,1,0);

	return vec3(0,0,1);
	//LOG("fault\n");
}


void  Camera::getImagePlane(plane& pl) const
{
    vec3 org = getCameraPose().get_origin();
    vec3 z_dir = getCameraPose().basis[2];

    real n = norm(center_of_look()-camera_pose_.get_origin());

    vec3 pnt = org + ((z_dir * n) + 3.0);
    
    pl.reset(pnt, z_dir);
}


vec2 Camera::projectToImagePlane(const vec3& p) const
{
	if (camera_mode_ == kParallel) {
		real left = (-(grid_size_ * aspect_ratio)/2.0) / scale_;
		real bottom = (-grid_size_/2.0) / scale_;
		vec3 pnt = camera_pose_.to_model(p);
		
		real lsize = left * -2.0;
		real rsize = bottom * -2.0;
		real a = -pnt[0] - left;
		real b = pnt[1] - bottom;

		real x = (a * viewport_.GetWidth())/lsize;
		real y = (b * viewport_.GetHeight())/rsize;

		vec2 ret(x, y);

		return ret;
	}

	if (camera_mode_ == kImage) {
		plane pl(vec3(0.0,0.0,1.0), near_);
		vec3 pnt = camera_pose_.to_model(p);
		line ray(vec3(0.0,0.0,0.0), pnt);
		pl.intersect(ray, pnt);

		vec2 a(-pnt[0], pnt[1]);
		vec2 ret = (a-bottom_left_)/(top_right_ - bottom_left_);
		ret = ret * vec2(viewport_.GetWidth(), viewport_.GetHeight());
		return ret;
	}

	plane pl(vec3(0.0,0.0,1.0), near_dist());
	vec3 pnt = camera_pose_.to_model(p);

	line ray(vec3(0.0,0.0,0.0), pnt);
	pl.intersect(ray, pnt);
	vec2 a(pnt[0], pnt[1]);

	real hr = tan(radian(angle)/2.0) * near_dist();
	real wr = hr * aspect_ratio;

	a = a/vec2(-wr, hr);

	real cenw= ((real)viewport_.GetWidth())/2.0;
	real cenh= ((real)viewport_.GetHeight())/2.0;

	a = a * vec2(cenw, cenh);

	vec2 r = a + vec2(cenw, cenh);
	return r;
}
vec3  Camera::projectToPlane(const vec2& a, const plane& ref, const frame& local) const
{
	line ray;
	getViewRay(local, a, ray);
	vec3 pnt;
	ref.intersect(ray, pnt);
	return pnt;
}

vec3 Camera::projectToPlane(const vec2& a, const plane& ref) const
{
	//LOG("project %f %f\n", a[0], a[1]);

	if (camera_mode_ == kParallel) {
		real left = (-(grid_size_ * aspect_ratio)/2.0) / scale_;
		real bottom = (-grid_size_/2.0) / scale_;

		real a0 = a[0] / viewport_.GetWidth();
		real a1 = a[1] / viewport_.GetHeight();

		
		real lsize = left * -2.0;
		real rsize = bottom * -2.0;

		a0 = lsize * a0;
		a1 = rsize * a1;		
		
		real a00 = a0 + left;
		real a11 = a1 + bottom;


		vec3 pnt(-a00, a11, 0.0);
		pnt = camera_pose_.to_world(pnt);
		line ray(pnt, pnt+camera_pose_.z_axis());

		ref.intersect(ray, pnt);
		return pnt;
	}

	if (camera_mode_ == kImage) {

		vec2 norm_coord = a/vec2(viewport_.GetWidth(), viewport_.GetHeight());
		vec2 new_coord = (norm_coord * (top_right_ - bottom_left_)) + bottom_left_;
		vec3 r2(-new_coord[0], new_coord[1], near_);
		vec3 r1 = camera_pose_.get_origin();
		r2 = camera_pose_.to_world(r2);

		vec3 dir = unit(r2 - r1);
		real t = (ref.d - inner(ref.n, r1))/inner(ref.n, dir);

		return (r1 + t * dir);

	}

	real cenw= ((real)viewport_.GetWidth())/2.0;
	real cenh= ((real)viewport_.GetHeight())/2.0;

	vec2 norm_coord = (a - vec2(cenw, cenh))/vec2(cenw, cenh);

	real hr = tan(radian(angle)/2.0) * near_dist();
	real wr = hr * aspect_ratio;

	vec2 new_coord = norm_coord * vec2(-wr, hr);

	vec3 r2(new_coord[0], new_coord[1], near_dist());

	r2 = camera_pose_.to_world(r2);

    vec3 r1 = camera_pose_.get_origin();

    vec3 dir = unit(r2 - r1);
    real t = (ref.d - inner(ref.n, r1))/inner(ref.n, dir);

    return (r1 + t * dir);
}

void Camera::getViewRay(const frame& local, const vec2& a, line& ray) const
{
	getViewRay(a, ray);
	ray = local.to_model(ray);
}

void Camera::getViewRay(const vec2& a, line& ray) const
{
	if (camera_mode_ == kParallel) {

		//LOG("get view ray %f %f\n", a[0], a[1]);
		real left = (-(grid_size_ * aspect_ratio)/2.0) / scale_;
		real bottom = (-grid_size_/2.0) / scale_;

		real a0 = a[0] / viewport_.GetWidth();
		real a1 = a[1] / viewport_.GetHeight();

		
		real lsize = left * -2.0;
		real rsize = bottom * -2.0;

		a0 = lsize * a0;
		a1 = rsize * a1;		
		
		real a00 = a0 + left;
		real a11 = a1 + bottom;

		//LOG("%f %f\n", a00, a11);

		vec3 pnt(-a00, a11, 0.0);
		pnt = camera_pose_.to_world(pnt);
		line ray1(pnt, pnt+camera_pose_.z_axis());
		ray = ray1;
		return;
	}

	if (camera_mode_ == kImage) {
		vec2 norm_coord = a/vec2(viewport_.GetWidth(), viewport_.GetHeight());
		vec2 new_coord = (norm_coord * (top_right_ - bottom_left_)) + bottom_left_;
		vec3 r2(-new_coord[0], new_coord[1], near_);
		vec3 r1 = camera_pose_.get_origin();

		r2 = camera_pose_.to_world(r2);
		ray.set_value(r1, r2);
		return;
	}

	real cenw= ((real)viewport_.GetWidth())/2.0;
	real cenh= ((real)viewport_.GetHeight())/2.0;

	vec2 norm_coord = (a - vec2(cenw, cenh))/vec2(cenw, cenh);

	real hr = tan(radian(angle)/2.0) * near_dist();
	real wr = hr * aspect_ratio;

	vec2 new_coord = norm_coord * vec2(-wr, hr);

	vec3 coord(new_coord[0], new_coord[1], near_dist());

	coord = camera_pose_.to_world(coord);
	ray.set_value(camera_pose_.get_origin(), coord);

}

void Camera::updateTrackBallRadius()
{
	if (camera_mode_ == kImage) return;

    set_view();
    line ray;
    getViewRay(vec2((real)viewport_.GetWidth()/2.0, (real)viewport_.GetHeight()/2.0),  ray);
    plane temp_ref;
    temp_ref.reset(center_of_look(), ray.get_direction());

    vec3 p1 = projectToPlane(vec2(0), temp_ref);
    vec3 p2 = projectToPlane(vec2(viewport_.GetWidth(), viewport_.GetHeight()), temp_ref);
    rotation_radius_ = norm(p1-p2)/3.5;
}

void Camera::rotateCamera(const ivec2& p1, const ivec2& p2)
{
	if (camera_mode_ == kImage) {
		if (camera_modeling_tool_) {
			camera_modeling_tool_->SwitchBackTo3DModelView();
		}
		else {
			camera_mode_ = kPerspective;
		}

		updateTrackBallRadius();
		return;
	}

	quater r1, r2;
	decomposeRotation(p1, p2, r1, r2);
	rotateCamera(r1, r2);
	set_view();
	need_update(true);
}

void Camera::panCamera(const ivec2& pre_p, const ivec2& p)
{
	if (camera_mode_ == kImage) {
		vec2 del = (vec2(p - pre_p) / (viewport_.GetWidth())) * get_view_size(camera_pose_.to_world(vec3(0,0,near_)));

		vec2 mov(del[0], del[1]);
		bottom_left_ -= mov;
		top_right_ -= mov;
		set_view();
		need_update(true);
		return;
	}

    vec2 del = (vec2(p - pre_p) / (viewport_.GetWidth())) * get_view_size(center_of_look());

    vec3 mov(del[0], -del[1], 0);
	frame camera = camera_pose_;

    mov = camera.to_world(mov);
    mov = mov - camera.get_origin();

	camera.translate_frame(mov);

	SetCameraPose(camera);

    set_center_of_look(center_of_look() + mov);

	set_view();
	need_update(true);
}

void Camera::updateViewFrustum()
{
	if (camera_mode_ != kImage) return;
	real hr = tan(radian(angle)/2.0) * near_;
	real wr = hr * aspect_ratio;

	bottom_left_ = vec2(-wr, -hr);
	top_right_ = vec2(wr, hr);
}

void Camera::zoomCamera(const ivec2& pre_p, const ivec2& p)
{
	if (camera_mode_ == kImage) {
		vec2 del = (vec2(p - pre_p) / (viewport_.GetWidth())) * norm(camera_pose_.get_origin()-camera_pose_.to_world(vec3(0,0,near_)));
		vec2 cen = (bottom_left_ + top_right_)/2.0;
		bottom_left_ = ((bottom_left_ - cen) + del[0]) + cen;
		top_right_ = ((top_right_ - cen) - del[0]) + cen;
		set_view();
		need_update(true);
		return;
	}

	vec2 del = (vec2(p - pre_p) / (viewport_.GetWidth())) * norm(camera_pose_.get_origin()-center_of_look());
	vec3 mov(0, 0, del[0]);
	frame camera = camera_pose_;
	mov = camera.to_world(mov);
	mov = mov - camera.get_origin();
	camera.translate_frame(mov);
	SetCameraPose(camera);

	if (camera_mode() == Camera::kParallel) {

		vec2 del = vec2(p - pre_p) / (viewport_.GetHeight()/2);

		set_scale(scale() * (1.0 + del[0]));
	}
	set_view();
	need_update(true);
}

void Camera::focusZoomCamera(int zdelta, const ivec2& pos)
{
	if (camera_mode_ == kImage) {

		line ray;
		getViewRay(vec2(pos), ray);

		vec3 dir = camera_pose_.to_model_normal(ray.get_direction());
		ray.set_position(vec3(0));
		ray.set_direction(dir);

		plane pl(vec3(0,0,1), near_);
		vec3 cen;
		pl.intersect(ray, cen);
		vec2 center(-cen[0], cen[1]);

		real zoom_factor = (zdelta>0?0.9:1.1);
		bottom_left_ = ((bottom_left_ - center) * zoom_factor) + center;
		top_right_ = ((top_right_ - center) * zoom_factor) + center;

		set_view();
		need_update(true);
		return;
	}

	frame camera = camera_pose_;
	
	plane pl;
	getImagePlane(pl);

	vec3 pos1 = projectToPlane(vec2(pos[0],pos[1]), pl);

	vec3 cam_dir = camera.z_axis();
	vec3 cam_pos = camera.get_origin();
	vec3 cen_model = center_of_model();
	line ray(cam_pos, cam_pos + cam_dir);

	
	real zoom_dist = norm(cen_model - cam_pos);
	getViewRay(vec2(pos[0],pos[1]), ray);
	vec3 ray_dir = ray.get_direction();

	ray_dir = ray_dir * zoom_dist * 0.1 * (zdelta>0?1.0:-1.0);

	if (camera_mode() == Camera::kPerspective) {
		camera.translate_frame(ray_dir);
		SetCameraPose(camera);
		updateTrackBallRadius();
	}

		
	if (camera_mode() == Camera::kParallel) {
		set_scale(scale() * (zdelta>0?1.1: 0.9));

		vec3 pos2 = projectToPlane(vec2(pos[0],pos[1]),pl);
		vec3 dir = pos1 - pos2;

		camera.translate_frame(dir);
		SetCameraPose(camera);
		updateTrackBallRadius();
		
	}

	update_center_of_look();
	set_view();
	need_update(true);
}

void Camera::decomposeRotation(const ivec2& a, const ivec2& b, quater& rot1, quater& rot2)
{
    set_view();

    sphere_section_projector pjt(sphere(vec3(0), rotation_radius_), this);

    frame space;
    space.set_origin(center_of_look());
    pjt.set_local_space(space);
    vec3 p1, p2, p3, p4;
    p1 = pjt.project(vec2(a[0], (real)viewport_.GetHeight()/2.0));
	p2 = pjt.project(vec2((real)viewport_.GetWidth()/2.0, a[1]));
    p3 = pjt.project(vec2(b[0], (real)viewport_.GetHeight()/2.0));
	p4 = pjt.project(vec2((real)viewport_.GetWidth()/2.0, b[1]));

	rot1 = inv(pjt.get_rot(p1,p3));
	rot2 = inv(pjt.get_rot(p2,p4));
}


void Camera::rotateCamera(const quater& rot1, const quater& rot2)
{
	frame camera = camera_pose_;

	quater r1 = rot1, r2 = rot2;

    if (r1 != _zero_quater() || r2 != _zero_quater()) { 
		
		vec3 cpos = camera.get_origin() - center_of_look();
		
		real phi = 0;
		vec3 axis(0);

		if (r1 != _zero_quater())
			get_rotation(r1, phi, axis);

		vec3 up_dir = camera.basis[1];
		if (r2 != _zero_quater()) {
			cpos = rot(r2, cpos);
			up_dir = rot(r2, up_dir);
		}

		if (fabs(inner(up_dir, vec3(0,0,1))) > 0.0) {
			if (r1 != _zero_quater()) {
				if (inner(vec3(0,0,1), axis) < 0.0)
					phi *= -1.0;

				r1 = orient(phi, vec3(0,0,1));
				if (r1 != _zero_quater()) {
					up_dir = rot(r1, up_dir);
					cpos = rot(r1, cpos);
				}
			}
		}


		camera.set_origin(cpos + center_of_look());

		real val = inner(up_dir, vec3(0,0,1));


		if (val < -0.0001) {	
			camera.set_look(center_of_look() - camera.get_origin(), vec3(0,0,-1));
			SetCameraPose(camera);
		}
		else  if (val > 0.0001) {
			camera.set_look(center_of_look() - camera.get_origin(), vec3(0,0,1));
			SetCameraPose(camera);
		}
		else {
			camera.set_look(center_of_look() - camera.get_origin(), up_dir);
			SetCameraPose(camera);
		}
		
    }
}
}; // namespace graphics