// Camera.h: interface for the Camera class.
//
//////////////////////////////////////////////////////////////////////

#ifndef	    __Camera_h
#define	    __Camera_h

#include "graphics/sys.h"
#include "graphics/real.h"
#include "graphics/frame.h"
#include "graphics/virtual_model.h"
#include "graphics/CameraModeling.h"

#include "graphics/viewport.h"
#include "graphics/color_buffer64.h"

namespace graphics {


class Camera  
{
public:

	enum CameraMode { kPerspective, kParallel, kImage};

public:

	Camera() {}

    Camera(int w, int h);

    Camera(const Camera& val);

	void reset();

    virtual ~Camera();

	virtual void ResizeViewport(int w, int h);

	virtual void ResizeViewport(int x, int y, int w, int h);

	int width() const { return viewport_.GetWidth(); }
	
	int height() const { return viewport_.GetHeight(); }

	void set_width(int a) ;

	void set_height(int a) ;

	int GetWidth() const;

	int GetHeight() const;

	// recalculate the far clipping plane distance
	// according to camera position
	real far_dist() const;

	void set_camera_mode(CameraMode mode);

	CameraMode camera_mode() const;


    inline const frame& getCameraPose() const;

	inline frame& getCameraPose();



	// Transform the camera by frame f
	void Transform(const frame& f);


	// Compute the size of the viewing plane (perpendicular to
	// the viewing direction and located at the input 'center'.
	inline real  get_view_size(const vec3& center) const;

	//
	// Fit the view to the input bounding box
	//
	void	FitViewToBoundingBox(const box3& model_box);

	void	SetTopView(const box3& model_box);

	void    SetObliqueView(const box3& model_box);

	void	SetView(const box3& bbox, const vec3& dir);

	real	get_angle() const { return angle; }

	real    get_near() const { return near_; }

	void	set_near(real n) { near_ = n; }

	real	get_aspect_ratio() const { return aspect_ratio; }

	void	set_far(real f) { far_ = f; }

	real    get_far() const { return far_; }

    //
    // Set the projection view for OpenGL drawings as well as determine the
    // default plane for the view in global space.
	// Note that I switched 4,5 items for modelling purpose. Therefore, in OpenGL rendering
	// gl_ModelView_Matrix is identity. For shading and rendering, use set_render_view() instead.
	//  1. view port setting
	//  2. glMatrixMode(GL_PROJECTION);
	//  3. Viewing frustum
	//  4. gluLookAt
	//  5. glMatrixMode(GL_MODELVIEW);

    virtual void  set_view();

    // Set the projection view for rendering for OpenGL drawings as well as determine the
    // default plane for the view in global space.
	//  1. view port setting
	//  2. glMatrixMode(GL_PROJECTION);
	//  3. Viewing frustum
	//  4. glMatrixMode(GL_MODELVIEW);
	//  5. gluLookAt
	//  
	virtual void set_render_view();

	virtual void  SetCameraPose(const frame& cam);


	// Set it to need update selection buffer formed by this camera
    void need_update(bool rhs);

	// Need to update the selection buffer?
    bool need_update() const ;


	vec3 active_plane_normal(const vec3& center) const;

	vec3 active_plane_normal(const vec2& image_pnt, const vec3& world_org=vec3(0,0,0)) const;

	vec3 active_plane_normal(const frame& local, const vec3& center) const;

	vec3 active_plane_normal(const frame& local, const vec2& image_pnt, const vec3& world_org=vec3(0,0,0)) const;

	void  getImagePlane(plane& pl)const;


	// Project the 3D world point, p, to the near clipping plane:
	// Return the window coordinate.
	virtual vec2  projectToImagePlane(const vec3& p) const; 

	// Project window point to the plane ref.
	// Return the world coordinate
	virtual vec3  projectToPlane(const vec2& a, const plane& ref) const;

	// project a to the ref at local coordinate system.
	virtual vec3  projectToPlane(const vec2& a, const plane& ref, const frame& local) const;
	// Compute view ray in world coordinate system, which
	// starts at the camera position and goes through the
	// image point a.
	virtual void  getViewRay(const vec2& a, line& ray) const;

	virtual void getViewRay(const frame& local, const vec2& a, line& ray) const;

	void updateTrackBallRadius();	

	void rotateCamera(const ivec2& p1, const ivec2& b);

	void panCamera(const ivec2& pre_p, const ivec2& p);

	void zoomCamera(const ivec2& pre_p, const ivec2& p);

	void focusZoomCamera(int zdelta, const ivec2& pos);

	color_buffer64& get_buffer();

	void  DrawSkyView() const;

	virtual void  DrawCameraGeometry(bool full = false) const;

	vec3 center_of_look() const { return center_of_look_; }

	void set_center_of_look(const vec3& cen) { center_of_look_ = cen; }

	real scale() const { return scale_; }

	void set_scale(real s) { scale_ = s; }

	real grid_size() const { return grid_size_; }

	void update_center_of_look();

	VirtualModel* model_associated() const { return model_associated_; }

	void set_model_associated(VirtualModel* model) { model_associated_ = model; }

	void set_cameral_modeling_tool(CameraModeling* cm_tool) { camera_modeling_tool_ = cm_tool; }

	vec3 center_of_model() const { return center_of_model_; }

	real near_dist() const { return norm(camera_pose_.get_origin()-center_of_look_) * 0.001; }

	void set_angle(real ang) { angle = ang; need_update(true); }

	void updateViewFrustum();

	void set_texture_id(unsigned int a) { texture_id_ = a; }
	void set_image_aspect_ratio(real a) { image_aspect_ratio_ = a; }

	real image_depth() const { return image_depth_; }
	void set_image_depth(real a) { image_depth_ = a; }


	ViewPort& getViewport() { return viewport_; }

	const ViewPort& getViewport() const { return viewport_; }

	void drawCameraImage() const;

	void disable_selection_buffer() { selection_buffer_enable_ = false; }

	void enable_selection_buffer() { selection_buffer_enable_ = true; }

protected:

	void rotateCamera(const quater& rot1, const quater& rot2);

	void decomposeRotation(const ivec2& a, const ivec2& b, quater& rot1, quater& rot2);

	void computeVanishingLine();


	bool	selection_buffer_enable_;

	// selection buffer
	color_buffer64	selection_buffer;

    real    angle, scale_;
    
	real	const_far_dist_;

    //real    near_dist;
    int	    grid_size_;

	ViewPort viewport_;

	//int		width_, height_;

    frame   camera_pose_;


	/* need to update the selection buffer by redrawing? */
    bool    need_update_;

	real	aspect_ratio;

	int		horizon_line_y_;
	bool	horizon_reversed_;

	real	rotation_radius_;

	CameraMode camera_mode_;

	vec3	center_of_look_;
	vec3	center_of_model_;

	// view frustum for the kImage---
	vec2    bottom_left_;
	vec2	top_right_;
	real	near_;
	real	far_;

	real    image_aspect_ratio_;
	real	image_depth_;


	unsigned int texture_id_;
	// ------------------------------

	VirtualModel* model_associated_;
	VirtualModel* default_model_associated_;

	CameraModeling* camera_modeling_tool_;


};




inline real Camera::get_view_size(const vec3& center) const
{
	plane pl;
	getImagePlane(pl);
	pl.reset(center, pl.get_normal());

	int w = GetWidth();
	int h = GetHeight();
	
	vec3 a = projectToPlane(vec2(0,h/2), pl);
	vec3 b = projectToPlane(vec2(w,h/2), pl);
	return norm(b-a);
}

inline const frame& Camera::getCameraPose() const 
{ 
    return camera_pose_; 
}

inline frame& Camera::getCameraPose() 
{ 
    return camera_pose_; 
}

}; // namespace graphics
#endif