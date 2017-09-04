// Camera.h: interface for the Camera class.
//
//////////////////////////////////////////////////////////////////////

#ifndef	    __image_camera_h
#define	    __image_camera_h

#include "graphics/sys.h"
#include "graphics/real.h"
#include "graphics/frame.h"
#include "graphics/virtual_model.h"

namespace graphics {


class ImageCamera : public Camera
{
public:

	ImageCamera() {}

    ImageCamera(int w, int h);

    ImageCamera(const ImageCamera& val);


    virtual ~ImageCamera();

    //
    // Set the projection view for OpenGL drawings as well as determine the
    // default plane for the view in global space.
    virtual void  set_view();

	virtual void  SetCameraPose(const frame& cam);


	// Project the 3D world point, p, to the near clipping plane:
	// Return the window coordinate.
	virtual vec2  projectToImagePlane(const vec3& p) const; 

	// Project window point to the plane ref.
	// Return the world coordinate
	virtual vec3  projectToPlane(const vec2& a, const plane& ref) const;

	// Compute view ray in world coordinate system, which
	// starts at the camera position and goes through the
	// image point a.
	virtual void  getViewRay(const vec2& a, line& ray) const;

};


}; // namespace graphics
#endif