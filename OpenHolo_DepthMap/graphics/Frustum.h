#ifndef __Frustum_h
#define __Frustum_h

#include	"graphics/camera_map.h"

namespace graphics {

class Frustum {

public:

	Frustum(CameraMap* cm);

	void SetCamera(CameraMap* cm);
	
	void Update();

	bool IsInside(const vec3& p) const;

private:

	Frustum();

	CameraMap* geom_;
	plane      plane_[6];
};

}
#endif