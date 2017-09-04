#ifndef __CameraModeling_h
#define __CameraModeling_h

namespace graphics {

// abstract class to define camera modeling such as
// MultipleCameraModeling and ImageCamera

class CameraModeling {

public:

	CameraModeling() {}

	virtual void SwitchBackTo3DModelView() = 0;
	virtual void Transform(const frame& f) = 0;
};

};

#endif