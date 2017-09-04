#ifndef __ModelView_h
#define __ModelView_h

namespace graphics {

class ModelView {
public:
	ModelView() {}

	virtual void UpdateDisplayList() = 0;
};

};
#endif