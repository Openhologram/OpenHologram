#ifndef __Viewer_h
#define __Viewer_h

#include "graphics/ivec.h"
#include "graphics/Camera.h"

class QWidget;

namespace graphics {

class Viewer {

public:

	virtual void draw() {}

	virtual void draw_helper() {}

	virtual void update() {}

	virtual void mousePressEvent(const graphics::ivec2& p) {}
	virtual void mouseDoubleClickEvent(const graphics::ivec2& p) {}
	virtual void mouseReleaseEvent(const graphics::ivec2& p) {}

	virtual void shiftSelectDownEvent(const graphics::ivec2& p) {}
	virtual void shiftSelectUpEvent(const graphics::ivec2& p) {}
	virtual void shiftSelectMoveEvent(const graphics::ivec2& p) {}
	virtual void controlSelectDownEvent(const graphics::ivec2& p) {}
	virtual void controlSelectUpEvent(const graphics::ivec2& p) {}
	virtual void characterInput(unsigned int ch) {}

	virtual void mouseMoveEvent(const graphics::ivec2& p) {}


	virtual void view_changed(bool val) { view_changed_ = val; }

	virtual bool view_changed() const { return view_changed_; }

	virtual void command(const std::string& cmd) {}

	virtual void tool(const std::string& cmd) {}

	void widget(QWidget* q) { widget_ = q; }

	QWidget* widget() const { return widget_; }

	virtual Camera& camera() = 0;

	virtual void draw_selection_buffer() = 0;

	virtual bool view_3d() { return false; }

	virtual void view_3d(bool val) { }

	virtual void wireframemode(bool val) {}

	virtual void normalviewmode(bool val) {}

	virtual int image_width() { return 1920; }

	virtual int image_height() { return 1080; }
public:

	Viewer(): view_changed_(true), widget_(0) {}

	bool        view_changed_;

	QWidget*	widget_;
};

}

#endif