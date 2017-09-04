
#ifndef     __CameraInterface_h
#define		__CameraInterface_h

#include	<graphics/sys.h>
#include	<graphics/event.h>

#include    <stdio.h>

#include    <graphics/real.h>
#include    <graphics/frame.h>
#include    <graphics/Camera.h>
#include    <graphics/Viewer.h>

class QWidget;

namespace graphics {

class CameraInterface {

public:

	CameraInterface ();
	virtual ~CameraInterface();

	virtual void initialize();

	virtual void beginDraw();
	virtual void draw();
	virtual void endDraw();

	virtual void beginDraw2D();
	virtual void draw2D();
	virtual void endDraw2D();
	
	virtual void resizeViewport(int w, int h);

	virtual void resizeViewport(int x, int y, int w, int h);

	

	const Camera& getCamera() const { return camera_;  }

	Camera& getCamera() { return camera_;  }


	int getWidth() const;

	int getHeight() const;



	void widget(QWidget* q) { widget_ = q; }
	
	QWidget* widget() const { return widget_; }


	virtual bool processEvent(Event* ev);

	virtual void rotX(int val) {}
	virtual void rotY(int val) {}
	virtual void rotZ(int val) {}

	virtual void tranX(float val) {}
	virtual void tranY(float val) {}
	virtual void tranZ(float val) {}

	virtual void scale(float val) {}

	virtual void setNext() {}
protected:

	virtual void wheelEventProc(int zdelta, const ivec2& pos);

    virtual void mouseMoveEventProc(const ivec2& p, int shift, int cont, int alt);

    virtual void mouseDownEventProc(const ivec2& p, int shift, int cont, int alt);

    virtual void mouseRightDownEventProc(const ivec2& p, int shift, int cont, int alt);

    virtual void mouseUpEventProc(const ivec2& p, int shift, int cont, int alt);

    virtual void mouseMiddlemouseDownEventProc(const ivec2& p, int shift, int cont, int alt);

	virtual void mouseDoubleClickEventProc(const ivec2& p, int shift, int cont, int alt);

	virtual void characterInputEventProc(graphics::Event*);

	virtual void keyboardDownEventProc(graphics::Event*);

	virtual void keyboardUpEventProc(graphics::Event*);


	void		drawAxes() const;

	bool		draw_axes_;
    Camera		camera_;			// opengl viewer : angle, far distance, camera, etc

    ivec2		org_p, pre_p;

	QWidget*	widget_;
};

}

#endif