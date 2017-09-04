#define GL_GLEXT_PROTOTYPES
#include    <graphics/gl_extension.h>
#include	"graphics/CameraInterface.h"

#include	<IL/il.h>
#include	<IL/ilu.h>
#include	<IL/ilut.h>

namespace graphics {

CameraInterface::CameraInterface()
	:camera_(100, 100)
{
	draw_axes_ = true;
}


CameraInterface::~CameraInterface()
{

}

void CameraInterface::initialize()
{

	glEnable(GL_TEXTURE_2D);
	
	glExtensionInitialize();
	

	glPixelStorei(GL_UNPACK_ALIGNMENT, 4);

	camera_.set_camera_mode(Camera::kParallel);
	camera_.SetTopView(box3(vec3(-500,-500,-1000), vec3(500,500,1000)));

}


int CameraInterface::getWidth() const
{
	return camera_.GetWidth();
}

int CameraInterface::getHeight() const
{
	return camera_.GetHeight();

}


static void SetFrontLight()
{
	int vport[4];
	double model[16];
	double proj[16];
	glGetIntegerv(GL_VIEWPORT, vport);
	glGetDoublev(GL_PROJECTION_MATRIX, proj);
	glGetDoublev(GL_MODELVIEW_MATRIX, model);
	double p[3];
	gluUnProject(vport[2]/2, vport[3]/2, 0, model, proj, vport, 
		&p[0], &p[1], &p[2]);

	float light[4];
	light[0] = p[0];
	light[1] = p[1];
	light[2] = p[2];
	light[3] = 0;
	float diffuse[4] = {0.5, 0.5, 0.5, 0.5};
	float ambient[4] = {0.2, 0.2, 0.2, 0.5};
	float spec[4] = {0.5, 0.5, 0.5, 1.0};

	glLightfv(GL_LIGHT0, GL_POSITION, light);
	glLightfv(GL_LIGHT0, GL_AMBIENT, ambient);
	glLightfv(GL_LIGHT0, GL_DIFFUSE, diffuse);
	glLightfv(GL_LIGHT0, GL_SPECULAR, spec);
}

void CameraInterface::beginDraw()
{

	glPushAttrib(GL_ALL_ATTRIB_BITS); 	
	glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);

    glClearColor(0.875, 0.875, 0.875, 0.0);
	glClear(GL_DEPTH_BUFFER_BIT|GL_COLOR_BUFFER_BIT);

	if (draw_axes_)
	camera_.DrawSkyView();

	glClear(GL_DEPTH_BUFFER_BIT);

	camera_.set_view();
    glPushMatrix();
    glLoadIdentity( );
    glEnable(GL_LIGHTING);
    glEnable(GL_LIGHT0);
    

    glDisable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_ALPHA_TEST);
    glEnable(GL_COLOR_MATERIAL);
	glDepthFunc(GL_LEQUAL);
	glLightModeli(GL_LIGHT_MODEL_TWO_SIDE, GL_TRUE);

	SetFrontLight();
	glEnable(GL_TEXTURE_2D);
	
	gl_color(vec3(0));



}

void CameraInterface::draw()
{

	drawAxes();	
}

void CameraInterface::endDraw()
{
    glPopAttrib();   

    glMatrixMode( GL_MODELVIEW );   
    glPopMatrix();   
    glMatrixMode( GL_PROJECTION );   
    glPopMatrix();   
}


void CameraInterface::beginDraw2D()
{
	glViewport(camera_.getViewport().GetX(), 
			   camera_.getViewport().GetY(), 
			   camera_.getViewport().GetWidth(), 
			   camera_.getViewport().GetHeight());
    glMatrixMode(GL_PROJECTION);
    glPushMatrix();
    glLoadIdentity( );

    gluOrtho2D(0, camera_.getViewport().GetWidth(), 0, camera_.getViewport().GetHeight());

    glMatrixMode(GL_MODELVIEW);
    glPushMatrix();
    glLoadIdentity( );

    glPushAttrib( GL_DEPTH_BUFFER_BIT | GL_LIGHTING_BIT );   
    glDisable( GL_DEPTH_TEST );   
    glDisable( GL_LIGHTING );   

    glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);

    glColor3f(0, 0, 0);

	glEnable(GL_TEXTURE_2D);
}

void CameraInterface::endDraw2D()
{
	glEnable(GL_TEXTURE_2D);

    glPopAttrib();   

    glMatrixMode( GL_PROJECTION );   
    glPopMatrix();   
    glMatrixMode( GL_MODELVIEW );   
    glPopMatrix(); 
}

void CameraInterface::draw2D()
{

}


void CameraInterface::resizeViewport(int w, int h)
{
	LOG("resize called %p\n", this);

	camera_.ResizeViewport(0, 0, w, h);
	camera_.need_update(true);

}



void CameraInterface::resizeViewport(int x, int y, int w, int h)
{

	camera_.ResizeViewport(x, y, w, h);
	camera_.need_update(true);


}

bool CameraInterface::processEvent(Event* ev)
{

		ev->y = camera_.height() - ev->y;

		ivec2 pos = ivec2(ev->x, ev->y);

		// convert window coordinates to viewport coordinates
		pos = camera_.getViewport().windowToViewPort(pos);

		if (ev->event_raised_by_ == kMouse_Move) {
			mouseMoveEventProc(pos,ev->get_key_state().get_shift_pressed(), ev->get_key_state().get_ctrl_pressed(), ev->get_key_state().get_alt_pressed());
			return false;
		}

		if (ev->event_raised_by_ == kMouse_Left_Button_Down) {
			mouseDownEventProc(pos,ev->get_key_state().get_shift_pressed(), ev->get_key_state().get_ctrl_pressed(), ev->get_key_state().get_alt_pressed());
			return false;
		}

		if (ev->event_raised_by_ == kMouse_Middle_Button_Down) {
			mouseMiddlemouseDownEventProc(pos,ev->get_key_state().get_shift_pressed(), ev->get_key_state().get_ctrl_pressed(), ev->get_key_state().get_alt_pressed());
			return false;
		}

		if (ev->event_raised_by_ == kMouse_Button_Up) {
			mouseUpEventProc(pos,ev->get_key_state().get_shift_pressed(), ev->get_key_state().get_ctrl_pressed(), ev->get_key_state().get_alt_pressed());
			return false;
		}

		if (ev->event_raised_by_ == kMouse_Right_Button_Down) {
			mouseRightDownEventProc(pos,ev->get_key_state().get_shift_pressed(), ev->get_key_state().get_ctrl_pressed(), ev->get_key_state().get_alt_pressed());
			return false;
		}

		if (ev->event_raised_by_ == kMouse_Left_Button_Double_Click) {
			mouseDoubleClickEventProc(pos,ev->get_key_state().get_shift_pressed(), ev->get_key_state().get_ctrl_pressed(), ev->get_key_state().get_alt_pressed());
			return false;
		}

		if (ev->event_raised_by_ == kChar) {
			characterInputEventProc(ev);
			return false;
		}
		if (ev->event_raised_by_ == kMouse_Wheel) {
			wheelEventProc(ev->wheel_delta_, pos);
			return false;
		}
		if (ev->event_raised_by_ == kKeyboard_Down) {
			keyboardDownEventProc(ev);
			return false;
		}
		if (ev->event_raised_by_ == kKeyboard_Up) {
			keyboardUpEventProc(ev);
			return false;
		}
}



void CameraInterface::keyboardDownEventProc(graphics::Event *ev)
{

}

void CameraInterface::keyboardUpEventProc(graphics::Event *ev)
{

}



void CameraInterface::characterInputEventProc(graphics::Event *ev)
{

}


void CameraInterface::wheelEventProc(int zdelta, const ivec2& pos)
{
}

void CameraInterface::mouseDoubleClickEventProc(const ivec2& p, int shift, int cont, int alt)
{

}

void CameraInterface::mouseMoveEventProc(const ivec2& p, int shift, int cont, int alt)
{

    pre_p = p;
}


void CameraInterface::mouseRightDownEventProc(const ivec2& p, int shift, int cont, int alt)
{
}

void CameraInterface::mouseDownEventProc(const ivec2& p, int shift, int cont, int alt)
{

}

void CameraInterface::mouseMiddlemouseDownEventProc(const ivec2& p, int shift, int cont, int alt)
{
}


void CameraInterface::mouseUpEventProc(const ivec2& p, int shift, int cont, int alt)
{
}


void CameraInterface::drawAxes() const
{
	if (!draw_axes_) return;

    glPushAttrib(GL_CURRENT_BIT | GL_LIGHTING_BIT);
    glDisable(GL_LIGHTING);
	glDisable(GL_TEXTURE_2D);
 
	glLineWidth(0.5);

    glDisable(GL_LINE_STIPPLE);

    glBegin(GL_LINES);
    gl_color(vec3(1.0, 0.0, 0.0));
    gl_vertex(vec3(1.0e+15, 0, 0));
    gl_vertex(vec3(0.0, 0, 0));

    gl_color(vec3(0.0, 1.0, 0.0));
    gl_vertex(vec3(0, 1.0e+15, 0));
    gl_vertex(vec3(0, 0.0, 0));

    gl_color(vec3(0.0, 0.0, 1.0));
    gl_vertex(vec3(0, 0, 1.0e+15));
    gl_vertex(vec3(0, 0, 0.0));
    glEnd();

    glEnable(GL_LINE_STIPPLE);
    glLineStipple(2, 0xAAAA);

    glBegin(GL_LINES);
    gl_color(vec3(1.0, 0.0, 0.0));
    
    gl_vertex(vec3(-1.0e+15, 0, 0));
    gl_vertex(vec3(0.0, 0, 0));
    
    gl_color(vec3(0.0, 1.0, 0.0));

    gl_vertex(vec3(0, -1.0e+15, 0));
    gl_vertex(vec3(0, 0.0, 0)); 

    gl_color(vec3(0.0, 0.0, 1.0));

    gl_vertex(vec3(0, 0, -1.0e+15));
    gl_vertex(vec3(0, 0, 0.0));
    glEnd();

    glDisable(GL_LINE_STIPPLE);


		
	glPopAttrib();
	glDisable(GL_TEXTURE_2D);
	glEnable(GL_LIGHTING);
	glDisable (GL_POLYGON_OFFSET_FILL);
}

}