// color_buffer.cpp: implementation of the color_buffer class.
//
//////////////////////////////////////////////////////////////////////

#include "graphics/color_buffer.h"
#include <set>
#include "graphics/geom.h"
#include "graphics/gl_extension.h"
#include "graphics/render_buffer.h"


namespace graphics {

//////////////////////////////////////////////////////////////////////
// Construction/Destruction
//////////////////////////////////////////////////////////////////////
color_buffer::color_buffer() 
: cache(true), render_buffer_(0), buffer(0), pos_x(0), pos_y(0), width(0), height(0), max_width(0), max_height(0), initialized_(false)
{	
    width = max_width;
    height = max_height;
}

color_buffer::~color_buffer()
{
	clearAll();
}



void
color_buffer::initialize()
{
	if (initialized_) return;
	initialized_ = true;
	render_buffer_ = new RenderBuffer();
	render_buffer_->GenerateObjects();
	render_buffer_->Resize(width, height);
}

void
color_buffer::clearAll()
{
	initialized_ = false;
    if (buffer) delete [] buffer;
	if (render_buffer_) delete render_buffer_;
	buffer = 0;
	render_buffer_ = 0;
}

void color_buffer::clearOglBuffer()
{
	if (render_buffer_) delete render_buffer_;
	render_buffer_ = 0;
}

void
color_buffer::resize(int w, int h)
{
    if (w < 0 || h < 0) return;
    
    if (((w * h) >= max_width * max_height)) {
		if (buffer) delete buffer; 
		buffer = new GLuint[w * h];
    }

    width = w;
    height = h;

	pos_x = 0;
	pos_y = 0;

	

	if (!initialized_) initialize();
	render_buffer_->Resize(width, height);

}
 
void
color_buffer::resize(int x, int y, int w, int h)
{
    if (w < 0 || h < 0) return;
    
	pos_x = x;
	pos_y = y;

    if (((w * h) >= max_width * max_height)) {
		delete buffer; 
		buffer = new GLuint[w * h];
    }

    width = w;
    height = h;

	if (!initialized_) initialize();
	render_buffer_->Resize(width, height);
}

void
color_buffer::get_objects(int x, int y, int w, int h, std::set<uint> &a_set) const
{
    a_set.clear();

    int sy = ((y-h) > 0) ? (y-h) : 0;
    int ey = ((y+h) >= height) ? height-1 : (y+h);
    int sx = ((x-w) > 0) ? (x-w) : 0;
    int ex = ((x+w) >= width) ? width-1 : (x+w);

    for (int i = sy ; i <= ey ;++i) {
	for (int j = sx ; j <= ex ; ++j) {
	    uint val = read(j, i);
	    if (val) a_set.insert(val);
	}
    }
}

void color_buffer::get_objects(int x, int y, std::set<uint>& a_set) const
{
	a_set.clear();
	uint val = read(x, y);
	if (val) a_set.insert(val);
}


void
color_buffer::get_objects(ivec2 c1, ivec2 c2, std::set<uint> &a_set) const
{
    a_set.clear();

	box2 bo;

	bo.extend(vec2(c1));
	bo.extend(vec2(c2));

	c1 = ivec2(bo.minimum[0], bo.minimum[1]);
	c2 = ivec2(bo.maximum[0], bo.maximum[1]);

    for (int i = c1[1] ; i <= c2[1] ;++i) {
	for (int j = c1[0] ; j <= c2[0] ; ++j) {
	    uint val = read(j, i);
	    if (val) a_set.insert(val);
	}
    }
}

void
color_buffer::read_in()
{
	render_buffer_->ReadData(buffer);
}


void
color_buffer::begin_draw() const
{
	gl_stat_.save_stat();

	render_buffer_->BeginDraw();
	glPushAttrib(GL_ALL_ATTRIB_BITS);
	glShadeModel(GL_FLAT); 
	glDisable(GL_TEXTURE_2D);
	//LOG("color buffer update happend\n");
	glDisable(GL_POLYGON_SMOOTH);
	glDisable(GL_POINT_SMOOTH);
    glDisable(GL_LINE_SMOOTH);
    glDisable(GL_BLEND);
	glEnable(GL_DEPTH_TEST);
    glDisable(GL_ALPHA_TEST);
    //glDisable(GL_COLOR_MATERIAL);
    glDisable(GL_DITHER);
	glDisable(GL_MULTISAMPLE);
    glDepthFunc(GL_LEQUAL);

    glDisable(GL_LIGHTING);
    glClearColor(0.0, 0.0, 0.0, 0.0);
	//glGetIntegerv(GL_DRAW_BUFFER, &kCurrentBuffer);
    glDrawBuffer(GL_BACK);
    glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT);

    /*glMatrixMode(GL_MODELVIEW);
    glPushMatrix();
	glLoadIdentity();*/
}

void
color_buffer::end_draw()
{
    //glPopMatrix();
	glPopAttrib();
	glDepthFunc(GL_LEQUAL);
	read_in();
	render_buffer_->EndDraw();

	gl_stat_.restore_stat();
}

}; // namespace graphics