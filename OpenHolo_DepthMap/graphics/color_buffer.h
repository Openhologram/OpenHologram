// color_buffer.h: interface for the color_buffer class.
//
//////////////////////////////////////////////////////////////////////

#ifndef	    __color_buffer_h
#define	    __color_buffer_h

#include "graphics/sys.h"
#include "graphics/unsigned.h"
#include "graphics/vector.h"
#include <set>
#include "graphics/ivec.h"
#include "graphics/gl_stat.h"



namespace graphics {

class RenderBuffer;

class color_buffer  
{
protected:
    unsigned int*   buffer;

	RenderBuffer* render_buffer_;

	//GLuint pbo_id_;

private:
	int		pos_x, pos_y;

    int	    width, height;
    int	    max_width, max_height;
    bool    cache;

	bool	initialized_;
	
	mutable gl_stat gl_stat_;


public:
    color_buffer();
    virtual ~color_buffer();


public:

	void initialize();

	void clearAll();

	void clearOglBuffer();

	int get_width() { return width; }

	int get_height() { return height; }

    virtual void resize(int w, int h);

	virtual void resize(int x, int y, int w, int h);

	virtual void get_objects(int x, int y, int w, int h, std::set<uint> &ret) const;

	virtual void get_objects(int x, int y, std::set<uint>& ret) const;

	virtual void get_objects(ivec2 x, ivec2 y, std::set<uint> &ret) const;

    virtual void begin_draw() const;

    virtual void end_draw();

	inline uint* get_buffer() const { return buffer; }



	inline uint read(int x, int y) const;

protected:



    

    virtual void read_in();


    inline int	get_max_width() const { return max_width; }

    inline int  get_max_height() const { return max_height; }

	inline void new_buffer(int s) { delete [] buffer; buffer = new GLuint[s]; }
};



inline uint color_buffer::read(int x, int y) const
{
	if (x < 0 || x > width || y < 0 || y > height) return 0;
    #ifdef GL_RADEON
	return buffer[(y%height)*width + (x%width)];
    #else
	uint ret = buffer[(y%height)*width + (x%width)];
	return (((ret&0xFF000000)>>24)|((ret&0x00FF0000)>>8)|((ret&0x0000FF00)<<8)|((ret&0x000000FF)<<24));
    #endif
}

}; // namespace graphics

#endif 
