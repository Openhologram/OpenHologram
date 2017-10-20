#ifndef  __VIEWPORT_H
#define  __VIEWPORT_H
#include "graphics/sys.h"
#include "graphics/real.h"

namespace graphics {

class ViewPort {

public:

	ViewPort(int x, int y, int w, int h)
		: position_(x,y), width_(w), height_(h)
	{
	}

	ViewPort(int w, int h)
		: position_(0), width_(w), height_(h)
	{
	}

	ViewPort(const ViewPort& vp)
		: position_(vp.position_), width_(vp.width_), height_(vp.height_)
	{
	}

	ViewPort()
	{
	}

	ivec2 windowToViewPort(const ivec2& xy) const
	{
		return xy - position_;
	}

	// return local coordinate of the center of the view port
	ivec2 GetCenter() const
	{
		return ivec2(width_/2, height_/2);
	}

	ivec2 GetPosition() const
	{
		return position_;
	}

	int  GetWidth() const 
	{
		return width_;
	}

	int GetHeight() const
	{
		return height_;
	}

	int GetX() const
	{
		return position_[0];
	}


	int GetY() const
	{
		return position_[1];
	}


	real GetAspectRatio() const
	{
		return (real)width_/(real)height_;
	}

	void SetPosition(const ivec2& pos)
	{
		position_ = pos;
	}

	void Resize(int w, int h)
	{
		width_ = w;
		height_ = h;
	}

private:
	
	ivec2	position_;

	int		width_;
	int		height_;
};

};

#endif