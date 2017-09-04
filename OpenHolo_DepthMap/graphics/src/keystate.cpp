/*
 *  keystate.cpp
 *  mac_static
 *
 *  Created by Jongmin Jang on 09. 11. 13.
 *  Copyright 2009 Wolfson Lab. Inc.. All rights reserved.
 *
 */


#include "graphics/keystate.h"

namespace graphics {
	
	KeyState::KeyState(bool shiftPressed, bool altPressed, bool ctrlPressed) :
	shift_pressed_(shiftPressed),
	alt_pressed_(altPressed),
	ctrl_pressed_(ctrlPressed)
	{
	}
	
	void KeyState::set_shift_pressed(bool value)
	{
		shift_pressed_ = value;
	}
	void KeyState::set_alt_pressed(bool value)
	{
		alt_pressed_ = value;
	}
	void KeyState::set_ctrl_pressed(bool value)
	{
		ctrl_pressed_ = value;
	}
	
	bool KeyState::get_shift_pressed(void) const
	{
		return shift_pressed_;
	}
	
	bool KeyState::get_alt_pressed(void) const
	{
		return alt_pressed_;
	}
	
	bool KeyState::get_ctrl_pressed(void) const
	{
		return ctrl_pressed_;
	}
	
}