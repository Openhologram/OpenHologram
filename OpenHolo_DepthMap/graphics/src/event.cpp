/*
 *  event.cpp
 *  mac_static
 *
 *  Created by Jongmin Jang on 09. 11. 13.
 *  Copyright 2009 Wolfson Lab. Inc.. All rights reserved.
 *
 */


#include "graphics/event.h"
#include "graphics/geom.h"
#include "graphics/keystate.h"
#include "graphics/special_keys.h"

namespace graphics {



Event::Event(EventType m) :
key(kKey_None)
{
	event_raised_by_ = m; 
}
	
graphics::vec2 Event::position(void) const 
{
	return graphics::vec2(x, y);
}
	
void Event::set_key_state(bool shift_pressed, bool alt_pressed, bool ctrl_pressed)
{
	key_state_.set_shift_pressed(shift_pressed);
	key_state_.set_alt_pressed(alt_pressed);
	key_state_.set_ctrl_pressed(ctrl_pressed);
}
	
KeyState Event::get_key_state(void) const
{
	return key_state_;
}

int MakeKeyState(int s, int c, int a)
{
	return (s ? 1 : 0) + (c ? 2 : 0) + (a ? 4 : 0);
}

}