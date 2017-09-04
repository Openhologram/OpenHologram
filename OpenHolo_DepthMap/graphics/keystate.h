/*
 *  keystate.h
 *  mac_static
 *
 *  Created by Jongmin Jang on 09. 11. 16.
 *  Copyright 2009 Wolfson Lab. Inc.. All rights reserved.
 *
 */

#ifndef KEYSTATE_H_
#define KEYSTATE_H_

namespace graphics {
	struct KeyState
	{
	public:
		KeyState(bool shiftPressed = false, bool altPressed = false, bool ctrlPressed = false);
		void set_shift_pressed(bool);
		void set_alt_pressed(bool);
		void set_ctrl_pressed(bool);
		
		bool get_shift_pressed(void) const;
		bool get_alt_pressed(void) const;
		bool get_ctrl_pressed(void) const;
	private:
		bool shift_pressed_;
		bool alt_pressed_;
		bool ctrl_pressed_;
	};
	
}

#endif