/*
 *  event.h
 *  mac_static
 *
 *  Created by Jongmin Jang on 09. 11. 16.
 *  Copyright 2009 Wolfson Lab. Inc.. All rights reserved.
 *
 */

#ifndef EVENT_H_
#define EVENT_H_

#ifdef _WIN32
#include <windows.h>
#endif

#include "graphics/geom.h"
#include "graphics/keystate.h"
#include "graphics/special_keys.h"

namespace graphics {

const int ks_no =		0;
const int ks_shift =		1;
const int ks_cont =		2;
const int ks_shift_cont =	3;
const int ks_alt =		4;
const int ks_shift_alt =	5;
const int ks_cont_alt =		6;
const int ks_shift_cont_alt =	7;

enum EventType 
{
	kNULL,								// 1
	kMouse_Left_Button_Down, 
	kMouse_Middle_Button_Down,			// 2
	kMouse_Right_Button_Down,			// 3
	kMouse_Button_Up, 
	kMouse_Left_Button_Double_Click,	// 5
	kMouse_Wheel,
	kMouse_Move,						// 7
	kMouse_Drag,						
	kKeyboard_Down,						// 9
	kKeyboard_Up,
	kChar,								// 11 	
	kSystem_Timer,
	kINPUTLANGCHANGE,					// 13
	kIME_STARTCOMPOSITION,
	kIME_COMPOSITION,					// 15
	kIME_ENDCOMPOSITION
};
	
struct Event {
	// WM_CHAR이외의 경우(KeyDown, KeyUp)에는 특수키가 눌러진 것으로 다룬다. 일반 키가 눌러지면 WM_CHAR가 발생하므로 그때 처리하면 된다.
	SpecialKeys key;
	// 윈도우에서는 글자가 완성되면 WM_CHAR가 발생한다. 이것이 발생하기 전에 나오는 코드들은 조합형이다.
	// 어쨋든 이것은 WM_CHAR가 발생하였을때 완성된 유니코드 글자를 돌려준다(현재 UTF-16) RegisterClass를 참고하라.
	wchar_t		character_code_;
	EventType	event_raised_by_;				// message type


	// 이벤트가 발생한 시점에서 마우스의 위치.
	int			x, y;
	int			wheel_delta_;
		
	Event(EventType m = kNULL);
		
	graphics::vec2 position(void) const;
	void set_key_state(bool shift_pressed, bool alt_pressed, bool ctrl_pressed);
	KeyState get_key_state(void) const;
private:
	KeyState key_state_;
};

int MakeKeyState(int s, int c, int a);

}
#endif