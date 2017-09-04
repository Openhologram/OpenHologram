/*
 *  special_keys.h
 *  mac_static
 *
 *  Created by Jongmin Jang on 09. 11. 16.
 *  Copyright 2009 Wolfson Lab. Inc.. All rights reserved.
 *
 */

#ifndef SPECIAL_KEYS_H_
#define SPECIAL_KEYS_H_
 
namespace graphics {
	enum SpecialKeys
	{
		kKey_None, // 이 목록에 없는 키나, 일반 키가 눌러졌을 때 설정된다.
		kKey_Tab,
		kKey_Back,
		kKey_Shift,
		kKey_Control,
		kKey_Alt,
		kKey_Capital,
		kKey_Hangul,
		kKey_PrintSrc,
		kKey_ScrollLock,
		kKey_Pause,
		kKey_Windows,
		kKey_Escape,
		kKey_ArrowUp,
		kKey_ArrowDown,
		kKey_ArrowLeft,
		kKey_ArrowRight,
		kKey_NumLock,
		kKey_Delete,
		kKey_ProcessKey,
		kKey_Return
	};
}
#endif