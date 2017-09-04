#ifndef FONT_H_
#define FONT_H_ 

#include "graphics/image.h"
#include "graphics/unsigned.h"

namespace graphics
{
const int	kFONT_SIZE = 16;
//const char* const kFONT_NAME = "malgun.ttf";
// 폰트크기가 설정되고 나면 (FONT_SIZE가 실제 폰트크기 아님)
// 이 변수에 폰트의 실제 픽셀 높이가 저장된다.
extern int text_height;

void InitializeFreeType(const char* fullpath);

void SetFontFace(const char* name);

void SetFontSize(int size);
	
int GetFontSize(void);

void TextToImage(int px, int py, const char* text, image<uint>& pix);

void TextToImage(const char* text, image<uint>& pix);

void TextToImage(const wchar_t* text, image<uint>& pix);
void TextToImage(const char* text, image<uchar>& pix);

void TextToImage(const wchar_t* text, image<uchar>& pix);
}
#endif //FONT_H_
