#ifndef TEXT_TEXTURE_H_
#define TEXT_TEXTURE_H_


#include <string>

#include <graphics/geom.h>

class QPainter;

namespace graphics
{
	class TextTexture
	{
	public:
		TextTexture(void);
		~TextTexture(void);
		void Draw(int x, int y);
		inline void Draw(const graphics::vec2& v)
		{
			Draw(v.v[0], v.v[1]);
		}

		void Draw(int x, int y, const vec4& color);
		inline void Draw(const graphics::vec2& v, const vec4& color)
		{
			Draw(v.v[0], v.v[1], color);
		}

		void SetText(const std::string&, int text_size);
		void SetText(std::wstring, int text_size);

		// 0: left, 1: center, 2: right
		void SetText(const std::string&, int text_size, int label_width, int align = 0);

		int GetHeight(void) const { return height_; }
		int GetWidth(void) const { return width_; }


		real aspect_ratio() const { return (real)width_/(real)height_; }

	public:

		int width_;
		int height_;
		GLuint texture_id_;
		QPainter* painter;
	};
}

#endif