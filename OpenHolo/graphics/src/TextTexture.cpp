#include "graphics/TextTexture.h"
#include "graphics/glInfo.h"
#include <string>

#include "graphics/font.h"
#include "graphics/image.h"
#include "graphics/gl_stat.h"

#include <QtGui/QImage>
#include <QtGui/QBrush>
#include <QtGui/QPen>
#include <QtGui/QPainter>
#include <QtGui/QTextLayout>
#include <QtGui/QTextBlock>
#include <QtGui/qtextdocument.h>

namespace graphics
{
	TextTexture::TextTexture() : width_(0), height_(0), texture_id_(0)
	{
	}

	TextTexture::~TextTexture()
	{
		if(texture_id_ != 0)
			glDeleteTextures(1, &texture_id_);
	}

	
	void TextTexture::Draw(int x, int y)
	{

		glEnable(GL_BLEND);
		glEnable(GL_ALPHA_TEST);
		glAlphaFunc(GL_GREATER, 0);
		glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA); 

		const int margin(6);
		const int box_width = width_ + margin;
		const int box_height = height_ + margin;
		const int fx(0), fy(0);

		glMatrixMode(GL_MODELVIEW);
		glPushMatrix();
		glLoadIdentity();
		glTranslatef(x, y, 0);
		x = 0; y = 0;
		

		// 글쓰기
		glColor4f(0,0,0,1);
		glBindTexture(GL_TEXTURE_2D, texture_id_);
		glEnable(GL_TEXTURE_2D);
		glBegin(GL_QUADS);
		glTexCoord2d(fx, !fy); glVertex2d(x, y);
		glTexCoord2d(fx, fy); glVertex2d(x, y + height_);
		glTexCoord2d(!fx, fy); glVertex2d(x + width_, y + height_);
		glTexCoord2d(!fx, !fy); glVertex2d(x + width_, y); 
		glEnd();
		glDisable(GL_TEXTURE_2D);
		glPopMatrix();

	}

	void TextTexture::Draw(int x, int y, const vec4& col)
	{

		glEnable(GL_BLEND);
		glEnable(GL_ALPHA_TEST);
		glAlphaFunc(GL_GREATER, 0);
		glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA); 

		const int margin(6);
		const int box_width = width_ + margin;
		const int box_height = height_ + margin;
		const int fx(0), fy(0);

		glMatrixMode(GL_MODELVIEW);
		glPushMatrix();
		glLoadIdentity();
		glTranslatef(x, y, 0);
		x = 0; y = 0;

		// 글상자.
		//glPushMatrix();
		//glTranslatef(-margin/2.0, -margin/2.0, 0);
		//glColor3f(1,1,1);
		//glBegin(GL_QUADS);
		//glVertex2d(x, y);
		//glVertex2d(x, y + box_height);
		//glVertex2d(x + box_width, y + box_height);
		//glVertex2d(x + box_width, y);
		//glEnd();

		//// 글상자 테두리
		//glLineWidth(1.0);
		//glColor3f(0,0,0);
		//glBegin(GL_LINE_LOOP);
		//glVertex2d(x, y);
		//glVertex2d(x, y + box_height);
		//glVertex2d(x + box_width, y + box_height);
		//glVertex2d(x + box_width, y);
		//glEnd();
		//glPopMatrix();		

		// 글쓰기
		glColor4f(col[0], col[1], col[2], col[3]);
		glBindTexture(GL_TEXTURE_2D, texture_id_);
		glEnable(GL_TEXTURE_2D);
		glBegin(GL_QUADS);
		glTexCoord2d(fx, !fy); glVertex2d(x, y);
		glTexCoord2d(fx, fy); glVertex2d(x, y + height_);
		glTexCoord2d(!fx, fy); glVertex2d(x + width_, y + height_);
		glTexCoord2d(!fx, !fy); glVertex2d(x + width_, y); 
		glEnd();
		glDisable(GL_TEXTURE_2D);
		glPopMatrix();

	}

	void TextTexture::SetText(std::wstring str, int font_size)
	{
		if(texture_id_ != 0)
		{
			glDeleteTextures(1, &texture_id_);
			texture_id_ = 0;
		}

		if(str.size() == 0) // do nothing if str is empty string.
			return;

		if(texture_id_ == 0)
			glGenTextures(1, &texture_id_);

		if(texture_id_ == 0) // check text generation fail
			return;

		glBindTexture(GL_TEXTURE_2D, texture_id_);

		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);		

		int cur_font_size = GetFontSize();
		if(font_size)
			SetFontSize(font_size);
		else 
			SetFontSize(kFONT_SIZE);

		image<uint> pix;
		TextToImage(str.c_str(), pix);

		width_ = pix.w; 
		height_ = pix.h;

		glBindTexture(GL_TEXTURE_2D, texture_id_);
		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, width_, height_, 0, GL_RGBA, GL_UNSIGNED_BYTE, pix.buf);
		SetFontSize(cur_font_size);
	}
	void TextTexture::SetText(const std::string& str, int font_size)
	{

		gl_stat stat_;

		stat_.save_stat();

		glPixelStorei(GL_UNPACK_ALIGNMENT, 4);

		glEnable(GL_TEXTURE_2D);
		if(texture_id_ != 0)
		{
			glDeleteTextures(1, &texture_id_);
			texture_id_ = 0;
		}

		if(str.size() == 0) // do nothing if str is empty string.
			return;

		if(texture_id_ == 0)
			glGenTextures(1, &texture_id_);

		if(texture_id_ == 0) // check text generation fail
			return;

		glBindTexture(GL_TEXTURE_2D, texture_id_);


		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);		

		int cur_font_size = GetFontSize();
		if(font_size)
			SetFontSize(font_size);
		else 
			SetFontSize(kFONT_SIZE);

		image<uint> pix;
		TextToImage(str.c_str(), pix);



		width_ = pix.w; 
		height_ = pix.h;

		glBindTexture(GL_TEXTURE_2D, texture_id_);
		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, width_, height_, 0, GL_RGBA, GL_UNSIGNED_BYTE, pix.buf);
		SetFontSize(cur_font_size);

		stat_.restore_stat();
	}

	void TextTexture::SetText(const std::string& str, int font_size, int label_width, int align)
	{

		gl_stat stat_;

		stat_.save_stat();

		glPixelStorei(GL_UNPACK_ALIGNMENT, 4);

		glEnable(GL_TEXTURE_2D);
		if (texture_id_ != 0)
		{
			glDeleteTextures(1, &texture_id_);
			texture_id_ = 0;
		}

		if (str.size() == 0) // do nothing if str is empty string.
			return;

		if (texture_id_ == 0)
			glGenTextures(1, &texture_id_);

		if (texture_id_ == 0) // check text generation fail
			return;

		glBindTexture(GL_TEXTURE_2D, texture_id_);


		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);


		QFont data_ft("Arial");
		data_ft.setPointSizeF(font_size);
		data_ft.setLetterSpacing(QFont::AbsoluteSpacing, 0); // kerning
		data_ft.setItalic(false);


		Qt::Alignment data_halign = (align == 0) ? Qt::AlignLeft : (align == 1)?Qt::AlignCenter:Qt::AlignRight;
		QString data_message = QString::fromStdString(str);

		QFontMetrics fm(data_ft);
		float leading = fm.leading() + 0; // 0: data leading
		float height = 0;

		QTextOption op(data_halign);
		op.setWrapMode(QTextOption::WrapAtWordBoundaryOrAnywhere);

		QTextLayout* layout;
		QRectF bounding;
		QTextDocument doc(data_message);
		std::vector<QTextLayout*> layouts;
		int total_width = 0;
		int cnt = 0;
		for (QTextBlock it = doc.begin(); it != doc.end(); it = it.next())
		{
			QString msg = it.text();

			layout = new QTextLayout(msg, data_ft);
			layouts.push_back(layout);
			layout->setTextOption(op);

			layout->beginLayout();
			QTextLine line = layout->createLine();
			while (line.isValid()) {
				line.setLineWidth(label_width);
				height += leading;
				line.setPosition(QPointF(0, height));
				height += line.height();
				total_width = max(total_width, line.width());
				line = layout->createLine();
			}
			layout->endLayout();


			bounding = bounding.united(layout->boundingRect());
		}

		QImage img(total_width, height, QImage::Format_ARGB32_Premultiplied);
		QColor black(0, 0, 0, 0);
		img.fill(black);

		QPainter painter;
		painter.begin(&img);

		painter.setPen(QColor(0, 150, 255));
		painter.setFont(data_ft);

		painter.translate(0, 0);

		for (int i = 0; i < layouts.size(); i++) {
			layouts[i]->draw(&painter, QPoint(0, 0));
			delete layouts[i];
		}

		painter.end();

		width_ = img.width();
		height_ = img.height();

		glBindTexture(GL_TEXTURE_2D, texture_id_);
		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, width_, height_, 0, GL_RGBA, GL_UNSIGNED_BYTE, img.bits());
	
		stat_.restore_stat();
	}
}