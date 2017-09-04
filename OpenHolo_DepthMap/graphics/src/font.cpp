#include <iostream>
#include <ft2build.h>
#include FT_FREETYPE_H
#include FT_GLYPH_H



#include "graphics/font.h"
#include <stdio.h>
#include <string>
#include "graphics/sys.h"

namespace graphics
{
	const int kMaxGlyphs = 200;
/*
#ifdef _WIN32
	char* font_dir = "c:/windows/fonts";
#else
	char* font_dir = "macclient/Resources";
#endif
*/
	const int f_load = FT_LOAD_NO_BITMAP;

	FT_Library library;
	FT_Face face;
	static char cur_name[80];
	static int cur_size;

	// 글자의 실제 픽셀 높이.
	extern int text_height = 0;
	static int fm_height;
	static int fm_ascender;
	static int fm_descender;

	static void DrawBits(image<uint>& pix, int px, int py, FT_Bitmap* bit)
	{
		for(int y = 0; y < bit->rows; y++)
			for(int x = 0; x < bit->width; x++) 
			{
				pix(px + x, py + y) = (((unsigned int)(bit->buffer[x + bit->width * y]))<<24) | ((unsigned int)(255<<16)) | ((unsigned int)(255<<8)) | ((unsigned int)(255)) ;
			}
	
	}

	static void DrawBits(image<uchar>& pix, int px, int py, FT_Bitmap* bit)
	{
		for(int y = 0; y < bit->rows; y++)
			for(int x = 0; x < bit->width; x++) 
			{
				pix(px + x, py + y) = (((uchar)(bit->buffer[x + bit->width * y]))<<24) ;
			}
	}
	void InitializeFreeType(const char* fullpath)
	{
		int error = FT_Init_FreeType( &library );
		if ( error ) {
			printf("freetype init error\n");
		}

		SetFontFace(fullpath);
		SetFontSize(kFONT_SIZE);
	}

	void SetFontFace(const char* fullpath)
	{
		char font_path[100];

		if(strcmp(fullpath, cur_name) == 0) 
			return;
/*
#ifdef _WIN32
		strcpy_s(cur_name, name);
		sprintf_s(font_path, "%s/%s", font_dir, name);
#else
		strcpy(cur_name, name);
		sprintf(font_path, "%s/%s", font_dir, name);
#endif
*/
		int error = FT_New_Face( library, fullpath, 0, &face);
		
		if( error == FT_Err_Unknown_File_Format ) 
		{
			printf("font format is unsupported\n");
			std::cout << "trying open font " << fullpath << std::endl;;
		}
		else if ( error ) 
		{
			printf("font could not opened\n");
			std::cout << "trying open font " << fullpath << std::endl;;
		}

		cur_size = 0;

		printf("%s: # of glyph : %d\n", fullpath, face->num_glyphs);
	}

	void SetFontSize(int size)
	{
		if(size == cur_size) 
			return;

		cur_size = size;

		int error = FT_Set_Pixel_Sizes(face, 0, size);

		fm_ascender = face->size->metrics.ascender >> 6;
		fm_descender = face->size->metrics.descender >> 6;
		fm_height = face->size->metrics.height >> 6;	

		// 글자의 실제 크기를 저장해 둔다.
		text_height = fm_height - fm_descender;
	}
	
	int GetFontSize(void)
	{
		return cur_size;
	}


	void TextToImage(int px, int py, const char* text, image<uint>& pix)
	{
		FT_GlyphSlot  slot = face->glyph;

		for(int i = 0; text[i];++i) 
		{
			int error = FT_Load_Char(face, text[i], f_load | FT_LOAD_RENDER);
			if (error == 0) // success.
			{
				continue;
			}
			else
			{
				std::cerr << "error while executing TextToImage. source text was: " << text << std::endl;
			}

			DrawBits(pix, px + slot->bitmap_left, py - slot->bitmap_top, &slot->bitmap);

			px += slot->advance.x >> 6;
			printf("adv.x(%c) %d %g\n", text[i], px, (float) slot->advance.x / 64);
		}
	}

	void TextToImage(const char* text, image<uint>& pix)
	{
		glPixelStorei(GL_UNPACK_ALIGNMENT, 4);
		wchar_t s[kMaxGlyphs];

		//std::string str(text);
		int n_text = strlen(text); //str.size();
		//std::cout << "TextToImage for \"" << text << "\" has " << n_text << " chars" << std::endl << std::flush;
		//printf("text %s has %d number of char\n", text, n_text);

		for(int n = 0; n <= n_text; n++) 
			s[n] = text[n];

		TextToImage(s, pix);
	}

	void TextToImage(const wchar_t* text, image<uint>& pix)
	{
		glPixelStorei(GL_UNPACK_ALIGNMENT, 4);
		FT_GlyphSlot slot = face->glyph;   
		FT_Glyph glyphs[kMaxGlyphs];  
		FT_Vector pos[kMaxGlyphs]; 

#ifdef _WIN32
		int num_chars = wcslen(text);
#else
		std::wstring str(text);
		int num_chars = str.size();
#endif

		int pen_x = 0; 

		FT_UInt num_glyphs  = 0;
		FT_Bool use_kerning = FT_HAS_KERNING( face );
		FT_UInt previous    = 0;

		int error;

		for(int n = 0; n < num_chars; n++) {
			FT_UInt glyph_index = FT_Get_Char_Index( face, text[n] );

			if ( use_kerning && previous && glyph_index ) {
				FT_Vector delta;
				FT_Get_Kerning( face, previous, glyph_index,
					FT_KERNING_DEFAULT, &delta );

				pen_x += delta.x >> 6;
			}

			pos[num_glyphs].x = pen_x;

			error = FT_Load_Glyph( face, glyph_index, f_load | FT_LOAD_DEFAULT);
			if ( error ) continue;

			error = FT_Get_Glyph( face->glyph, &glyphs[num_glyphs] );
			if ( error ) continue;  

			pen_x += slot->advance.x >> 6;
			previous = glyph_index;

			num_glyphs++;
		}

	
		pix.alloc(pen_x, fm_height);

		for(unsigned int n = 0; n < num_glyphs; n++) {
			FT_Glyph image = glyphs[n];

			FT_Vector  pen;
			pen.x = pos[n].x * 64;
			pen.y = 0;

			error = FT_Glyph_To_Bitmap(&image, FT_RENDER_MODE_NORMAL, &pen, 0);
			if ( !error ) {
				FT_BitmapGlyph  bit = (FT_BitmapGlyph)image;

				DrawBits(pix, bit->left, fm_ascender - bit->top, &bit->bitmap); 
				// printf("draw %d %d\n", bit->left, fm_ascender - bit->top);

				
			}
			FT_Done_Glyph( image );
		}


	}


		void TextToImage(const char* text, image<uchar>& pix)
	{
		glPixelStorei(GL_UNPACK_ALIGNMENT, 4);
		wchar_t s[kMaxGlyphs];

		//std::string str(text);
		int n_text = strlen(text); //str.size();
		//std::cout << "TextToImage for \"" << text << "\" has " << n_text << " chars" << std::endl << std::flush;
		//printf("text %s has %d number of char\n", text, n_text);

		for(int n = 0; n <= n_text; n++) 
			s[n] = text[n];

		TextToImage(s, pix);
	}

	void TextToImage(const wchar_t* text, image<uchar>& pix)
	{
		glPixelStorei(GL_UNPACK_ALIGNMENT, 4);
		FT_GlyphSlot slot = face->glyph;   
		FT_Glyph glyphs[kMaxGlyphs];  
		FT_Vector pos[kMaxGlyphs]; 

#ifdef WIN32
		int num_chars = wcslen(text);
#else
		std::wstring str(text);
		int num_chars = str.size();
#endif

		int pen_x = 0; 

		FT_UInt num_glyphs  = 0;
		FT_Bool use_kerning = FT_HAS_KERNING( face );
		FT_UInt previous    = 0;

		int error;

		for(int n = 0; n < num_chars; n++) {
			FT_UInt glyph_index = FT_Get_Char_Index( face, text[n] );

			if ( use_kerning && previous && glyph_index ) {
				FT_Vector delta;
				FT_Get_Kerning( face, previous, glyph_index,
					FT_KERNING_DEFAULT, &delta );

				pen_x += delta.x >> 6;
			}

			pos[num_glyphs].x = pen_x;

			error = FT_Load_Glyph( face, glyph_index, f_load | FT_LOAD_DEFAULT);
			if ( error ) continue;

			error = FT_Get_Glyph( face->glyph, &glyphs[num_glyphs] );
			if ( error ) continue;  

			pen_x += slot->advance.x >> 6;
			previous = glyph_index;

			num_glyphs++;
		}

		pix.alloc(pen_x, fm_height);

		for(unsigned int n = 0; n < num_glyphs; n++) {
			FT_Glyph image = glyphs[n];

			FT_Vector  pen;
			pen.x = pos[n].x * 64;
			pen.y = 0;

			error = FT_Glyph_To_Bitmap(&image, FT_RENDER_MODE_NORMAL, &pen, 0);
			if ( !error ) {
				FT_BitmapGlyph  bit = (FT_BitmapGlyph)image;

				DrawBits(pix, bit->left, fm_ascender - bit->top, &bit->bitmap); 
				// printf("draw %d %d\n", bit->left, fm_ascender - bit->top);

				FT_Done_Glyph( image );
			}
		}
	}
}


/*
void ft_text_subpixel(char* text)
{
const int MAX_GLYPHS = 100;

FT_GlyphSlot  slot = face->glyph;

FT_Glyph      glyphs[MAX_GLYPHS];
FT_Vector     pos   [MAX_GLYPHS]; 

int num_chars = strlen(text);

int pen_x = 0; 
int pen_y = 0;

FT_UInt num_glyphs  = 0;
FT_Bool use_kerning = FT_HAS_KERNING( face );
FT_UInt previous    = 0;

int error;

FT_BBox  bb;
bb.xMin = bb.yMin =  32000 * 64;	// max value
bb.xMax = bb.yMax = -32000 * 64;	// min value

// Get Glyphs & bbox
for (int n = 0; n < num_chars; n++ ) {
FT_UInt glyph_index = FT_Get_Char_Index( face, text[n] );

if ( use_kerning && previous && glyph_index ) {
FT_Vector  delta;
FT_Get_Kerning( face, previous, glyph_index,
FT_KERNING_DEFAULT, &delta );

pen_x += delta.x;
}

pos[num_glyphs].x = pen_x;
pos[num_glyphs].y = pen_y;

error = FT_Load_Glyph( face, glyph_index, FT_LOAD_DEFAULT );
if ( error ) continue;

error = FT_Get_Glyph( face->glyph, &glyphs[num_glyphs] );
if ( error ) continue;  

FT_BBox  gbb;
FT_Glyph_Get_CBox( glyphs[n], ft_glyph_bbox_subpixels, &gbb );

printf("bbox %g %g %g %g\n",
(float) gbb.xMin / 64,
(float) gbb.xMax / 64,
(float) gbb.yMin / 64,
(float) gbb.yMax / 64);

if(gbb.xMin + pen_x < bb.xMin) bb.xMin = gbb.xMin + pen_x;
if(gbb.yMin + pen_y < bb.yMin) bb.yMin = gbb.yMin + pen_y;
if(gbb.xMax + pen_x > bb.xMax) bb.xMax = gbb.xMax + pen_x;
if(gbb.yMax + pen_y > bb.yMax) bb.yMax = gbb.yMax + pen_y;

pen_x += slot->advance.x;
printf("pen_x %g %g\n",
(float) pen_x / 64, (float) slot->advance.x / 64);

previous = glyph_index;

num_glyphs++;
}
printf("bbox %g %g %g %g\n",
(float) bb.xMin / 64,
(float) bb.xMax / 64,
(float) bb.yMin / 64,
(float) bb.yMax / 64);

int width = bb.xMax - bb.xMin;
printf("width %g\n", (float) width / 64);

for (int n = 0; n < num_glyphs; n++ ) {
FT_Glyph image = glyphs[n];

FT_Vector  pen;
pen.x = pos[n].x;
pen.y = pos[n].y;

error = FT_Glyph_To_Bitmap(&image, FT_RENDER_MODE_NORMAL, &pen, 0);
if ( !error ) {
FT_BitmapGlyph  bit = (FT_BitmapGlyph)image;

// my_draw_bitmap( bit->bitmap, 
// bit->left, my_target_height - bit->top );
printf("draw %d %d\n", bit->left, bit->top);

FT_Done_Glyph( image );
}
}
}
*/
