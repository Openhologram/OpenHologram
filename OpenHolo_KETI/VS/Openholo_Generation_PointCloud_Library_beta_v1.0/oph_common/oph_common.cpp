#include "oph_common.h"


#define OPH_Bitsperpixel 8 //24 // 3byte=24 
#define OPH_Planes 1
#define OPH_Compression 0
#define OPH_Xpixelpermeter 0x130B //2835 , 72 DPI
#define OPH_Ypixelpermeter 0x130B //2835 , 72 DPI
#define OPH_Pixel 0xFF


#pragma pack(push,1)
typedef struct fileheader {
	uint8_t signature[2];
	uint32_t filesize;
	uint32_t reserved;
	uint32_t fileoffset_to_pixelarray;
} fileheader;

typedef struct bitmapinfoheader {
	uint32_t dibheadersize;
	uint32_t width;
	uint32_t height;
	uint16_t planes;
	uint16_t bitsperpixel;
	uint32_t compression;
	uint32_t imagesize;
	uint32_t ypixelpermeter;
	uint32_t xpixelpermeter;
	uint32_t numcolorspallette;
	uint32_t mostimpcolor;
} bitmapinfoheader;

typedef struct rgbquad {
	uint8_t rgbBlue;
	uint8_t rgbGreen;
	uint8_t rgbRed;
	uint8_t rgbReserved;
} rgbquad;

typedef struct bitmap {
	fileheader fileheader;
	bitmapinfoheader bitmapinfoheader;
	rgbquad rgbquad[256]; // 8 bit 256 Color(Grayscale)
} bitmap;
#pragma pack(pop)


void openholo::ophSaveFileBmp(unsigned char* pixelbuffer, int pic_width, int pic_height, char* file_name) {
	int _height = pic_height;
	int _width = pic_width;
	int _pixelbytesize = _height*_width*OPH_Bitsperpixel / 8;
	int _filesize = _pixelbytesize + sizeof(bitmap);

	char bmpFile[256];
	sprintf(bmpFile, "%s.bmp", file_name);
	FILE *fp = fopen(bmpFile, "wb");
	bitmap *pbitmap = (bitmap*)calloc(1, sizeof(bitmap));
	memset(pbitmap, 0x00, sizeof(bitmap));

	// File Header
	pbitmap->fileheader.signature[0] = 'B';
	pbitmap->fileheader.signature[1] = 'M';
	pbitmap->fileheader.filesize = _filesize;
	pbitmap->fileheader.fileoffset_to_pixelarray = sizeof(bitmap);

	// Initialize pallets : to Grayscale
	for (int i = 0; i < 256; i++) {
		pbitmap->rgbquad[i].rgbBlue = i;
		pbitmap->rgbquad[i].rgbGreen = i;
		pbitmap->rgbquad[i].rgbRed = i;
	}

	// Image Header
	pbitmap->bitmapinfoheader.dibheadersize = sizeof(bitmapinfoheader);
	pbitmap->bitmapinfoheader.width = _width;
	pbitmap->bitmapinfoheader.height = _height;
	pbitmap->bitmapinfoheader.planes = OPH_Planes;
	pbitmap->bitmapinfoheader.bitsperpixel = OPH_Bitsperpixel;
	pbitmap->bitmapinfoheader.compression = OPH_Compression;
	pbitmap->bitmapinfoheader.imagesize = _pixelbytesize;
	pbitmap->bitmapinfoheader.ypixelpermeter = OPH_Ypixelpermeter;
	pbitmap->bitmapinfoheader.xpixelpermeter = OPH_Xpixelpermeter;
	pbitmap->bitmapinfoheader.numcolorspallette = 256;
	fwrite(pbitmap, 1, sizeof(bitmap), fp);
	fwrite(pixelbuffer, 1, _pixelbytesize, fp);
	fclose(fp);
	free(pbitmap);
}


void openholo::ophNormalize(float *src, unsigned char *dst, const int nx, const int ny) {
	float minVal, maxVal;
	for (int ydx = 0; ydx < ny; ydx++) {
		for (int xdx = 0; xdx < nx; xdx++) {
			float *temp_pos = src + xdx + ydx*nx;
			if ((xdx == 0) && (ydx == 0)) {
				minVal = *(temp_pos);
				maxVal = *(temp_pos);
			}
			else {
				if (*(temp_pos) < minVal) minVal = *(temp_pos);
				if (*(temp_pos) > maxVal) maxVal = *(temp_pos);
			}
		}
	}

	for (int ydx = 0; ydx < ny; ydx++) {
		for (int xdx = 0; xdx < nx; xdx++) {
			float *src_pos = src + xdx + ydx*nx;
			unsigned char *res_pos = dst + xdx + (ny - ydx - 1)*nx;	// Flip image vertically to consider flipping by Fourier transform and projection geometry

			*(res_pos) = (unsigned char)(((*(src_pos)-minVal) / (maxVal - minVal)) * 255 + 0.5);
		}
	}
}