/**
* @mainpage OpenHolo Project Common Library
* @brief Common Function
* @details Common function in OpenHolo
*/

#ifndef OPH_COMMON_LIB_H
#define OPH_COMMON_LIB_H

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <memory.h>
	

namespace openholo {

	/**
	\defgroup Common_Modules
	* @{
	*/
	/**
	* @brief Save Pixel Buffer to Bitmap File format Image.
	* @param pixelbuffer: Image data pointer to save
	* @param pic_width: Width of image
	* @param pic_height: Height of image
	* @param file_name: filename to save
	*/
	void ophSaveFileBmp(unsigned char* pixelbuffer, int pic_width, int pic_height, char* file_name);

	/**
	* @brief normalize calculated fringe pattern to 8bit grayscale value.
	* @param src: Input float type pointer
	* @param dst: Output char tpye pointer
	* @param nx: The number of pixels in X
	* @param ny: The number of pixels in Y
	*/
	void ophNormalize(float *src, unsigned char *dst, const int nx, const int ny);
	/** @} */
}

#endif