#ifndef __gaussian_pyramid_h
#define __gaussian_pyramid_h

#include "graphics/sys.h"
#include "graphics/gl.h"
#include "graphics/gl_extension.h"
#include "graphics/unsigned.h"


namespace graphics {

struct  {
  int ncols; // width
  int nrows; // height
  float *data;
}  FloatImageRec, *FloatImage;

class GaussianPyr {
public:

	GaussianPyr();


protected:

	FloatImage input_;
	float sigma_;
	float deriv_sigma_;
	int n_levels_;
	std::vector<FloatImage> output_;
	std::vector<FloatImage> output_deriv_x_;
	std::vector<FloatImage> output_deriv_y_;
};

};
#endif
