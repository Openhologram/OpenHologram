#ifndef __GL_STAT_H
#define __GL_STAT_H

#include "graphics/sys.h"

namespace graphics {


class gl_stat {

public:

	gl_stat();


	void save_stat();

	void restore_stat();
private:

	GLboolean texture_2d_;
	GLboolean texture_1d_;
	GLboolean multisample_;
	GLboolean polygon_smooth_;
	GLboolean line_smooth_;
	GLboolean depth_test_;
	GLboolean blend_;
	GLboolean alpha_test_;
	GLboolean point_smooth_;
	GLboolean lighting_;
	GLboolean line_stipple_;
	GLboolean polygon_stipple_;
	GLboolean polygon_offset_;
	GLboolean color_material_;

	int blend_src_;
	int blend_dst_;

	int	alpha_func_;
	int	alpha_ref_;

	int  matrix_mode_;

	int  unpack_alignment_;
	int  pack_alignment_;

	float projection_matrix_[16];
	float model_matrix_[16];
	float line_width_;
	float point_size_;
	int vport_[4];

};

};
#endif