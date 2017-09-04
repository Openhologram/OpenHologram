#include "graphics/gl_stat.h"
#include "GL/glext.h"

namespace graphics {

gl_stat::gl_stat()
{

}

void gl_stat::save_stat()
{
	glGetBooleanv(GL_TEXTURE_2D, &texture_2d_);
	glGetBooleanv(GL_TEXTURE_1D, &texture_1d_);
	glGetBooleanv(GL_MULTISAMPLE, &multisample_);
	glGetBooleanv(GL_POLYGON_SMOOTH, &polygon_smooth_);

	glGetBooleanv(GL_LINE_SMOOTH, &line_smooth_);
	glGetBooleanv(GL_DEPTH_TEST, &depth_test_);
	glGetBooleanv(GL_BLEND, &blend_);
	glGetBooleanv(GL_ALPHA_TEST, &alpha_test_);
	glGetBooleanv(GL_POINT_SMOOTH, &point_smooth_);
	glGetBooleanv(GL_LIGHTING, &lighting_);
	glGetBooleanv(GL_LINE_STIPPLE, &line_stipple_);
	glGetBooleanv(GL_POLYGON_STIPPLE, &polygon_stipple_);
	glGetBooleanv(GL_POLYGON_OFFSET_FILL, &polygon_offset_);
	glGetBooleanv(GL_COLOR_MATERIAL, &color_material_);
	
	glGetIntegerv(GL_ALPHA_TEST_FUNC, &alpha_func_);
	glGetIntegerv(GL_ALPHA_TEST_REF, &alpha_ref_);

	glGetIntegerv(GL_BLEND_DST, &blend_dst_);
	glGetIntegerv(GL_BLEND_SRC, &blend_src_);

	glGetIntegerv(GL_VIEWPORT, vport_);
	glGetIntegerv(GL_UNPACK_ALIGNMENT, &unpack_alignment_);
	glGetIntegerv(GL_PACK_ALIGNMENT, &pack_alignment_);
	glGetFloatv(GL_PROJECTION_MATRIX, projection_matrix_);
	glGetFloatv(GL_MODELVIEW_MATRIX, model_matrix_);
	glGetFloatv(GL_LINE_WIDTH, &line_width_);
	glGetFloatv(GL_POINT_SIZE, &point_size_);
	glGetIntegerv(GL_MATRIX_MODE, &matrix_mode_);
}

void gl_stat::restore_stat()
{
	if (texture_2d_) glEnable(GL_TEXTURE_2D);
	else glDisable(GL_TEXTURE_2D);

	if (multisample_) glEnable(GL_MULTISAMPLE);
	else glDisable(GL_MULTISAMPLE);	

	if (texture_1d_) glEnable(GL_TEXTURE_1D);
	else glDisable(GL_TEXTURE_1D);	

	if (polygon_smooth_) glEnable(GL_POLYGON_SMOOTH);
	else glDisable(GL_POLYGON_SMOOTH);

	if (line_smooth_) glEnable(GL_LINE_SMOOTH);
	else glDisable(GL_LINE_SMOOTH);	

	if (depth_test_) glEnable(GL_DEPTH_TEST);
	else glDisable(GL_DEPTH_TEST);


	if (blend_) glEnable(GL_BLEND);
	else glDisable(GL_BLEND);

	if (alpha_test_) glEnable(GL_ALPHA_TEST);
	else glDisable(GL_ALPHA_TEST);	

	if (point_smooth_) glEnable(GL_POINT_SMOOTH);
	else glDisable(GL_POINT_SMOOTH);


	if (lighting_) glEnable(GL_LIGHTING);
	else glDisable(GL_LIGHTING);

	if (line_stipple_) glEnable(GL_LINE_STIPPLE);
	else glDisable(GL_LINE_STIPPLE);	

	if (polygon_stipple_) glEnable(GL_POLYGON_STIPPLE);
	else glDisable(GL_POLYGON_STIPPLE);


	if (polygon_offset_) glEnable(GL_POLYGON_OFFSET_FILL);
	else glDisable(GL_POLYGON_OFFSET_FILL);

	if (color_material_) glEnable(GL_COLOR_MATERIAL);
	else glDisable(GL_COLOR_MATERIAL);
	
	glAlphaFunc(alpha_func_, alpha_ref_);
	glBlendFunc(blend_src_, blend_dst_); 

	glPixelStorei(GL_UNPACK_ALIGNMENT, unpack_alignment_);
	glPixelStorei(GL_PACK_ALIGNMENT, pack_alignment_);

	glLineWidth(line_width_);
	glPointSize(point_size_);
	glViewport(vport_[0], vport_[1], vport_[2], vport_[3]);
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();  
	glMultMatrixf(projection_matrix_);
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();  
	glMultMatrixf(model_matrix_);
	glMatrixMode(matrix_mode_);
}
};