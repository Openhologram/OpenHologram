#include "graphics/homography.h"
#include "graphics/barycentric.h"
#include "graphics/geom.h"
#include "graphics/gmatrix.h"
#include "graphics/matrix_math.h"
#include "graphics/barycentric.h"

namespace graphics {


Homography::Homography(std::vector<vec2>& xy, std::vector<vec2>& uv)
: homography_defined_(false), well_defined_(false), output_()
{
	if (xy.size() != uv.size()) return;
	if (xy.size() < 3) return;

	if (xy.size() >= 3) {
		if (ComputeHomography(xy, uv) == 2) return;
	}
}


Homography::Homography(const Homography& cp)
: homography_defined_(cp.homography_defined_), well_defined_(cp.well_defined_), output_(cp.output_), p1_(cp.p1_), p2_(cp.p2_), p3_(cp.p3_), uv1_(cp.uv1_), uv2_(cp.uv2_), uv3_(cp.uv3_)
{
}

Homography::Homography()
: homography_defined_(false), well_defined_(false), output_()
{
}

vec2 Homography::map(const vec2& xy)
{
	if (!homography_defined_) return vec2(0);

	vec3 ret;
	ret[0] = (output_[0] * xy[0]) + (output_[1] * xy[1]) + (output_[2]);
	ret[1] = (output_[3] * xy[0]) + (output_[4] * xy[1]) + (output_[5]);
	ret[2] = (output_[6] * xy[0]) + (output_[7] * xy[1]) + (output_[8]);

	return vec2(ret[0]/ret[2], ret[1]/ret[2]);
}

bool Homography::IsHomographyDefined() const
{
	return homography_defined_;
}

std::vector<real>& Homography::GetMatrix() 
{ return output_; }


int Homography::ComputeHomography(std::vector<vec2>& xy, std::vector<vec2>& uv)
{
	bool pass = false;
	int  third = -1;
	for (int i = 2 ; i < xy.size() ;++i){
		real area = tri_area(xy[0], xy[1], xy[i]);
		if (fabs(area)> 0.0000001) {
			pass = true;
			third = i;
			break;
		}
	}

	if (!pass) return 2;

	p1_ = xy[0];
	p2_ = xy[1];
	p3_ = xy[third];
	uv1_ = uv[0];
	uv2_ = uv[1];
	uv3_ = uv[third];

	pass = false;
	int forth = -1;

	homography_defined_ = true;

	for (int i = 2; i < xy.size() ;++i){
		if (i == third) continue;
		real area1 = tri_area(p1_, p2_, xy[i]);
		real area2 = tri_area(p1_, p3_, xy[i]);
		real area3 = tri_area(p2_, p3_, xy[i]);
		if (fabs(area1) >= 0.000000000001 && fabs(area2) >= 0.000000000001 && fabs(area3) >= 0.000000000001) {
			pass = true;
			forth = i;
			break;
		}
	}


	if (!pass) {
		std::vector<vec2> xy_tmp(3);
		std::vector<vec2> uv_tmp(3);

		xy_tmp[0] = p1_;
		xy_tmp[1] = p2_;
		xy_tmp[2] = p3_;


		uv_tmp[0] = uv1_;
		uv_tmp[1] = uv2_;
		uv_tmp[2] = uv3_;

		gmatrix<real> matrix(6,7);
		matrix = 0.0;

		for (int i = 0 ; i < 3  ;++i){
			matrix(i*2, 0) = -xy_tmp[i][0];
			matrix(i*2, 1) = -xy_tmp[i][1];
			matrix(i*2, 2) = -1;
			matrix(i*2, 6) = uv_tmp[i][0];

			matrix(i*2 + 1, 3) = -xy_tmp[i][0];
			matrix(i*2 + 1, 4) = -xy_tmp[i][1];
			matrix(i*2 + 1, 5) = -1;
			matrix(i*2 + 1, 6) = uv_tmp[i][1];
		}

		SVDMatrix<real> svd(matrix) ;

		int ith = -1;
		real minsig = 1.0e+15;

		for (int i = 0 ; i < svd.sig.size() ;++i){
			if (svd.sig[i] < minsig) {
				ith = i;
				minsig = svd.sig[i];
			}
		}

		output_.resize(9);
		for (int i = 0 ; i < 6 ;++i){
			output_[i] = svd.V_(i, ith); //sol[i];
			//output[i] = sol[i];
		}
		output_[6] = output_[7] = 0.0;
		output_[8] = svd.V_(6, ith);

		homography_defined_ = true;
		return 3;
	}

	well_defined_ = true;
	std::vector<vec2> xy_tmp(4);
	std::vector<vec2> uv_tmp(4);

	xy_tmp[0] = p1_;
	xy_tmp[1] = p2_;
	xy_tmp[2] = p3_;
	xy_tmp[3] = xy[forth];

	uv_tmp[0] = uv1_;
	uv_tmp[1] = uv2_;
	uv_tmp[2] = uv3_;
	uv_tmp[3] = uv[forth];
	gmatrix<real> matrix(8,9);
	matrix = 0.0;

	for (int i = 0 ; i < 4  ;++i){
		matrix(i*2, 0) = -xy_tmp[i][0];
		matrix(i*2, 1) = -xy_tmp[i][1];
		matrix(i*2, 2) = -1;
		matrix(i*2, 6) = xy_tmp[i][0] * uv_tmp[i][0];
		matrix(i*2, 7) = xy_tmp[i][1] * uv_tmp[i][0];
		matrix(i*2, 8) = uv_tmp[i][0];

		matrix(i*2 + 1, 3) = -xy_tmp[i][0];
		matrix(i*2 + 1, 4) = -xy_tmp[i][1];
		matrix(i*2 + 1, 5) = -1;
		matrix(i*2 + 1, 6) = xy_tmp[i][0] * uv_tmp[i][1];
		matrix(i*2 + 1, 7) = xy_tmp[i][1] * uv_tmp[i][1];
		matrix(i*2 + 1, 8) = uv_tmp[i][1];
	}

	SVDMatrix<real> svd(matrix) ;

	int ith = -1;
	real minsig = 1.0e+15;

	for (int i = 0 ; i < svd.sig.size() ;++i){
		if (svd.sig[i] < minsig) {
			ith = i;
			minsig = svd.sig[i];
		}
	}

	output_.resize(9);
	for (int i = 0 ; i < output_.size() ;++i){
		output_[i] = svd.V_(i, ith); //sol[i];
		//output[i] = sol[i];
	}
	return 4;
}
};