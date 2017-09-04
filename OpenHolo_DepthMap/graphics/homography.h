#ifndef __homography_h
#define __homography_h

#include "graphics/vec.h"
#include <vector>


namespace graphics {

class Homography {
public:

	Homography(std::vector<vec2>& xy, std::vector<vec2>& uv);

	Homography(const Homography& cp);

	Homography();

	Homography& operator=(const Homography& cp)
    {
		homography_defined_ = cp.homography_defined_;
		well_defined_ = cp.well_defined_;
		output_ = cp.output_;
		p1_ = cp.p1_;
		p2_ = cp.p2_;
		p3_ = cp.p3_;
		uv1_ = cp.uv1_;
		uv2_ = cp.uv2_;
		uv3_ = cp.uv3_;

		return *this;
	}

	vec2 map(const vec2& xy);

	bool IsHomographyDefined() const;

	std::vector<real>& GetMatrix();

private:

	int ComputeHomography(std::vector<vec2>& xy, std::vector<vec2>& uv);

	bool homography_defined_;

	bool well_defined_;

	std::vector<real> output_;

	vec2 p1_;
	vec2 p2_;
	vec2 p3_;

	vec2 uv1_;
	vec2 uv2_;
	vec2 uv3_;
};

};

#endif