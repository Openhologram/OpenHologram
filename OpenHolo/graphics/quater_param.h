#ifndef __quater_param_h
#define __quater_param_h

#include "graphics/quater.h"


namespace graphics {

class QuaternionParametrization {

public:

	QuaternionParametrization(): h0_(0), a_(0), b_(0), c_(0), d_(0) {}

	QuaternionParametrization(const quater& q);

	QuaternionParametrization(const QuaternionParametrization& cp);

	quater Prametrize(const vec3& input) const;

	void SetInitialQuaternion(const quater& q);

private:

	void Setup();

	

	quater h0_;
	vec3 a_;
	vec3 b_;
	vec3 c_;
	vec3 d_;
};

};

#endif