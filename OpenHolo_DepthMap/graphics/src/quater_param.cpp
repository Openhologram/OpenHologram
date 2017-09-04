#include "graphics/quater_param.h"
#include "graphics/matrix_math.h"

namespace graphics {

QuaternionParametrization::QuaternionParametrization(const quater& q)
: h0_(q)
{
	SetInitialQuaternion(q);
}

void QuaternionParametrization::SetInitialQuaternion(const quater& q)
{
	h0_ = q;

	real n1 = q[0];
	real n2 = q[1];
	real n3 = q[2];
	real n4 = q[3];

	gmatrix<real> mat(4,3);
	mat = 0.0;

	if (n1 > epsilon) {
		mat(0,0) = -n2/n1;
		mat(0,1) = -n3/n1;
		mat(0,2) = -n4/n1;
		mat(1,0) = 1.0;
		mat(2,1) = 1.0;
		mat(3,2) = 1.0;
	}
	else if (n2 > epsilon) {
		mat(1,0) = -n1/n2;
		mat(1,1) = -n3/n2;
		mat(1,2) = -n4/n2;
		mat(0,0) = 1.0;
		mat(2,1) = 1.0;
		mat(3,2) = 1.0;
	}
	else if (n3 > epsilon) {
		mat(2,0) = -n1/n3;
		mat(2,1) = -n2/n3;
		mat(2,2) = -n4/n3;
		mat(0,0) = 1.0;
		mat(1,1) = 1.0;
		mat(3,2) = 1.0;
	}
	else if (n4 > epsilon) {
		mat(3,0) = -n1/n4;
		mat(3,1) = -n2/n4;
		mat(3,2) = -n3/n4;
		mat(0,0) = 1.0;
		mat(1,1) = 1.0;
		mat(2,2) = 1.0;
	}

	SVDMatrix<real> svd(mat);
	a_[0] = svd.U(0,0);
	a_[1] = svd.U(0,1);
	a_[2] = svd.U(0,2);

	b_[0] = svd.U(1,0);
	b_[1] = svd.U(1,1);
	b_[2] = svd.U(1,2);
	
	c_[0] = svd.U(2,0);
	c_[1] = svd.U(2,1);
	c_[2] = svd.U(2,2);
	
	d_[0] = svd.U(3,0);
	d_[1] = svd.U(3,1);
	d_[2] = svd.U(3,2);
}

QuaternionParametrization::QuaternionParametrization(const QuaternionParametrization& cp)
: a_(cp.a_), b_(cp.b_), c_(cp.c_), d_(cp.d_), h0_(cp.h0_)
{
}

quater QuaternionParametrization::Prametrize(const vec3& input) const
{
	vec4 v4;

	v4[0] = inner(a_, input);
	v4[1] = inner(b_, input);
	v4[2] = inner(c_, input);
	v4[3] = inner(d_, input);

	real theta = norm(v4);
	quater out;

	if (theta < 1.0e-20) {
		out = h0_;
	}
	else {
		v4 /= theta;
		out = (sin(theta) * v4) + (cos(theta) * h0_);
	}

	return out;
}
};