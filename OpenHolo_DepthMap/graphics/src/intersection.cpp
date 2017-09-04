#include "graphics/intersection.h"

namespace graphics {

bool CreateConvexHull(const frame& pl,
					  std::vector<vec3>& points,
					  std::vector<vec3>& out_v)
{
	vec3 a3 = pl.to_model(points[0]);
	vec3 b3 = pl.to_model(points[1]);
	vec3 c3 = pl.to_model(points[2]);
	vec3 d3 = pl.to_model(points[3]);

	vec2 a(a3[0], a3[1]);
	vec2 b(b3[0], b3[1]);
	vec2 c(c3[0], c3[1]);
	vec2 d(d3[0], d3[1]);

	bool abc = ccw(a,b,c);
	bool abd = ccw(a,b,d);
	bool bcd = ccw(b,c,d);
	bool cad = ccw(c,a,d);

	if (!abc) {
		abc = !abc;
		abd = !abd;
		bcd = !bcd;
		cad = !cad;
	}

	if (abc && abd && bcd && cad) {
		//abc
		return false;
	}
	if (abc && abd && bcd && !cad) {
		//abcd
		return true;
	}
	if (abc && abd && !bcd && cad) {
		//abdc
		vec3 tmp = points[2];
		points[2] = points[3];
		points[3] = tmp;
		tmp = out_v[2];
		out_v[2] = out_v[3];
		out_v[3] = tmp;
		return true;
	}
	if (abc && abd && !bcd && !cad) {
		//abd
		return false;
	}
	if (abc && !abd && bcd && cad) {
		//adbc
		std::vector<vec3> tmp = points;
		points[1] = tmp[3];
		points[2] = tmp[1];
		points[3] = tmp[2];
		tmp = out_v;
		out_v[1] = tmp[3];
		out_v[2] = tmp[1];
		out_v[3] = tmp[2];
		return true;
	}
	if (abc && !abd && bcd && !cad) {
		return false;
	}
	if (abc && !abd && !bcd && !cad) {
		return false;
	}
	return false;
}

bool TetrahedronPlaneIntersect(const frame& pl, 
							   const vec3&  a, 
							   const vec3&  b, 
							   const vec3&  c, 
							   const vec3&  d, 
							   const vec3&  av,
							   const vec3&  bv,
							   const vec3&  cv,
							   const vec3&  dv,
							   std::vector<vec3>& poly,
							   std::vector<vec3>& out_v)
{
	real v1 = pl.distance_to(a);
	real v2 = pl.distance_to(b);
	real v3 = pl.distance_to(c);
	real v4 = pl.distance_to(d);

	if (v1*v2 <= 0.0 || v1*v3 <= 0.0 || v1*v4 <= 0.0 || v2*v3 <= 0.0 || v2*v4 <= 0.0 || v3*v4 <= 0.0) {

		real t;
		vec3 p;
		std::vector<vec3> tmp_poly;
		std::vector<vec3> tmp_val;

		bool mark1 = false, mark2 = false, mark3 = false, mark4 = false;

		if (v1*v2<=0.0) {

			if (apx_equal(v1,0, zero_epsilon)) {
				tmp_poly.push_back(a);
				tmp_val.push_back(av);
				mark1 = true;
			}
			else if (apx_equal(v2, 0, zero_epsilon)) {
				tmp_poly.push_back(b);
				tmp_val.push_back(bv);
				mark2 = true;
			}
			else if (pl.intersect(a,b,t,p)) {
				tmp_poly.push_back(p);
				tmp_val.push_back(av + (bv-av)*t);
			}
		}
		if (v1*v3<=0.0) {

			if (!mark1 && apx_equal(v1,0, zero_epsilon)) {
				tmp_poly.push_back(a);
				tmp_val.push_back(av);
				mark1 = true;
			}
			else if (apx_equal(v3, 0, zero_epsilon)) {
				tmp_poly.push_back(c);
				tmp_val.push_back(cv);
				mark3 = true;
			}
			else if (!mark1 && !mark3 && pl.intersect(a,c,t,p)) {
				tmp_poly.push_back(p);
				tmp_val.push_back(av + (cv-av)*t);
			}
		}
		if (v1*v4<=0.0) {
			if (!mark1 && apx_equal(v1,0, zero_epsilon)) {
				tmp_poly.push_back(a);
				tmp_val.push_back(av);
				mark1 = true;
			}
			else if (apx_equal(v4, 0, zero_epsilon)) {
				tmp_poly.push_back(d);
				tmp_val.push_back(dv);
				mark4 = true;
			}
			else if (!mark1 && !mark4&& pl.intersect(a,d,t,p)) {
				tmp_poly.push_back(p);
				tmp_val.push_back(av + (dv-av)*t);
			}
		}
		if (v2*v3<=0.0) {
			if (!mark2 && apx_equal(v2,0, zero_epsilon)) {
				tmp_poly.push_back(b);
				tmp_val.push_back(bv);
				mark2 = true;
			}
			else if (!mark3 && apx_equal(v3, 0, zero_epsilon)) {
				tmp_poly.push_back(c);
				tmp_val.push_back(cv);
				mark3 = true;
			}
			else if (!mark2 && !mark3 && pl.intersect(b,c,t,p)) {
				tmp_poly.push_back(p);
				tmp_val.push_back(bv + (cv-bv)*t);
			}
		}
		if (tmp_poly.size() < 4 && v2*v4<=0.0) {
			if (!mark2 && apx_equal(v2,0, zero_epsilon)) {
				tmp_poly.push_back(b);
				tmp_val.push_back(bv);
				mark2 = true;
			}
			else if (!mark4 && apx_equal(v4, 0, zero_epsilon)) {
				tmp_poly.push_back(d);
				tmp_val.push_back(dv);
				mark4 = true;
			}
			else if (!mark2 && !mark4 && pl.intersect(b,d,t,p)) {
				tmp_poly.push_back(p);
				tmp_val.push_back(bv + (dv-bv)*t);
			}
		}
		if (tmp_poly.size() < 4 && v3*v4<=0.0) {
			if (!mark3 && apx_equal(v3,0, zero_epsilon)) {
				tmp_poly.push_back(c);
				tmp_val.push_back(cv);
				mark3 = true;
			}
			else if (!mark4 && apx_equal(v4, 0, zero_epsilon)) {
				tmp_poly.push_back(d);
				tmp_val.push_back(dv);
				mark4 = true;
			}
			else if (!mark3 && !mark4 && pl.intersect(c,d,t,p)) {
				tmp_poly.push_back(p);
				tmp_val.push_back(cv + (dv-cv)*t);
			}
		}

		if (tmp_poly.size() >= 3) {
			if (tmp_poly.size() == 3) {
				poly = tmp_poly;
				out_v = tmp_val;
			}
			else {
				bool ret = CreateConvexHull(pl, tmp_poly, tmp_val);
				if (!ret) return false;
				poly = tmp_poly;
				out_v = tmp_val;
			}

			return true;
		}
	}

	return false;
}

};

