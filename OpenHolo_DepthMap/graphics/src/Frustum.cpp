#include "graphics/Frustum.h"
#include "graphics/epsilon.h"

namespace graphics {

	Frustum::Frustum(CameraMap* cm)
		: geom_(cm)
	{
		Update();
	}

	Frustum::Frustum()
	{

	}


	static void get_frustum(CameraMap* view_, std::vector<vec3>& top, std::vector<vec3>&bottom)
	{

		real y = view_->forced_far_depth() * tan(radian(view_->get_angle()/2.0));

		real x = y * view_->get_aspect_ratio();

		vec3 a2(x,y,view_->forced_far_depth());
		vec3 b2(-x,y,view_->forced_far_depth());
		vec3 c2(-x,-y,view_->forced_far_depth());
		vec3 d2(x,-y, view_->forced_far_depth());

		y = view_->forced_near_depth() * tan(radian(view_->get_angle()/2.0));
		x = y * view_->get_aspect_ratio();

		vec3 a1(x,y,view_->forced_near_depth());
		vec3 b1(-x,y,view_->forced_near_depth());
		vec3 c1(-x,-y,view_->forced_near_depth());
		vec3 d1(x,-y, view_->forced_near_depth());

		a1 = view_->getCameraPose().to_world(a1);
		b1 = view_->getCameraPose().to_world(b1);
		c1 = view_->getCameraPose().to_world(c1);
		d1 = view_->getCameraPose().to_world(d1);

		a2 = view_->getCameraPose().to_world(a2);
		b2 = view_->getCameraPose().to_world(b2);
		c2 = view_->getCameraPose().to_world(c2);
		d2 = view_->getCameraPose().to_world(d2);

		top.push_back(a1);
		top.push_back(b1);
		top.push_back(c1);
		top.push_back(d1);


		bottom.push_back(a2);
		bottom.push_back(b2);
		bottom.push_back(c2);
		bottom.push_back(d2);
	}

	void Frustum::SetCamera(CameraMap* cm)
	{
		geom_ = cm;
		Update();
	}

	void Frustum::Update()
	{
		std::vector<vec3> top;
		std::vector<vec3> bottom;

		get_frustum(geom_, top, bottom);

		plane_[0].reset(top[0], top[1], top[2]);
		plane_[1].reset(bottom[2], bottom[1], bottom[0]);
		plane_[2].reset(top[1],bottom[1], bottom[2]);
		plane_[3].reset(top[0], top[3], bottom[3]);
		plane_[4].reset(top[1], top[0], bottom[0]);
		plane_[5].reset(top[3], top[2], bottom[2]);
	}

	bool Frustum::IsInside(const vec3& p) const
	{
		for (int i =0 ; i < 6 ;++i) {
			real d = plane_[i].signed_distance(p);
			if (apx_equal(d, 0, epsilon) || d < 0) return false;
		}
		return true;
	}

}