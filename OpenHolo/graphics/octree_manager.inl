#include <math.h>


template<typename T>
OctreeManager<T>::OctreeManager(int exponent, box3 bbox) : bounding_box_(bbox), num_of_octree_node_((int)::pow(2.0, exponent)), octree_(num_of_octree_node_)
{
	bounding_box_.get_bounds(xmin, ymin, zmin, xmax, ymax, zmax);
	xwidth = xmax-xmin;
	ywidth = ymax-ymin;
	zwidth = zmax-zmin;
}

template<typename T>
const typename OctreeManager<T>::LeafItem& OctreeManager<T>::At(const vec3& v) const
{
	OctreeManager<T>::LeafItem empty;
	if(bounding_box_.is_empty())
		return empty;
	if(num_of_octree_node_ <= 0)
		return empty;

	real a(v.v[0]), b(v.v[1]), c(v.v[2]);
	real aa = 1-(xmax-a)/(xmax-xmin);
	aa *= num_of_octree_node_;
	real bb = 1-(ymax-b)/(ymax-ymin);
	bb *= num_of_octree_node_;
	real cc = 1-(zmax-c)/(zmax-zmin);
	cc *= num_of_octree_node_;

	return octree_(aa,bb,cc);
}

template<typename T>
typename OctreeManager<T>::LeafItem& OctreeManager<T>::At(const vec3& v) {
	OctreeManager<T>::LeafItem empty;
	if(bounding_box_.is_empty())
		return empty;
	if(num_of_octree_node_ <= 0)
		return empty;

	real a(v.v[0]), b(v.v[1]), c(v.v[2]);
	real aa = 1-(xmax-a)/(xmax-xmin);
	aa *= num_of_octree_node_;
	real bb = 1-(ymax-b)/(ymax-ymin);
	bb *= num_of_octree_node_;
	real cc = 1-(zmax-c)/(zmax-zmin);
	cc *= num_of_octree_node_;

	return octree_(aa,bb,cc);
}

template<typename T>
void OctreeManager<T>::Insert(const graphics::box3& bbox, T id)
{
	using namespace graphics;

	if(bounding_box_.is_empty())
		return;
	if(num_of_octree_node_ <= 0)
		return;

	real norm_min_x = ::floor((1.0-(xmax-bbox.get_minimum().v[0])/xwidth)*num_of_octree_node_);
	real norm_min_y = ::floor((1.0-(ymax-bbox.get_minimum().v[1])/ywidth)*num_of_octree_node_);
	real norm_min_z = ::floor((1.0-(zmax-bbox.get_minimum().v[2])/zwidth)*num_of_octree_node_);

	real norm_max_x = ::ceil((1.0-(xmax-bbox.get_maximum().v[0])/xwidth)*num_of_octree_node_);
	real norm_max_y = ::ceil((1.0-(ymax-bbox.get_maximum().v[1])/ywidth)*num_of_octree_node_);
	real norm_max_z = ::ceil((1.0-(zmax-bbox.get_maximum().v[2])/zwidth)*num_of_octree_node_);
	
	if ((int)norm_min_x == (int)norm_max_x) norm_max_x += 1.0;
	if ((int)norm_min_y == (int)norm_max_x) norm_max_y += 1.0;
	if ((int)norm_min_z == (int)norm_max_x) norm_max_z += 1.0;

	for(int i= (int)norm_min_x; i<(int)norm_max_x; i++)
	{
		for(int j= (int)norm_min_y; j< (int) norm_max_y; j++)
		{
			for(int k= (int)norm_min_z; k< (int)norm_max_z; k++)
			{						
				octree_(i, j, k).push_back(id);
			}	
		}
	}
}

template<typename T>
void OctreeManager<T>::At(int i, int j, int k, std::set<T>& ret) const
{
	using namespace graphics;
	const OctreeManager<T>::LeafItem& val = octree_(i, j, k);
	for (uint l = 0; l < val.size() ; l++) {
		ret.insert(val[l]);
	}
}

template<typename T>
void OctreeManager<T>::At(const graphics::box3& bbox, std::set<T>& ret) const
{
	using namespace graphics;

	if(bounding_box_.is_empty())
		return;
	if(num_of_octree_node_ <= 0)
		return;

	real norm_min_x = ::floor((1.0-(xmax-bbox.get_minimum().v[0])/xwidth)*num_of_octree_node_);
	real norm_min_y = ::floor((1.0-(ymax-bbox.get_minimum().v[1])/ywidth)*num_of_octree_node_);
	real norm_min_z = ::floor((1.0-(zmax-bbox.get_minimum().v[2])/zwidth)*num_of_octree_node_);

	real norm_max_x = ::ceil((1.0-(xmax-bbox.get_maximum().v[0])/xwidth)*num_of_octree_node_);
	real norm_max_y = ::ceil((1.0-(ymax-bbox.get_maximum().v[1])/ywidth)*num_of_octree_node_);
	real norm_max_z = ::ceil((1.0-(zmax-bbox.get_maximum().v[2])/zwidth)*num_of_octree_node_);

	for(int i=norm_min_x; i<norm_max_x; i++)
	{
		for(int j=norm_min_y; j<norm_max_y; j++)
		{
			for(int k=norm_min_z; k<norm_max_z; k++)
			{						
				const OctreeManager<T>::LeafItem& val = octree_(i, j, k);
				for (int l = 0; l < val.size() ; l++) {
					ret.insert(val[l]);
				}
			}	
		}
	}
}