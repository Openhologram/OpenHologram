#ifndef  __slic_hierarchy_h
#define  __slic_hierarchy_h

#include "graphics/geom.h"
#include <set>
#include <vector>

namespace graphics {

	struct s_segment{
		box2 box;
		std::set<int> parents;
		s_segment() : box(), parents() {}
		s_segment(const s_segment& c) : box(c.box), parents(c.parents) {}
	};

	struct s_level {
		int* labels;
		std::vector<s_segment> segments;
		s_level() : labels(0), segments() {}
		s_level(const s_level& c) : labels(c.labels), segments(c.segments) {}
		~s_level() { if (labels) delete labels; }
	};

class slic_hierarchy {

public:

	slic_hierarchy(): hierarchy_() {}
	~slic_hierarchy() {}

	void build(unsigned int* img, int w, int h, int l0, int l1, int l2, int l3, int compact);
	void build(unsigned int* img, int w, int h, int l0, int l1, int l2, int compact);
	std::vector<s_level> hierarchy_;
};

}

#endif