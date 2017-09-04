#ifndef OCTREE_MANAGER_H_
#define OCTREE_MANAGER_H_

#include <vector>
#include "graphics/geom.h"
#include "graphics/octree.h"
#include <set>

namespace graphics
{
	/*
	 * OctreeManager는 T의 리스트를 각 leaf에 저장하는 octree를 관리한다.
	 * 이 클래스는 bounding box 하나를 생성자에서 받으며 해당 인스턴스의 Insert에 box3를 넘겨주면
	 * 이 box3를 bounding box에 normalize해서 이 box3의 범위에 들어있는 octree노드에 Insert에 함께 넘겨진 T타입의 인자를 leaf에 저장한다.
	 * Search를 사용하면 좌표를 octree leaf로 계산하고 이 leaf에 들어있는 T 리스트를 반환한다.
	 */
	template< typename T>
	class OctreeManager
	{
	public:
		typedef std::vector<T> LeafItem;	
		/*
		 * exponent
		 * pow(2, exponent)로 octree 크기가 결정된다.
		 * box
		 * bounding box의 크기
		 */
		OctreeManager(int exponent, box3);
		const LeafItem& At(const vec3&) const;
		LeafItem& At(const vec3&);
		void At(const box3& bbox, std::set<T>& ret) const;
		void Insert(const box3& bbox, T id);
		Octree<LeafItem> octree() { return octree_; }
		void At(int i, int j, int k, std::set<T>& ret) const;
	public:
		int num_of_octree_node_;
		Octree< LeafItem > octree_;
		box3 bounding_box_;
		real xmin,ymin,zmin,xmax,ymax,zmax;
		real xwidth, ywidth, zwidth;
	};

#include "graphics/octree_manager.inl"
}

#endif