#ifndef __quad_tree_h_
#define __quad_tree_h_

#include "graphics/vec.h"
#include <vector>

namespace graphics
{
#ifndef max
#define max(a,b) (a>b?a:b)
#endif

	/*! THIS SHOULD BE A TEMPLATE OF BOX!!!!! */
	class  Box2f
	{
		vec2 min_; // Lower-left
		vec2 max_; // Upper-right
	public:
		Box2f() : min_(0, 0), max_(-1, 0) {}
		Box2f(float x, float y, float r, float t) {
			min_[0] = x;
			max_[0] = r;
			min_[1] = y;
			max_[1] = t;
		}
		Box2f(const vec2 p) {
			min_ = p;
			max_ = p;
		}
		Box2f(const vec2 p1, const vec2 p2) {
			min_ = p1;
			max_ = p2;
		}

		const vec2& mymin() const { return min_; }
		const vec2& mymax() const { return max_; }
		float x() const { return min_[0]; }
		void x(float v) { min_[0] = v; }
		float y() const { return min_[1]; }
		void y(float v) { min_[1] = v; }
		float r() const { return max_[0]; }
		void r(float v) { max_[0] = v; }
		float t() const { return max_[1]; }
		void t(float v) { max_[1] = v; }
		float w() const { return max_[0] - min_[0]; }
		float h() const { return max_[1] - min_[1]; }
		vec2 center() const { return (min_ + max_) / 2.0f; }

		void set(float x, float y, float r, float t) {
			min_= vec2(x, y);
			max_= vec2(r, t);
		}
		void set(vec2 p1, vec2 p2) {
			min_ = p1;
			max_ = p2;
		}
		void set(const Box2f& v) { *this = v; }
		void set_min(float x, float y) { min_=vec2(x, y); }
		void set_max(float x, float y) { max_=vec2(x, y); }

		/*! Return true if \b b is inside this box. */
		bool contains(const Box2f& b) const
		{
			return inside(b.mymin()) && inside(b.mymax());
		}
		/*! Return true if the point is inside this box. */
		bool inside(const vec2& p) const
		{
			return p[0] >= min_[0] && p[0] <= max_[0] && p[1] >= min_[1] && p[1] <= max_[1];
		}
		/*! Return true if the point is inside this box. */
		bool inside(float x, float y) const
		{
			return x >= min_[0] && x <= max_[0] && y >= min_[1] && y <= max_[1];
		}
	};

	//----------------------------------------------------------------------

	/*! Quadtree node structure contains the actual data stored in
	the tree, and an array of four possible child nodes. */
	template <class T>
	struct QuadtreeNode
	{
		std::vector<T> data;      //!< The actual data stored in the tree
		QuadtreeNode* child_nodes[4]; //!< Four possible child subnodes

		/*! Constructor clears the child node pointers. */
		QuadtreeNode() { memset(child_nodes, 0, 4 * sizeof(QuadtreeNode*)); }
		/*! Destructor deletes all allocated child nodes. */
		~QuadtreeNode() {
			for (int i = 0; i < 4; ++i)
				delete child_nodes[i];
		}
	};

	//----------------------------------------------------------------------

	/*! This template class provides spatial subdivision functionality of
	a 2D rectangular area and methods to add to or return its contents.
	*/
	template <class T>
	class Quadtree
	{
	public:
		Quadtree(const Box2f &bbox, int max_depth = 16) :
			_bbox(bbox), _max_depth(max_depth) {}

		const Box2f& bbox() const { return _bbox; }
		int max_depth() const { return _max_depth; }
		const QuadtreeNode<T>& root() const { return _root; }

		/*! Add an object with its bounding box to all nodes in the tree that
		\b obj_bbox intersects, adding subdivisions until \b obj_bbox
		is larger than the subdivision size, or the maximum recursion level
		is reached.
		*/
		void add(const T& object, const Box2f& obj_bbox, float min_D = 0.0f)
		{
			_add(&_root, _bbox, object, obj_bbox,
				max(squaredNorm(obj_bbox.mymin()-obj_bbox.mymax()), min_D * min_D), 0);
		}

		/*! Add the object to the root with no further tests. */
		void add_to_root(const T& object) { _root.data.push_back(object); }

		/*! Find the last node in the tree that \b xy intersects, returning a
		pointer to the node's data vector. */
		const std::vector<T>* find(float x, float y) const
		{
			return _bbox.inside(x, y) ? _find(&_root, _bbox, x, y) : 0;
		}

		/*! Find all nodes in the tree that intersect \b bbox and add them to
		/b node_list. */
		int intersections(const Box2f&                         bbox,
			std::vector<const QuadtreeNode<T>*>& node_list) const
		{
			return _intersections(&_root, _bbox, bbox, node_list);
		}

	protected:
		/*! Recursive version of add() that finds a node to add object to. */
		inline void _add(QuadtreeNode<T>* node, const Box2f& node_bbox,
			const T& object, const Box2f& obj_bbox, float D, int depth);
		/*! Recursive version of find() that steps through the tree. */
		inline const std::vector<T>* _find(const QuadtreeNode<T>* node,
			const Box2f& node_bbox,
			float x, float y) const;
		/*! Recursive version of intersections() that steps through the tree. */
		inline int _intersections(const QuadtreeNode<T>*               node,
			const Box2f&                         node_bbox,
			const Box2f&                         bbox,
			std::vector<const QuadtreeNode<T>*>& node_list) const;

	private:
		Box2f _bbox;       //!< The quadtree's 2D extent
		int _max_depth;    //!< Maximum recursion depth
		QuadtreeNode<T> _root; //!< Tree's start node
	};


	//----------------------------------------------------------------------
	//----------------------------------------------------------------------
	// INLINE METHODS:

	/*! Recursive version of add() that finds a node to add object to.
	*/
	template <class T>
	inline void Quadtree<T>::_add(QuadtreeNode<T>* node, const Box2f& node_bbox,
		const T& object, const Box2f& obj_bbox,
		float D, int depth)
	{
		// Check if we can add the object to the current node:
		if (depth >= _max_depth || squaredNorm(node_bbox.mymin()-node_bbox.mymax()) < D) {
			if (node->data.size() && node->data.size() >= node->data.capacity())
				node->data.reserve(node->data.capacity() * 2);
			node->data.push_back(object);
			return;
		}

		// Check which child nodes the object bbox intersect:
		const vec2 mid = node_bbox.center();
		bool over[4];
		over[0] = over[2] = (obj_bbox.x() <= mid[0]);
		over[1] = over[3] = (obj_bbox.r() >  mid[0]);
		over[0] &= (obj_bbox.y() <= mid[1]);
		over[1] &= (obj_bbox.y() <= mid[1]);
		over[2] &= (obj_bbox.t() >  mid[1]);
		over[3] &= (obj_bbox.t() >  mid[1]);

		// Add child nodes which have an intersection flag set:
		++depth;
		Box2f cb;
		for (int i = 0; i < 4; ++i) {
			if (!over[i])
				continue;
			if (!node->child_nodes[i])
				node->child_nodes[i] = new QuadtreeNode<T>;
			// Bbox for child node:
			if (i & 1) {
				cb.x(mid[0]);
				cb.r(node_bbox.r());
			}
			else {
				cb.x(node_bbox.x());
				cb.r(mid[0]);
			}
			if (i & 2) {
				cb.y(mid[1]);
				cb.t(node_bbox.t());
			}
			else {
				cb.y(node_bbox.y());
				cb.t(mid[1]);
			}
			// Add the child node:
			_add(node->child_nodes[i], cb, object, obj_bbox, D, depth);
		}
	}

	/*! Recursive version of find() that steps through the tree.
	*/
	template <class T>
	inline const std::vector<T>* Quadtree<T>::_find(const QuadtreeNode<T>* node,
		const Box2f& node_bbox,
		float x, float y) const
	{
		// Find the child node xy is inside:
		const vec2 mid = node_bbox.center();
		int i = (x > mid[0] ? 1 : 0) + (y > mid[1] ? 2 : 0);
		// If no child node exist, we're done:
		if (!node->child_nodes[i])
			return &node->data;

		// Bbox for child node:
		Box2f cb;
		if (i & 1) {
			cb.x(mid[0]);
			cb.r(node_bbox.r());
		}
		else {
			cb.x(node_bbox.x());
			cb.r(mid[0]);
		}
		if (i & 2) {
			cb.y(mid[1]);
			cb.t(node_bbox.t());
		}
		else {
			cb.y(node_bbox.y());
			cb.t(mid.y);
		}

		// Continue looking on the child node:
		return _find(node->child_nodes[i], cb, x, y);
	}

	/*! Recursive version of intersections() that steps through the tree.
	*/
	template <class T>
	inline int Quadtree<T>::_intersections(const QuadtreeNode<T>*               node,
		const Box2f&                         node_bbox,
		const Box2f&                         bbox,
		std::vector<const QuadtreeNode<T>*>& node_list) const
	{
		// Add this node if it has objects:
		int count = int(node->data.size());
		if (count)
			node_list.push_back(node);

		// Check which child nodes the bbox intersect:
		const vec2 mid = node_bbox.center();
		bool over[4];
		over[0] = over[2] = (bbox.x() <= mid[0]);
		over[1] = over[3] = (bbox.r() >  mid[0]);
		over[0] &= (bbox.y() <= mid[1]);
		over[1] &= (bbox.y() <= mid[1]);
		over[2] &= (bbox.t() >  mid[1]);
		over[3] &= (bbox.t() >  mid[1]);

		// Recurse into any child nodes which have intersections:
		Box2f cb;
		for (int i = 0; i < 4; ++i) {
			const QuadtreeNode<T>* child = node->child_nodes[i];
			if (!over[i] || !child)
				continue;
			// Bbox for child node:
			if (i & 1) {
				cb.x(mid[0]);
				cb.r(node_bbox.r());
			}
			else {
				cb.x(node_bbox.x());
				cb.r(mid[0]);
			}
			if (i & 2) {
				cb.y(mid[1]);
				cb.t(node_bbox.t());
			}
			else {
				cb.y(node_bbox.y());
				cb.t(mid[1]);
			}
			// Call the child node:
			count += _intersections(node->child_nodes[i], cb, bbox, node_list);
		}
		return count;
	}
}

#endif