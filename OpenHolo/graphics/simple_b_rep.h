//
// B-Rep data structure for implicit surface polygonization
//

#pragma once

#include <vector>
#include <map>
#include <set>
#include "graphics/vec.h"
#include "graphics/Delaunay.h"

using namespace graphics;

namespace graphics {

struct GEdge;
struct GVertex;
struct GTriangle;

typedef std::vector<GEdge*>			GEdges;
typedef std::vector<GVertex*>		GVertices;
typedef std::vector<GTriangle*>		GTriangles;

typedef std::map<long, GVertex*>	GVertexMap;
typedef std::map<long, GTriangle*>	GTriangleMap;


struct GVertex {
	long	id;
	void	*user_data;	//user data
	vec3	pos_3d;
	bool	active;
	vec2	pos;
	GEdges	stellar;
	
	GVertex(): stellar(), pos(0), pos_3d(0), id(-1), user_data(0), active(true) {}

	bool stellar_has(GEdge *e) const;
	bool stellar_has(GVertex* v) const;

	GEdges::iterator find(GEdge* e);
	GEdges& get_stellar() 
		{ return stellar; }

	const GEdges& get_stellar() const 
		{ return stellar; }

	int order() const
		{ return stellar.size();	}

	// Connect the newly created Edge, e, to this Vertex
	bool add_edge(GEdge* e);

	// disconnect the edge; remove it from the stellar
	bool remove_edge(GEdge* e);

	void set_user_data(void* val);
};

struct GEdge {

	GVertex	*v1, *v2;
	int user_data;


	GEdge(): v1(0), v2(0), user_data(0) {}

};

struct GTriangle {
	long id;
	GVertex *v1, *v2, *v3;
	
	GTriangle(): v1(0), v2(0), v3(0), id(-1) {}
};

struct GMesh {

	long				cur_vid;
	long				cur_tid;

	GVertexMap			vertex_map;
	GTriangleMap		triangle_map;
	DVertexSet			delaunay_vertex_set_;
	DEdgeSet			delaunay_edge_set_;
	DTriangleSet		delaunay_triangle_set_;

	GMesh(): vertex_map(), triangle_map(), cur_vid(0), cur_tid(0), delaunay_vertex_set_(), delaunay_edge_set_(), delaunay_triangle_set_() {}

	GVertex*			add_vertex(const vec2 &pos);
	GVertex*			add_vertex(const vec2 &pos, void* v);
	GVertex*			find_vertex(long id) const;
	GEdge*				find_edge(const GVertex *v1, const GVertex *v2) const;
	GEdge*				find_edge(const long id1, const long id2) const;
	void				add_triangle(GVertex *v1, GVertex *v2, GVertex *v3);

	void				add_delaunay_vertices(std::vector<vec2>& positions, std::vector<void*>& user_data);
	void				touch_edge(GTriangle* t);

	void				clear();

	~GMesh();
};

typedef std::set<GEdge*> GEdgeSet;

class GMeshEdgeIter {

	GMesh *mesh;
	GEdgeSet traversed;
	GEdgeSet::iterator cur_edge;

	GMeshEdgeIter() {}

	void init();
	void traverse();

public:

	GMeshEdgeIter(GMesh *inMesh);

	GEdge* get();
	void  next();
	bool  is_end();

	GEdgeSet& get_all() 
		{ return traversed; }

	const GEdgeSet& get_all() const 
		{ return traversed; }
};


class GMeshVertexIter {

	GMesh *mesh;
	GMeshVertexIter() {}

	GVertexMap::iterator current_vertex;

public:

	GMeshVertexIter(GMesh* in_mesh);

	GVertex* get();

	bool  is_end() const;

	GVertex *const get() const;

	void next();

	void reset();

	void init();
	
};

class GMeshTriangleIter {

	GMesh *mesh;
	GMeshTriangleIter() {}

	GTriangleMap::iterator current_face;

public:

	GMeshTriangleIter(GMesh* in_mesh);

	GTriangle* get();

	bool  is_end() const;

	GTriangle *const get() const;

	void next();

	void reset();

	void init();
	
};

}; // namespace kernel