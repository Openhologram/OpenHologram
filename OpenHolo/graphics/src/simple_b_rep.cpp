#include "graphics/simple_b_rep.h"

namespace graphics {

bool 
GVertex::add_edge(GEdge* e)
{
	if (e->v1 == this || e->v2 == this) {
		if (!stellar_has(e)) { 
			stellar.push_back(e); 
			return true; 
		}
	}

	return false;
}

void
GVertex::set_user_data(void* val)
{
	user_data = val;
}

bool 
GVertex::stellar_has(GEdge *e) const
{
	for (unsigned int i = 0 ; i < stellar.size() ;++i){
		if (stellar[i] == e) return true;
	}
	return false;
}

bool 
GVertex::stellar_has(GVertex* v) const
{
	for (unsigned int i = 0 ; i < stellar.size() ;++i){
		if (stellar[i]->v1 == v) return true;
		if (stellar[i]->v2 == v) return true;
	}
	return false;
}

GEdges::iterator GVertex::find(GEdge* e)
{
	GEdges::iterator ret = stellar.begin();

	for (; ret != stellar.end(); ret++) {
		if (*ret == e) 
			return ret;
	}
	return stellar.end();
}


bool 
GVertex::remove_edge(GEdge* e)
{
	GEdges::iterator i = find(e);

	if (i != stellar.end()) {
		stellar.erase(i);
		//LOG("stellar size %d\n",stellar.size());
		return true;
	}

	return false;
}

void 
GMesh::touch_edge(GTriangle* t)
{
	GEdge* e1 = find_edge(t->v1, t->v2);
	GEdge* e2 = find_edge(t->v2, t->v3);
	GEdge* e3 = find_edge(t->v3, t->v1);

	if (!e1 || !e2 || !e3) return;
	
	e1->user_data += 1;
	e2->user_data += 1;
	e3->user_data += 1;
}

GVertex* 
GMesh::add_vertex(const vec2 &pos)
{
	GVertex* v = new GVertex;
	v->pos = pos;
	v->id = cur_vid++;
	vertex_map[v->id] = v;

	return v;
}

GVertex* 
GMesh::add_vertex(const vec2 &pos, void* vv)
{
	GVertex* v = new GVertex;
	v->user_data = vv;
	v->pos = pos;
	v->id = cur_vid++;
	vertex_map[v->id] = v;

	return v;
}
GEdge*
GMesh::find_edge(const GVertex *v1, const GVertex *v2) const
{
	if (!v1 || !v2) return 0;

	for (int i = 0 ; i < v1->stellar.size() ;++i){
		GEdge *e = v1->stellar[i];
		if (e->v1 == v1 && e->v2 == v2)
			return e;
	}
	return 0;
}

GEdge*		
GMesh::find_edge(const long id1, const long id2) const
{
	GVertex *v1 = find_vertex(id1);
	if (!v1) return 0;
	GVertex *v2 = find_vertex(id2);
	if (!v2) return 0;

	return find_edge(v1, v2);
}

GVertex*
GMesh::find_vertex(long id) const 
{
	GVertexMap::const_iterator p;

	p = vertex_map.find(id);
	if (p == vertex_map.end()) return 0;
	
	GVertex* v = p->second;
	return v;
}

void
GMesh::add_delaunay_vertices(std::vector<vec2>& positions, std::vector<void*>& user_data)
{
	if (positions.size() != user_data.size()) return;

	delaunay_vertex_set_.clear();

	for (int i = 0 ; i < positions.size() ;++i){
		DVertex v(positions[i]);
		v.SetData(user_data[i]);
		delaunay_vertex_set_.insert(v);
	}

	for (DVertexSet::iterator i = delaunay_vertex_set_.begin() ; i != delaunay_vertex_set_.end() ;++i)
	{
		GVertex* v = add_vertex((*i).GetPoint(), (*i).GetData());
		//(*i).SetUserData(v);
		(*i).SetUserData(v);
	}

	Delaunay d;

	delaunay_triangle_set_.clear();
	delaunay_edge_set_.clear();

	d.Triangulate(delaunay_vertex_set_, delaunay_triangle_set_);
	for (DTriangleSet::iterator i = delaunay_triangle_set_.begin() ; i != delaunay_triangle_set_.end() ;++i)
	{
		GVertex* v1 = (GVertex*) (*i).GetVertex(0)->GetUerData();
		GVertex* v2 = (GVertex*) (*i).GetVertex(1)->GetUerData();
		GVertex* v3 = (GVertex*) (*i).GetVertex(2)->GetUerData();
		add_triangle(v1, v2, v3);
	}
}

void
GMesh::add_triangle(GVertex *v1, GVertex *v2, GVertex *v3)
{
	GTriangle *triangle = new GTriangle();
	
	GVertex* vv1 = v1;
	GVertex* vv2 = v2;
	if (!find_edge(vv1, vv2)) {
		GEdge* e = new GEdge();
		e->v1 = vv1;
		e->v2 = vv2;
		vv1->add_edge(e);
		vv2->add_edge(e);
	}

	vv1 = v2;
	vv2 = v3;
	if (!find_edge(vv1, vv2)) {
		GEdge* e = new GEdge();
		e->v1 = vv1;
		e->v2 = vv2;
		vv1->add_edge(e);
		vv2->add_edge(e);
	}

	vv1 = v3;
	vv2 = v1;
	if (!find_edge(vv1, vv2)) {
		GEdge* e = new GEdge();
		e->v1 = vv1;
		e->v2 = vv2;
		vv1->add_edge(e);
		vv2->add_edge(e);
	}

	triangle->v1 = v1;
	triangle->v2 = v2;
	triangle->v3 = v3;

	triangle->id = cur_tid++;
	triangle_map[triangle->id] = triangle;
}

void GMesh::clear()
{
	GMeshEdgeIter ME(this);
	
	GEdgeSet &es = ME.get_all();
	GEdgeSet::iterator i = es.begin();
	for (; i != es.end() ;++i){
		GEdge *e = *i;
		delete e;
	}

	// now delete wings
	GMeshTriangleIter MT(this);
	for (; !MT.is_end() ; MT.next()) {
		GTriangle *t = MT.get();
		delete t;
	}

	// now delete vertices
	GMeshVertexIter MV(this);
	for (; !MV.is_end() ; MV.next()) {
		delete MV.get();
	}

	cur_vid = 0;
	cur_tid = 0;

	triangle_map.clear();
	vertex_map.clear();
}

GMesh::~GMesh()
{
	clear();
}

GMeshEdgeIter::GMeshEdgeIter(GMesh *inMesh)
: mesh(inMesh), traversed()
{
	init();
}

void GMeshEdgeIter::init()
{
	if (!mesh) return;
	traverse();
	cur_edge = traversed.begin();
}

void GMeshEdgeIter::traverse()
{
	traversed.clear();

	GMeshVertexIter viter(mesh);

	for ( ; !viter.is_end() ; viter.next()) {
		GVertex *v = viter.get();
		GEdges& edges = v->get_stellar();
		//printf("vertex id %d: stellar size %d\n", v->get_id(), edges.size());
		for (unsigned int i = 0 ; i < edges.size() ;++i){
			traversed.insert(edges[i]);
		}
	}
}

GEdge* 
GMeshEdgeIter::get()
{
	if (!mesh) return 0;

	if (cur_edge != traversed.end()) 
		return *cur_edge;

	return 0;
}

void  
GMeshEdgeIter::next()
{
	cur_edge++;
}

bool  
GMeshEdgeIter::is_end()
{
	if (!mesh) return true;
	if (cur_edge != traversed.end()) return false;
	return true;
}


GMeshVertexIter::GMeshVertexIter(GMesh* in_mesh)
	: mesh(in_mesh)
{
	init();
}

void
GMeshVertexIter::init()
{
	if (!mesh) return;

	current_vertex = mesh->vertex_map.begin();
}

GVertex* 
GMeshVertexIter::get()
{
	if (!mesh) return 0;
	return current_vertex->second;
}

bool  
GMeshVertexIter::is_end() const
{
	if (!mesh) return true;
	if (current_vertex == mesh->vertex_map.end()) return true;
	return false;
}

GVertex *const 
GMeshVertexIter::get() const
{
	if (!mesh) return 0;
	return current_vertex->second;
}

void 
GMeshVertexIter::next()
{
	if (!mesh) return;
	current_vertex++;
}

void 
GMeshVertexIter::reset()
{
	if (!mesh) return;
	mesh = 0;
}



GMeshTriangleIter::GMeshTriangleIter(GMesh* in_mesh)
	: mesh(in_mesh)
{
	init();
}

void
GMeshTriangleIter::init()
{
	if (!mesh) return;
	current_face = mesh->triangle_map.begin();
}

GTriangle* 
GMeshTriangleIter::get()
{
	if (!mesh) return 0;
	return current_face->second;
}

bool  
GMeshTriangleIter::is_end() const
{
	if (!mesh) return true;
	if (current_face == mesh->triangle_map.end()) return true;
	return false;
}

GTriangle *const 
GMeshTriangleIter::get() const
{
	if (!mesh) return 0;
	return current_face->second;
}

void 
GMeshTriangleIter::next()
{
	if (!mesh) return;
	current_face++;
}

void 
GMeshTriangleIter::reset()
{
	if (!mesh) return;
	mesh = 0;
}
};