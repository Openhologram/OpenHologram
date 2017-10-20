#ifndef __Delaunay_h
#define __Delaunay_h

#include <set>
#include <algorithm>
#include <math.h>
#include "graphics/real.h"
#include "graphics/vec.h"

namespace graphics {

///////////////////
// DVertex

class DVertex
{
public:
	DVertex()					: m_Pnt(0.0F, 0.0F), m_data(0)		{}
	DVertex(const DVertex& v)		: m_Pnt(v.m_Pnt), m_data(v.m_data), m_user_data(v.m_user_data) {}
	DVertex(const vec2& pnt)		: m_Pnt(pnt), m_data(0)				{}
	DVertex(real x, real y)		: m_Pnt(x, y), m_data(0)				{}
	DVertex(int x, int y)		: m_Pnt((real) x, (real) y)	{}

	bool operator<(const DVertex& v) const
	{
		if (m_Pnt[0] == v.m_Pnt[0]) return m_Pnt[1] < v.m_Pnt[1];
		return m_Pnt[0] < v.m_Pnt[0];
	}

	bool operator==(const DVertex& v) const
	{
		return apx_equal(v.m_Pnt, m_Pnt);
	}
	
	real GetX()	const	{ return m_Pnt[0]; }
	real GetY() const	{ return m_Pnt[1]; }

	void SetX(real x)		{ m_Pnt[0] = x; }
	void SetY(real y)		{ m_Pnt[1] = y; }

	const vec2& GetPoint() const		{ return m_Pnt; }
	void* GetData() const { return m_data; }
	void  SetData(void* a) { m_data = a; }
	void  SetUserData(void* a) const{ m_user_data = a; }
	void* GetUerData() const { return m_user_data; }
public:
	vec2	m_Pnt;
	void*   m_data;
	mutable void*   m_user_data;
};

typedef std::set<DVertex> DVertexSet;
typedef std::set<DVertex>::iterator vIterator;
typedef std::set<DVertex>::const_iterator cvIterator;

///////////////////
// DTriangle

class DTriangle
{
public:
	DTriangle()
	{
	}
	DTriangle(const DTriangle& tri)
		: m_Center(tri.m_Center)
		, m_R(tri.m_R)
		, m_R2(tri.m_R2)
	{
		for (int i = 0; i < 3;++i) m_Vertices[i] = tri.m_Vertices[i];
	}
	DTriangle(DVertex * p0, DVertex * p1, DVertex * p2)
	{
		m_Vertices[0] = p0;
		m_Vertices[1] = p1;
		m_Vertices[2] = p2;
		SetCircumCircle();
	}
	DTriangle(DVertex * pV)
	{
		for (int i = 0; i < 3;++i) m_Vertices[i] = pV++;
		SetCircumCircle();
	}

	const DTriangle& operator=( const DTriangle& tri) const
	{
		m_Center = (tri.m_Center);
		m_R = tri.m_R;
		m_R2 = tri.m_R2;
		for (int i = 0; i < 3;++i) m_Vertices[i] = tri.m_Vertices[i];

		return *this;
	}

	//const DTriangle& operator=(const DTriangle& tri) const
	//{
	//	m_Center = tri.m_Center;
	//	m_R = tri.m_R;
	//	m_R2 = tri.m_R2;
	//	for (int i = 0; i < 3;++i) m_Vertices[i] = tri.m_Vertices[i];

	//	return *this;
	//}

	bool operator<(const DTriangle& tri) const
	{
		if (m_Center[0] == tri.m_Center[0]) return m_Center[1] < tri.m_Center[1];
		return m_Center[0] < tri.m_Center[0];
	}

	DVertex * GetVertex(int i) const
	{
		return m_Vertices[i];
	}

	bool IsLeftOf(cvIterator itVertex) const
	{
		// returns true if * itVertex is to the right of the DTriangle's circumcircle
		return itVertex->GetPoint()[0] > (m_Center[0] + m_R);
	}

	bool CCEncompasses(cvIterator itVertex) const
	{
		// Returns true if * itVertex is in the DTriangle's circumcircle.
		// A DVertex exactly on the circle is also considered to be in the circle.

		// These next few lines look like optimisation, however in practice they are not.
		// They even seem to slow down the algorithm somewhat.
		// Therefore, I've commented them out.

		// First check boundary box.
//		real x = itVertex->GetPoint()[0];
//				
//		if (x > (m_Center[0] + m_R)) return false;
//		if (x < (m_Center[0] - m_R)) return false;
//
//		real y = itVertex->GetPoint()[1];
//				
//		if (y > (m_Center[1] + m_R)) return false;
//		if (y < (m_Center[1] - m_R)) return false;

		vec2 dist = itVertex->GetPoint() - m_Center;		// the distance between v and the circle center
		real dist2 = dist[0] * dist[0] + dist[1] * dist[1];		// squared
		return dist2 <= m_R2;								// compare with squared radius
	}
public:
	mutable DVertex * m_Vertices[3];	// the three DTriangle vertices
	mutable vec2 m_Center;				// center of circumcircle
	mutable real m_R;			// radius of circumcircle
	mutable real m_R2;			// radius of circumcircle, squared

	void SetCircumCircle();
};

// Changed in verion 1.1: collect DTriangles in a multiset.
// In version 1.0, I used a set, preventing the creation of multiple
// DTriangles with identical center points. Therefore, more than three
// co-circular vertices yielded incorrect results. Thanks to Roger Labbe.
typedef std::multiset<DTriangle> DTriangleSet;
typedef std::multiset<DTriangle>::iterator tIterator;
typedef std::multiset<DTriangle>::const_iterator ctIterator;

///////////////////
// DEdge

class DEdge
{
public:
	DEdge(const DEdge& e)	: m_pV0(e.m_pV0), m_pV1(e.m_pV1)	{}
	DEdge(DVertex * pV0, DVertex * pV1)
		: m_pV0(pV0), m_pV1(pV1)
	{
	}

	bool operator<(const DEdge& e) const
	{
		if (m_pV0 == e.m_pV0) return * m_pV1 < * e.m_pV1;
		return * m_pV0 < * e.m_pV0;
	}

	DVertex * m_pV0;
	DVertex * m_pV1;
};

typedef std::set<DEdge> DEdgeSet;
typedef std::set<DEdge>::iterator DEdgeIterator;
typedef std::set<DEdge>::const_iterator cDEdgeIterator;

///////////////////
// Delaunay

class Delaunay
{
public:
	// Calculate the Delaunay triangulation for the given set of vertices.
	void Triangulate(DVertexSet& vertices, DTriangleSet& output);

	// Put the DEdges of the DTriangles in an DEdgeSet, eliminating double DEdges.
	// This comes in useful for drawing the triangulation.
	void TrianglesToEdges(DTriangleSet& DTriangles, DEdgeSet& DEdges);
protected:
	void HandleEdge(DVertex * p0, DVertex * p1, DEdgeSet& DEdges);
};

};

#endif