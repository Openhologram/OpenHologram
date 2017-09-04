#include "graphics/Delaunay.h"

#include <iterator>

using namespace std;
namespace graphics {

const real sqrt3 = 1.732050808F;

void DTriangle::SetCircumCircle()
{
	real x0 = m_Vertices[0]->GetX();
	real y0 = m_Vertices[0]->GetY();

	real x1 = m_Vertices[1]->GetX();
	real y1 = m_Vertices[1]->GetY();

	real x2 = m_Vertices[2]->GetX();
	real y2 = m_Vertices[2]->GetY();

	real y10 = y1 - y0;
	real y21 = y2 - y1;

	bool b21zero = y21 > -epsilon && y21 < epsilon;

	if (y10 > -epsilon && y10 < epsilon)
	{
		if (b21zero)	// All three vertices are on one horizontal line.
		{
			if (x1 > x0)
			{
				if (x2 > x1) x1 = x2;
			}
			else
			{
				if (x2 < x0) x0 = x2;
			}
			m_Center[0] = (x0 + x1) * .5F;
			m_Center[1] = y0;
		}
		else	// m_Vertices[0] and m_Vertices[1] are on one horizontal line.
		{
			real m1 = - (x2 - x1) / y21;

			real mx1 = (x1 + x2) * .5F;
			real my1 = (y1 + y2) * .5F;

			m_Center[0] = (x0 + x1) * .5F;
			m_Center[1] = m1 * (m_Center[0] - mx1) + my1;
		}
	}
	else if (b21zero)	// m_Vertices[1] and m_Vertices[2] are on one horizontal line.
	{
		real m0 = - (x1 - x0) / y10;

		real mx0 = (x0 + x1) * .5F;
		real my0 = (y0 + y1) * .5F;

		m_Center[0] = (x1 + x2) * .5F;
		m_Center[1] = m0 * (m_Center[0] - mx0) + my0;
	}
	else	// 'Common' cases, no multiple vertices are on one horizontal line.
	{
		real m0 = - (x1 - x0) / y10;
		real m1 = - (x2 - x1) / y21;

		real mx0 = (x0 + x1) * .5F;
		real my0 = (y0 + y1) * .5F;

		real mx1 = (x1 + x2) * .5F;
		real my1 = (y1 + y2) * .5F;

		m_Center[0] = (m0 * mx0 - m1 * mx1 + my1 - my0) / (m0 - m1);
		m_Center[1] = m0 * (m_Center[0] - mx0) + my0;
	}

	real dx = x0 - m_Center[0];
	real dy = y0 - m_Center[1];

	m_R2 = dx * dx + dy * dy;	// the radius of the circumcircle, squared
	m_R = (real) sqrt(m_R2);	// the proper radius

	// Version 1.1: make m_R2 slightly higher to ensure that all DEdges
	// of co-circular vertices will be caught.
	// Note that this is a compromise. In fact, the algorithm isn't really
	// suited for very many co-circular vertices.
	m_R2 *= 1.000001f;
}

// Function object to check whether a DTriangle has one of the vertices in SuperTriangle.
// operator() returns true if it does.
class DTriangleHasVertex
{
public:
	DTriangleHasVertex(const DVertex SuperTriangle[3]) : m_pSuperTriangle(SuperTriangle)	{}
	bool operator()(const DTriangle& tri) const
	{
		for (int i = 0; i < 3;++i)
		{
			const DVertex * p = tri.GetVertex(i);
			if (p >= m_pSuperTriangle && p < (m_pSuperTriangle + 3)) return true;
		}
		return false;
	}
protected:
	const DVertex * m_pSuperTriangle;
};

// Function object to check whether a DTriangle is 'completed', i.e. doesn't need to be checked
// again in the algorithm, i.e. it won't be changed anymore.
// Therefore it can be removed from the workset.
// A DTriangle is completed if the circumcircle is completely to the left of the current DVertex.
// If a DTriangle is completed, it will be inserted in the output set, unless one or more of it's vertices
// belong to the 'super DTriangle'.
class DTriangleIsCompleted
{
public:
	DTriangleIsCompleted(cvIterator itVertex, DTriangleSet& output, const DVertex SuperTriangle[3])
		: m_itVertex(itVertex)
		, m_Output(output)
		, m_pSuperTriangle(SuperTriangle)
	{}
	bool operator()(const DTriangle& tri) 
	{
		bool b = tri.IsLeftOf(m_itVertex);

		if (b)
		{
			DTriangleHasVertex thv(m_pSuperTriangle);
			if (! thv(tri)) m_Output.insert(tri);
		}
		return b;
	}

protected:
	cvIterator m_itVertex;
	DTriangleSet& m_Output;
	const DVertex * m_pSuperTriangle;
};

// Function object to check whether DVertex is in circumcircle of DTriangle.
// operator() returns true if it does.
// The DEdges of a 'hot' DTriangle are stored in the DEdgeSet DEdges.
class DVertexIsInCircumCircle
{
public:
	DVertexIsInCircumCircle(cvIterator itVertex,  DEdgeSet& DEdges) : m_itVertex(itVertex), m_Edges(DEdges)	{}
	bool operator()(const DTriangle& tri) const
	{
		bool b = tri.CCEncompasses(m_itVertex);

		if (b)
		{
			HandleEdge(tri.GetVertex(0), tri.GetVertex(1));
			HandleEdge(tri.GetVertex(1), tri.GetVertex(2));
			HandleEdge(tri.GetVertex(2), tri.GetVertex(0));
		}
		return b;
	}
protected:
	void HandleEdge(DVertex * p0, DVertex * p1) const
	{
		DVertex * pVertex0(NULL);
		DVertex * pVertex1(NULL);

		// Create a normalized DEdge, in which the smallest DVertex comes first.
		if (* p0 < * p1)
		{
			pVertex0 = p0;
			pVertex1 = p1;
		}
		else
		{
			pVertex0 = p1;
			pVertex1 = p0;
		}

		DEdge e(pVertex0, pVertex1);

		// Check if this DEdge is already in the buffer
		DEdgeIterator found = m_Edges.find(e);

		if (found == m_Edges.end()) m_Edges.insert(e);		// no, it isn't, so insert
		else m_Edges.erase(found);							// yes, it is, so erase it to eliminate double DEdges
	}

	cvIterator m_itVertex;
	DEdgeSet& m_Edges;
};

void Delaunay::Triangulate(DVertexSet& vertices, DTriangleSet& output)
{
	if (vertices.size() < 3) return;	// nothing to handle

	// Determine the bounding box.
	cvIterator itVertex = vertices.begin();

	real xMin = itVertex->GetX();
	real yMin = itVertex->GetY();
	real xMax = xMin;
	real yMax = yMin;

	++itVertex;		// If we're here, we know that vertices is not empty.
	for (; itVertex != vertices.end(); itVertex++)
	{
		xMax = itVertex->GetX();	// Vertices are sorted along the x-axis, so the last one stored will be the biggest.
		real y = itVertex->GetY();
		if (y < yMin) yMin = y;
		if (y > yMax) yMax = y;
	}

	real dx = xMax - xMin;
	real dy = yMax - yMin;

	// Make the bounding box slightly bigger, just to feel safe.
	real ddx = dx * 0.01F;
	real ddy = dy * 0.01F;

	xMin -= ddx;
	xMax += ddx;
	dx += 2 * ddx;

	yMin -= ddy;
	yMax += ddy;
	dy += 2 * ddy;

	// Create a 'super DTriangle', encompassing all the vertices. We choose an equilateral DTriangle with horizontal base.
	// We could have made the 'super DTriangle' simply very big. However, the algorithm is quite sensitive to
	// rounding errors, so it's better to make the 'super DTriangle' just big enough, like we do here.
	DVertex vSuper[3];

	vSuper[0] = DVertex(xMin - dy * sqrt3 / 3.0F, yMin);	// Simple highschool geometry, believe me.
	vSuper[1] = DVertex(xMax + dy * sqrt3 / 3.0F, yMin);
	vSuper[2] = DVertex((xMin + xMax) * 0.5F, yMax + dx * sqrt3 * 0.5F);

	DTriangleSet workset;
	workset.insert(DTriangle(vSuper));

	for (itVertex = vertices.begin(); itVertex != vertices.end(); itVertex++)
	{
		// First, remove all 'completed' DTriangles from the workset.
		// A DTriangle is 'completed' if its circumcircle is entirely to the left of the current DVertex.
		// Vertices are sorted in x-direction (the set container does this automagically).
		// Unless they are part of the 'super DTriangle', copy the 'completed' DTriangles to the output.
		// The algorithm also works without this step, but it is an important optimalization for bigger numbers of vertices.
		// It makes the algorithm about five times faster for 2000 vertices, and for 10000 vertices,
		// it's thirty times faster. For smaller numbers, the difference is negligible.
		tIterator itEnd = remove_if(workset.begin(), workset.end(), DTriangleIsCompleted(itVertex, output, vSuper));

		DEdgeSet DEdges;

		// A DTriangle is 'hot' if the current DVertex v is inside the circumcircle.
		// Remove all hot DTriangles, but keep their DEdges.
		itEnd = remove_if(workset.begin(), itEnd, DVertexIsInCircumCircle(itVertex, DEdges));
		workset.erase(itEnd, workset.end());	// remove_if doesn't actually remove; we have to do this explicitly.

		// Create new DTriangles from the DEdges and the current DVertex.
		for (DEdgeIterator it = DEdges.begin(); it != DEdges.end(); it++)
			workset.insert(DTriangle(it->m_pV0, it->m_pV1, const_cast<DVertex*>(&(*itVertex))));
	}

	// Finally, remove all the DTriangles belonging to the 'super DTriangle' and move the remaining
	// DTriangles tot the output; remove_copy_if lets us do that in one go.
	tIterator where = output.begin();
	remove_copy_if(workset.begin(), workset.end(), inserter(output, where), DTriangleHasVertex(vSuper));
	//for (DTriangleSet::iterator it = workset.begin(); it != workset.end() ; it++) {
	//	DTriangleHasVertex dh(vSuper);
	//	if (!dh((*it))) output.insert((*it));
	//}
}

void Delaunay::TrianglesToEdges(DTriangleSet& DTriangles, DEdgeSet& DEdges)
{
	for (ctIterator it = DTriangles.begin(); it != DTriangles.end(); ++it)
	{
		HandleEdge(it->GetVertex(0), it->GetVertex(1), DEdges);
		HandleEdge(it->GetVertex(1), it->GetVertex(2), DEdges);
		HandleEdge(it->GetVertex(2), it->GetVertex(0), DEdges);
	}
}

void Delaunay::HandleEdge(DVertex * p0, DVertex * p1, DEdgeSet& DEdges)
{
	DVertex * pV0(NULL);
	DVertex * pV1(NULL);

	if (* p0 < * p1)
	{
		pV0 = p0;
		pV1 = p1;
	}
	else
	{
		pV0 = p1;
		pV1 = p0;
	}

	// Insert a normalized DEdge. If it's already in DEdges, insertion will fail,
	// thus leaving only unique DEdges.
	DEdges.insert(DEdge(pV0, pV1));
}

};