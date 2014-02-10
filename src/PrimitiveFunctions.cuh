#ifndef PRIMITIVEFUNCTIONS_CUH
#define PRIMITIVEFUNCTIONS_CUH

#include <BaseGraph.h>
#include <Primitive.h>
#include <MathExtras.h>
#include <VectorMath.h>
#include <Line2D.h>
#include <GraphFunctions.cuh>
#include <BaseGraphFunctions.cuh>

#include <algorithm>
#include <iterator>
#include <list>
#include <exception>

//////////////////////////////////////////////////////////////////////////
struct Triangle
{
	int v1, v2, v3;

	Triangle(int v1, int v2, int v3) : v1(v1), v2(v2), v3(v3) {}

	bool operator == (const Triangle& other) const
	{
		return v1 == other.v1 && v2 == other.v2 && v3 == other.v3;
	}

	bool operator != (const Triangle& other) const
	{
		return *this != other;
	}

};

//////////////////////////////////////////////////////////////////////////
struct PolygonSorter
{
	PolygonSorter(const std::vector<int>& originalPolygon) : originalPolygon(originalPolygon) {}

	bool operator()(int a, int b)
	{
		std::vector<int>::const_iterator it1 = std::find(originalPolygon.begin(), originalPolygon.end(), a);
		unsigned int posA = std::distance(originalPolygon.begin(), it1);
		it1 = std::find(originalPolygon.begin(), originalPolygon.end(), b);
		unsigned int posB = std::distance(originalPolygon.begin(), it1);
		return posA < posB;
	}

private:
	const std::vector<int>& originalPolygon;

};

//////////////////////////////////////////////////////////////////////////
bool isConvex(const BaseGraph* graph, Primitive& primitive)
{
	for (unsigned int i = 0, j = primitive.numVertices - 1 ; i < primitive.numVertices - 1; j = i++) 
	{
		vml_vec2 p1 = graph->vertices[primitive.vertices[j]].getPosition();
		vml_vec2 p2 = graph->vertices[primitive.vertices[i]].getPosition();
		vml_vec2 p3 = graph->vertices[primitive.vertices[i + 1]].getPosition();

		vml_vec2 e1 = p1 - p2;
		vml_vec2 e2 = p3 - p2;

		if (vml_dot_perp(e1, e2) > 0)
		{
			return false;
		}
	}
	return true;
}

//////////////////////////////////////////////////////////////////////////
bool isConvex(const BaseGraph* graph, const std::vector<int>& polygon)
{
	for (unsigned int i = 0, j = polygon.size() - 1 ; i < polygon.size() - 1; j = i++) 
	{
		vml_vec2 p1 = graph->vertices[polygon[j]].getPosition();
		vml_vec2 p2 = graph->vertices[polygon[i]].getPosition();
		vml_vec2 p3 = graph->vertices[polygon[i + 1]].getPosition();

		vml_vec2 e1 = p1 - p2;
		vml_vec2 e2 = p3 - p2;

		if (vml_dot_perp(e1, e2) > 0)
		{
			return false;
		}
	}
	return true;
}

//////////////////////////////////////////////////////////////////////////
void getAreaAndCentroid(const BaseGraph* graph, const Primitive& primitive, float& area, vml_vec2& centroid)
{
	float twiceArea = 0, x = 0, y = 0, f = 0;
	for (unsigned int i = 0, j = primitive.numVertices - 1 ; i < primitive.numVertices; j = i++) 
	{
		vml_vec2 p1 = graph->vertices[primitive.vertices[i]].getPosition(); 
		vml_vec2 p2 = graph->vertices[primitive.vertices[j]].getPosition();
		f = p1.x * p2.y - p2.x * p1.y;
		twiceArea += f;
		x += (p1.x + p2.x) * f;
		y += (p1.y + p2.y) * f;
	}
	area = abs(twiceArea * 0.5f);
	f = twiceArea * 3.0f;
	centroid = vml_vec2(x / f, y / f);
}

//////////////////////////////////////////////////////////////////////////
unsigned int appendTriangle(std::vector<int>& polygon, const Triangle& triangle, std::vector<int>& originalPolygon)
{
	unsigned int mask = 0;

	if (std::find(polygon.begin(), polygon.end(), triangle.v1) == polygon.end())
	{
		polygon.push_back(triangle.v1);
		mask |= 1;
	}

	if (std::find(polygon.begin(), polygon.end(), triangle.v2) == polygon.end())
	{
		polygon.push_back(triangle.v2);
		mask |= 2;
	}

	if (std::find(polygon.begin(), polygon.end(), triangle.v3) == polygon.end())
	{
		polygon.push_back(triangle.v3);
		mask |= 4;
	}

	std::sort(polygon.begin(), polygon.end(), PolygonSorter(originalPolygon));

	return mask;
}

//////////////////////////////////////////////////////////////////////////
void removeTriangle(std::vector<int>& polygon, const Triangle& triangle, std::vector<int>& originalPolygon, unsigned int mask)
{
	std::vector<int>::iterator it1;
	if ((mask & 1) != 0)
	{
		it1 = std::find(polygon.begin(), polygon.end(), triangle.v1);

		// FIXME: checking invariants
		if (it1 == polygon.end())
		{
			throw std::exception("it1 == polygon.end()");
		}

		polygon.erase(it1);
	}

	if ((mask & 2) != 0)
	{
		it1 = std::find(polygon.begin(), polygon.end(), triangle.v2);

		// FIXME: checking invariants
		if (it1 == polygon.end())
		{
			throw std::exception("it1 == polygon.end()");
		}

		polygon.erase(it1);
	}

	if ((mask & 4) != 0)
	{
		it1 = std::find(polygon.begin(), polygon.end(), triangle.v3);

		// FIXME: checking invariants
		if (it1 == polygon.end())
		{
			throw std::exception("it1 == polygon.end()");
		}

		polygon.erase(it1);
	}

	std::sort(polygon.begin(), polygon.end(), PolygonSorter(originalPolygon));
}

//////////////////////////////////////////////////////////////////////////
void consolidadeEdges(Graph* graph, Primitive& primitive)
{
	for (unsigned int i = 0, j = primitive.numVertices - 1; i < primitive.numVertices; j = i++)
	{
		int edgeIndex = findEdge(graph, primitive.vertices[i], primitive.vertices[j]);
		if (edgeIndex == -1)
		{
			edgeIndex = connect(graph, primitive.vertices[i], primitive.vertices[j], 1);

			if (edgeIndex == -1)
			{
				// FIXME: checking invariants
				throw std::exception("connect(..) == -1");
			}
		}

		insertEdge(primitive, edgeIndex);
	}
}

//////////////////////////////////////////////////////////////////////////
bool isInsideTriangle(const BaseGraph* graph, const vml_vec2& A, const vml_vec2& B, const vml_vec2& C, const vml_vec2& P)
{
	//compute vectors        
	vml_vec2 v0 = C - A;
	vml_vec2 v1 = B - A;
	vml_vec2 v2 = P - A;

	// compute dot products
	float dot00 = vml_dot(v0, v0);
	float dot01 = vml_dot(v0, v1);
	float dot02 = vml_dot(v0, v2);
	float dot11 = vml_dot(v1, v1);
	float dot12 = vml_dot(v1, v2);

	// compute barycentric coordinates
	float invDenom = 1.0f / (dot00 * dot11 - dot01 * dot01);
	float u = (dot11 * dot02 - dot01 * dot12) * invDenom;
	float v = (dot00 * dot12 - dot01 * dot02) * invDenom;

	// check if point is in triangle
	return (u >= 0) && (v >= 0) && (u + v < 1);
}

//////////////////////////////////////////////////////////////////////////
// TODO: optimize
void clipEars(const BaseGraph* graph, const Primitive& primitive, std::list<Triangle>& triangles)
{
	// FIXME: checking invariants
	if (primitive.numVertices < 4)
	{
		throw std::exception("must be a polygon with 4 or more edges");
	}

	// copying vertices to a linked list to reduce remove cost to O(1)
	// note: "std::list" is implemented as a doubly-linked list
	std::list<int> vertices;
	for (unsigned int i = 0; i < primitive.numVertices; i++)
	{
		vertices.push_back(primitive.vertices[i]);
	}

	std::list<int>::iterator current = vertices.begin();
	while (vertices.size() > 2)
	{
		// FIXME: checking invariants
		if (current == vertices.end())
		{
			current = vertices.begin();
		}

		int i1 = *current;

		std::list<int>::iterator next = current;
		next++;
		if (next == vertices.end())
		{
			next = vertices.begin();
		}
		int i2 = *next;

		next++;
		if (next == vertices.end())
		{
			next = vertices.begin();
		}
		int i3 = *next;

		vml_vec2 v1 = graph->vertices[i1].getPosition();
		vml_vec2 v2 = graph->vertices[i2].getPosition();
		vml_vec2 v3 = graph->vertices[i3].getPosition();

		// checking if p2 is convex

		vml_vec2 e1 = v1 - v2;
		vml_vec2 e2 = v3 - v2;

		if (vml_dot_perp(e1, e2) >= 0)
		{
			// if v2 is concave (or v1v2v3 are colinear), move to next point
			current++;
			continue;
		}

		// check if triangle v1v2v3 contains no concave vertex inside it

		bool isEar = true;
		std::list<int>::iterator it1 = vertices.begin();
		while (it1 != vertices.end())
		{
			// for every point v5 that doesn't belong to the triangle, check if it's inside the triangle

			int i5 = *it1;

			if (i5 == i1 || i5 == i2 || i5 == i3)
			{
				it1++;
				continue;
			}

			vml_vec2 v5 = graph->vertices[i5].getPosition();

			if (isInsideTriangle(graph, v1, v2, v3, v5))
			{
				isEar = false;
				break;
			}

			it1++;
		} // end while

		if (isEar)
		{
			vertices.remove(i2);
			triangles.push_back(Triangle(i1, i2, i3));
		}

		else
		{
			current++;
		}
	} // end for
}

#endif