#ifndef QUADREE_H
#define QUADREE_H

#include <Defines.h>

#include <Line.h>
#include <Circle.h>
#include <AABB.h>
#include <glm/glm.hpp>

#include <exception>

namespace RoadNetworkGraph
{

struct EdgeReference
{
	EdgeIndex index;
	VertexIndex source;
	VertexIndex destination;
	glm::vec3 sourcePosition;
	glm::vec3 destinationPosition;

	EdgeReference& operator = (const EdgeReference& other)
	{
		index = other.index;
		source = other.source;
		destination = other.destination;
		sourcePosition = other.sourcePosition;
		destinationPosition = other.destinationPosition;
		return *this;
	}

};

class QuadTree
{
public:
	QuadTree(const AABB& bounds, float cellArea) : bounds(bounds), cellArea(cellArea), lastEdgeReferenceIndex(0), northWest(0), northEast(0), southWest(0), southEast(0) {}
	~QuadTree()
	{
		if (northWest != 0)
		{
			delete northWest;
		}

		if (northEast != 0)
		{
			delete northEast;
		}

		if (southWest != 0)
		{
			delete southWest;
		}

		if (southEast != 0)
		{
			delete southEast;
		}
	}

	bool insert(EdgeIndex index, VertexIndex source, VertexIndex destination, const glm::vec3& sourcePosition, const glm::vec3& destinationPosition)
	{
		if (!bounds.isIntersected(Line(sourcePosition, destinationPosition)))
		{
			return false;
		}

		if (bounds.getArea() <= cellArea)
		{
			addEdgeReference(index, source, destination, sourcePosition, destinationPosition);
			return true;
		}

		if (northWest == 0)
		{
			subdivide();
		}

		if (northWest->insert(index, source, destination, sourcePosition, destinationPosition))
		{
			return true;
		}

		if (northEast->insert(index, source, destination, sourcePosition, destinationPosition))
		{
			return true;
		}

		if (southWest->insert(index, source, destination, sourcePosition, destinationPosition))
		{
			return true;
		}

		if (southEast->insert(index, source, destination, sourcePosition, destinationPosition))
		{
			return true;
		}

		// FIXME: should never happen!
		throw std::exception("couldn't insert point");
	}

	void subdivide()
	{
		float halfWidth = bounds.getExtents().x / 2.0f;
		float halfHeight = bounds.getExtents().y / 2.0f;
		northWest = new QuadTree(AABB(bounds.min.x, bounds.min.y + halfHeight, halfWidth, halfHeight), cellArea);
		northEast = new QuadTree(AABB(bounds.min.x + halfWidth, bounds.min.y + halfHeight, halfWidth, halfHeight), cellArea);
		southWest = new QuadTree(AABB(bounds.min.x, bounds.min.y, halfWidth, halfHeight), cellArea);
		southEast = new QuadTree(AABB(bounds.min.x + halfWidth, bounds.min.y, halfWidth, halfHeight), cellArea);
	}

	/*void query(const AABB& aabb, EdgeReference* edgeReferences, unsigned int& size, unsigned int offset = 0) const
	{
		size = 0;
		query_(aabb, edgeReferences, size, offset);
	}*/

	void query(const Circle& circle, EdgeReference* edgeReferences, unsigned int& size, unsigned int offset = 0) const
	{
		size = 0;
		query_(circle, edgeReferences, size, offset);
	}

	inline bool isLeaf() const
	{
		return northEast == 0;
	}

	inline bool notEmpty() const
	{
		return lastEdgeReferenceIndex > 0;
	}

	inline const QuadTree* getNorthWest() const
	{
		return northWest;
	}

	inline const QuadTree* getNorthEast() const
	{
		return northEast;
	}

	inline const QuadTree* getSouthWest() const
	{
		return southWest;
	}

	inline const QuadTree* getSouthEast() const
	{
		return southEast;
	}

	inline const AABB& getBounds() const
	{
		return bounds;
	}

private:
	AABB bounds;
	float cellArea;
	QuadTree* northWest;
	QuadTree* northEast;
	QuadTree* southWest;
	QuadTree* southEast;
	EdgeReference edgeReferences[MAX_VERTICES];
	unsigned int lastEdgeReferenceIndex;

	/*void query_(const AABB& aabb, EdgeReference* edgeReferences, unsigned int& size, unsigned int offset) const
	{
		if ((offset + size) == MAX_EDGE_REFERENCIES_PER_QUERY)
		{
			return;
		}

		if (!bounds.intersects(aabb))
		{
			return;
		}

		if (isLeaf())
		{
			for (unsigned int i = 0; i < lastEdgeReferenceIndex; i++)
			{
				const EdgeReference& edgeReference = this->edgeReferences[i];

				if (Line(edgeReference.sourcePosition, edgeReference.destinationPosition).intersects(aabb))
				{
					edgeReferences[offset + size] = edgeReference;
					size++;
					if ((offset + size) == MAX_EDGE_REFERENCIES_PER_QUERY)
					{
						return;
					}
				}
			}
		}

		else
		{
			northWest->query_(aabb, edgeReferences, size, offset);
			northEast->query_(aabb, edgeReferences, size, offset);
			southWest->query_(aabb, edgeReferences, size, offset);
			southEast->query_(aabb, edgeReferences, size, offset);
		}
	}*/

	void query_(const Circle& circle, EdgeReference* edgeReferences, unsigned int& size, unsigned int offset) const
	{
		if ((offset + size) == MAX_EDGE_REFERENCIES_PER_QUERY)
		{
			return;
		}

		if (!bounds.intersects(circle))
		{
			return;
		}

		if (isLeaf())
		{
			for (unsigned int i = 0; i < lastEdgeReferenceIndex; i++)
			{
				const EdgeReference& edgeReference = this->edgeReferences[i];

				if (Line(edgeReference.sourcePosition, edgeReference.destinationPosition).intersects(circle))
				{
					edgeReferences[offset + size] = edgeReference;
					size++;
					if ((offset + size) == MAX_EDGE_REFERENCIES_PER_QUERY)
					{
						return;
					}
				}
			}
		}

		else
		{
			northWest->query_(circle, edgeReferences, size, offset);
			northEast->query_(circle, edgeReferences, size, offset);
			southWest->query_(circle, edgeReferences, size, offset);
			southEast->query_(circle, edgeReferences, size, offset);
		}
	}

	void addEdgeReference(EdgeIndex edgeIndex, VertexIndex source, VertexIndex destination, const glm::vec3& sourcePosition, const glm::vec3& destinationPosition) 
	{
		EdgeReference& edgeReference = edgeReferences[lastEdgeReferenceIndex++];
		edgeReference.index = edgeIndex;
		edgeReference.source = source;
		edgeReference.destination = destination;
		edgeReference.sourcePosition = sourcePosition;
		edgeReference.destinationPosition = destinationPosition;
	}

};

}

#endif