#ifndef QUADTREE_H
#define QUADTREE_H

#include <AABB.h>
#include <Line.h>
#include <Circle.h>

#include <vector>
#include <exception>

class QuadTree
{
public:
	QuadTree(const AABB& bounds, float smallestArea) : bounds(bounds), smallestArea(smallestArea), northWest(0), northEast(0), southWest(0), southEast(0) {}
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

	bool insert(const Line& line)
	{
		if (!bounds.contains(line.start) && !bounds.contains(line.end))
		{
			return false;
		}

		if (bounds.area() <= smallestArea)
		{
			lines.push_back(line);
			return true;
		}

		if (northWest == 0)
		{
			subdivide();
		}

		if (northWest->insert(line))
		{
			return true;
		}

		if (northEast->insert(line))
		{
			return true;
		}

		if (southWest->insert(line))
		{
			return true;
		}

		if (southEast->insert(line))
		{
			return true;
		}

		// FIXME: should never happen!
		throw std::exception("couldn't insert segment");
	}

	void subdivide()
	{
		float halfWidth = bounds.extents().x / 2.0f;
		float halfHeight = bounds.extents().y / 2.0f;
		northWest = new QuadTree(AABB(bounds.min.x, bounds.min.y + halfHeight, halfWidth, halfHeight), smallestArea);
		northEast = new QuadTree(AABB(bounds.min.x + halfWidth, bounds.min.y + halfHeight, halfWidth, halfHeight), smallestArea);
		southWest = new QuadTree(AABB(bounds.min.x, bounds.min.y, halfWidth, halfHeight), smallestArea);
		southEast = new QuadTree(AABB(bounds.min.x + halfWidth, bounds.min.y, halfWidth, halfHeight), smallestArea);
	}

	void query(const AABB& area, std::vector<Line>& result) const
	{
		if (isLeaf())
		{
			for (unsigned int i = 0; i < lines.size(); i++)
			{
				if (area.contains(lines[i].start) || area.contains(lines[i].end))
				{
					result.push_back(lines[i]);
				}
			}
		}
		else
		{
			northWest->query(area, result);
			northEast->query(area, result);
			southWest->query(area, result);
			southEast->query(area, result);
		}
	}

	void query(const Circle& circle, std::vector<Line>& result) const
	{
		if (!bounds.intersects(circle))
		{
			return;
		}

		if (isLeaf())
		{
			for (unsigned int i = 0; i < lines.size(); i++)
			{
				const Line& line = lines[i];

				if (line.intersects(circle))
				{
					result.push_back(line);
				}
			}
		}
		else
		{
			northWest->query(circle, result);
			northEast->query(circle, result);
			southWest->query(circle, result);
			southEast->query(circle, result);
		}
	}

	inline bool isLeaf() const
	{
		return northEast == 0;
	}

	inline bool hasLines() const
	{
		return lines.size() > 0;
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
	float smallestArea;
	QuadTree* northWest;
	QuadTree* northEast;
	QuadTree* southWest;
	QuadTree* southEast;
	std::vector<Line> lines;

};

#endif