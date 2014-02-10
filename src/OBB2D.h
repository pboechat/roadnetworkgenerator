#ifndef OBB2D_H
#define OBB2D_H

#pragma once

#include <VectorMath.h>
#include <vector>

// compute a minimum-area oriented box containing the specified points.
// the algorithm uses the rotating calipers method.
// if the input points represent a counterclockwise-ordered polygon, set 'isConvexPolygon' to 'true' otherwise, set 'isConvexPolygon' to 'false'.
struct OBB2D
{
	vml_vec2 center;
	vml_vec2 axis[2];
	float extent[2];

	//OBB2D(const vml_vec2* hullPoints, unsigned int numHullPoints)
	OBB2D(std::vector<vml_vec2> hullPoints)
	{
		// the input points are V[0] through V[N-1] and are assumed to be the vertices of a convex polygon that are counterclockwise ordered
		// the input points must not contain three consecutive collinear points
		// unit-length edge directions of convex polygon
		// these could be precomputed and passed to this routine if the application requires it
		unsigned int numPointsM1 = hullPoints.size() - 1;
		vml_vec2* edges = new vml_vec2[hullPoints.size()];
		bool* visited = new bool[hullPoints.size()];
		unsigned int i;
		for (i = 0; i < numPointsM1; ++i)
		{
			edges[i] = hullPoints[i + 1] - hullPoints[i];
			edges[i] = vml_normalize(edges[i]);
			visited[i] = false;
		}
		edges[numPointsM1] = hullPoints[0] - hullPoints[numPointsM1];
		edges[numPointsM1] = vml_normalize(edges[numPointsM1]);
		visited[numPointsM1] = false;

		// find the smallest axis-aligned box containing the points
		// keep track of the extremity indices, L (left), R (right), B (bottom), and T (top) so that the following constraints are met:
		//   V[L].x <= V[i].x for all i and V[(L+1)%N].x > V[L].x
		//   V[R].x >= V[i].x for all i and V[(R+1)%N].x < V[R].x
		//   V[B].y <= V[i].y for all i and V[(B+1)%N].y > V[B].y
		//   V[T].y >= V[i].y for all i and V[(T+1)%N].y < V[T].y
		float xmin = hullPoints[0].x, xmax = xmin;
		float ymin = hullPoints[0].y, ymax = ymin;
		unsigned int LIndex = 0, RIndex = 0, BIndex = 0, TIndex = 0;
		for (i = 1; i < hullPoints.size(); ++i)
		{
			if (hullPoints[i].x <= xmin)
			{
				xmin = hullPoints[i].x;
				LIndex = i;
			}
			if (hullPoints[i].x >= xmax)
			{
				xmax = hullPoints[i].x;
				RIndex = i;
			}

			if (hullPoints[i].x <= ymin)
			{
				ymin = hullPoints[i].y;
				BIndex = i;
			}
			if (hullPoints[i].y >= ymax)
			{
				ymax = hullPoints[i].y;
				TIndex = i;
			}
		}

		// apply wrap-around tests to ensure the constraints mentioned above are satisfied
		if (LIndex == numPointsM1)
		{
			if (hullPoints[0].x <= xmin)
			{
				xmin = hullPoints[0].x;
				LIndex = 0;
			}
		}

		if (RIndex == numPointsM1)
		{
			if (hullPoints[0].x >= xmax)
			{
				xmax = hullPoints[0].x;
				RIndex = 0;
			}
		}

		if (BIndex == numPointsM1)
		{
			if (hullPoints[0].y <= ymin)
			{
				ymin = hullPoints[0].y;
				BIndex = 0;
			}
		}

		if (TIndex == numPointsM1)
		{
			if (hullPoints[0].y >= ymax)
			{
				ymax = hullPoints[0].y;
				TIndex = 0;
			}
		}

		// the dimensions of the axis-aligned box
		// the extents store width and height for now
		center.x = 0.5f * (xmin + xmax);
		center.y = 0.5f * (ymin + ymax);
		axis[0] = vml_vec2(1.0f, 0.0f);
		axis[1] = vml_vec2(0.0f, 1.0f);
		extent[0] = 0.5f * (xmax - xmin);
		extent[1] = 0.5f * (ymax - ymin);
		float minAreaDiv4 = extent[0] * extent[1];

		// the rotating calipers algorithm
		vml_vec2 U = vml_vec2(1.0f, 0.0f);
		vml_vec2 V = vml_vec2(0.0f, 1.0f);

		bool done = false;
		while (!done)
		{
			// determine the edge that forms the smallest angle with the current box edges
			int flag = F_NONE;
			float maxDot = 0.0f;
			float dot = vml_dot(U, edges[BIndex]);
			if (dot > maxDot)
			{
				maxDot = dot;
				flag = F_BOTTOM;
			}

			dot = vml_dot(V, edges[RIndex]);
			if (dot > maxDot)
			{
				maxDot = dot;
				flag = F_RIGHT;
			}

			dot = -vml_dot(U, edges[TIndex]);
			if (dot > maxDot)
			{
				maxDot = dot;
				flag = F_TOP;
			}

			dot = -vml_dot(V, edges[LIndex]);
			if (dot > maxDot)
			{
				maxDot = dot;
				flag = F_LEFT;
			}

			switch (flag)
			{
			case F_BOTTOM:
				if (visited[BIndex])
				{
					done = true;
				}
				else
				{
					// compute box axes with E[B] as an edge
					U = edges[BIndex];
					V = -vml_perp(U);
					updateBox(hullPoints[LIndex], hullPoints[RIndex], hullPoints[BIndex], hullPoints[TIndex], U, V, minAreaDiv4);

					// mark edge visited and rotate the calipers
					visited[BIndex] = true;
					if (++BIndex == hullPoints.size())
					{
						BIndex = 0;
					}
				}
				break;
			case F_RIGHT:
				if (visited[RIndex])
				{
					done = true;
				}
				else
				{
					// compute box axes with E[R] as an edge
					V = edges[RIndex];
					U = vml_perp(V);
					updateBox(hullPoints[LIndex], hullPoints[RIndex], hullPoints[BIndex], hullPoints[TIndex], U, V, minAreaDiv4);

					// mark edge visited and rotate the calipers
					visited[RIndex] = true;
					if (++RIndex == hullPoints.size())
					{
						RIndex = 0;
					}
				}
				break;
			case F_TOP:
				if (visited[TIndex])
				{
					done = true;
				}
				else
				{
					// compute box axes with E[T] as an edge
					U = -edges[TIndex];
					V = -vml_perp(U);
					updateBox(hullPoints[LIndex], hullPoints[RIndex], hullPoints[BIndex], hullPoints[TIndex], U, V, minAreaDiv4);

					// mark edge visited and rotate the calipers
					visited[TIndex] = true;
					if (++TIndex == hullPoints.size())
					{
						TIndex = 0;
					}
				}
				break;
			case F_LEFT:
				if (visited[LIndex])
				{
					done = true;
				}
				else
				{
					// compute box axes with E[L] as an edge
					V = -edges[LIndex];
					U = vml_perp(V);
					updateBox(hullPoints[LIndex], hullPoints[RIndex], hullPoints[BIndex], hullPoints[TIndex], U, V, minAreaDiv4);

					// mark edge visited and rotate the calipers
					visited[LIndex] = true;
					if (++LIndex == hullPoints.size())
					{
						LIndex = 0;
					}
				}
				break;
			case F_NONE:
				// the polygon is a rectangle
				done = true;
				break;
			}
		}

		delete[] visited;
		delete[] edges;
	}

private:
	// flags for the rotating calipers algorithm.
	enum { F_NONE, F_LEFT, F_RIGHT, F_BOTTOM, F_TOP };

	void updateBox(const vml_vec2& LPoint, const vml_vec2& RPoint, const vml_vec2& BPoint, const vml_vec2& TPoint, const vml_vec2& U, const vml_vec2& V, float& minAreaDiv4)
	{
		vml_vec2 RLDiff = RPoint - LPoint;
		vml_vec2 TBDiff = TPoint - BPoint;
		float extent0 = 0.5f * (vml_dot(U, RLDiff));
		float extent1 = 0.5f * (vml_dot(V, TBDiff));
		float areaDiv4 = extent0*extent1;
		if (areaDiv4 < minAreaDiv4)
		{
			minAreaDiv4 = areaDiv4;
			axis[0] = U;
			axis[1] = V;
			extent[0] = extent0;
			extent[1] = extent1;
			vml_vec2 LBDiff = LPoint - BPoint;
			center = LPoint + U*extent0 + V*(extent1 - vml_dot(V, LBDiff));
		}
	}

};

#endif