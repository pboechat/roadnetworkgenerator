#ifndef CONVEXHULL_H
#define CONVEXHULL_H

#pragma once

#include <MathExtras.h>
#include <VectorMath.h>

#include <exception>

#define ccw(pt1, pt2, pt3) ((pt2.x - pt1.x) * (pt3.y - pt1.y) - (pt2.y - pt1.y) * (pt3.x - pt1.x))

/***********************************************************************************************************/
/* Source: http://stackoverflow.com/questions/11107687/implementing-graham-scan-to-find-the-convex-hull    */
/***********************************************************************************************************/
struct ConvexHull
{
	vml_vec2* hullPoints;
	unsigned int numHullPoints;

	ConvexHull(const vml_vec2* vertices, unsigned int numVertices)
	{
		// FIXME: checking invariants
		if (numVertices < 3)
		{
			throw std::exception("numVertices < 3");
		}

		vml_vec2* tmpHullPoints = new vml_vec2[numVertices + 1];
		vml_vec2 tmpPoint = vertices[0];
		float tmpAngle = 0;
		int tmpK = 0;
		for (unsigned int i = 1; i < numVertices; i++)
		{
			if (vertices[i].y < tmpPoint.y)
			{
				tmpHullPoints[i + 1] = tmpPoint;
				tmpPoint = vertices[i];
			}
			else
			{
				tmpHullPoints[i + 1] = vertices[i];
			}
		}

		tmpHullPoints[1] = tmpPoint;
		for (unsigned int i = 2; i <= numVertices; i++)
		{
			tmpPoint = tmpHullPoints[i];
			tmpAngle = vml_dot(tmpHullPoints[1], tmpPoint);
			tmpK = i;
			for (unsigned int j = 1; j <= numVertices - i; j++)
			{
				if (vml_dot(tmpHullPoints[1], tmpHullPoints[i + j]) > tmpAngle)
				{
					tmpPoint = tmpHullPoints[i+j];
					tmpAngle = vml_dot(tmpHullPoints[1], tmpHullPoints[i + j]);
					tmpK = i + j;
				}
			}
			tmpHullPoints[tmpK] = tmpHullPoints[i];
			tmpHullPoints[i] = tmpPoint;
		}
		tmpHullPoints[0] = tmpHullPoints[numVertices];

		numHullPoints = 1;
		for (unsigned int i = 2; i <= numVertices; i++)
		{
			while (ccw(tmpHullPoints[numHullPoints - 1], tmpHullPoints[numHullPoints], tmpHullPoints[i]) <= 0)
			{
				if (numHullPoints > 1)
				{
					numHullPoints--;
				}
				else if (i == numVertices)
				{
					break;
				}
				else
				{
					i += 1;
				}
			}
			numHullPoints++;
			tmpPoint = tmpHullPoints[numHullPoints];
			tmpHullPoints[numHullPoints] = tmpHullPoints[i];
			tmpHullPoints[i] = tmpHullPoints[numHullPoints];
		}

		hullPoints = new vml_vec2[numHullPoints + 1];
		for (unsigned int i = 0; i < numHullPoints; i++)
		{
			hullPoints[i] = tmpHullPoints[i + 1];
		}
		hullPoints[numHullPoints] = tmpHullPoints[1];

		delete[] tmpHullPoints;
	}
	
	~ConvexHull()
	{
		if (hullPoints != 0)
		{
			delete[] hullPoints;
		}
	}

};

#endif