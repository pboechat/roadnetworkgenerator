#ifndef GEOMETRY_H
#define GEOMETRY_H

#pragma once

class Geometry
{
protected:
	Geometry() {}
	Geometry(const Geometry&) {}
	Geometry& operator =(Geometry&)
	{
		return *this;
	}
	~Geometry() {}

public:
	virtual void draw() = 0;

};

#endif