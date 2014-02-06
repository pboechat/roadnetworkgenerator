#ifndef IMAGEMAP_H
#define IMAGEMAP_H

#pragma once

#include <CpuGpuCompatibility.h>

struct ImageMap
{
	unsigned int width;
	unsigned int height;
	const unsigned char* data;

	HOST_AND_DEVICE_CODE ImageMap() : width(0), height(0), data(0) {}
	HOST_AND_DEVICE_CODE ~ImageMap() {}

};

#endif