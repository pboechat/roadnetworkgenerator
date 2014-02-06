#ifndef CUDATEXTURE2DMOCK_H
#define CUDATEXTURE2DMOCK_H

struct CudaTexture2DMock
{
	unsigned int width;
	unsigned int height;
	const unsigned char* data;

};

#endif