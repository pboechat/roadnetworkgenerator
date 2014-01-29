#ifndef IMAGEUTILS_H
#define IMAGEUTILS_H

#include <FreeImage.h>

#include <exception>

class ImageUtils
{
public:
	//////////////////////////////////////////////////////////////////////////
	static void loadImage(const char* filePath, int width, int height, unsigned char* data)
	{
		FREE_IMAGE_FORMAT format = FreeImage_GetFileType(filePath, 0);
		FIBITMAP* bitmap = FreeImage_Load(format, filePath);
		FIBITMAP* image = FreeImage_ConvertTo32Bits(bitmap);
		int imageWidth = FreeImage_GetWidth(image);
		int imageHeight = FreeImage_GetHeight(image);

		if (imageWidth != width || imageHeight != height)
		{
			image = FreeImage_Rescale(image, width, height, FILTER_BOX);
		}

		int size = width * height;
		unsigned char* bgra = (unsigned char*)FreeImage_GetBits(image);

		if (bgra == 0)
		{
			throw std::exception("invalid image file");
		}

		for (int i = 0, j = 0; i < size; i++, j += 4)
		{
			// grayscale = (0.21 R + 0.71 G + 0.07 B)
			data[i] = (unsigned char)(0.21f * bgra[j + 2] + 0.71f * bgra[j + 1] + 0.07f * bgra[j]);
		}

		FreeImage_Unload(image);
	}

};

#endif